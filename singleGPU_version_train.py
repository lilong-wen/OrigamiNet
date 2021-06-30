import os
import sys
import time
import random
import string
import argparse
from collections import namedtuple
import copy

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
from torch import autograd
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.nn.parallel import DistributedDataParallel as pDDP

from torchsummary import summary
from torchvision.utils import save_image
import horovod.torch as hvd
import gin

import numpy as np
from tqdm import tqdm, trange
from PIL import Image

import ds_load

from utils import CTCLabelConverter, Averager, ModelEma, Metric, random_select_txt_snippets, gt_txt_sim
from cnv_model_r import OrigamiNet_extended, ginM
# from cnv_model import OrigamiNet, ginM
from test import validation

from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gInit(opt):

    gin.parse_config_file(opt.gin)

    cudnn.benchmark = True

def rSeed(sd):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed(sd)

def init_bn(model):
    if type(model) in [torch.nn.InstanceNorm2d, torch.nn.BatchNorm2d]:
        init.ones_(model.weight)
        init.zeros_(model.bias)

    elif type(model) in [torch.nn.Conv2d]:
        init.kaiming_uniform_(model.weight)

@gin.configurable
def train(opt, AMP, WdB, train_data_path, train_data_list, test_data_path, test_data_list, experiment_name,
            train_batch_size, val_batch_size, workers, lr, valInterval, num_iter, wdbprj, bert_base_model, continue_model=''):


    os.makedirs(f'./saved_models/{experiment_name}', exist_ok=True)

    train_dataset = ds_load.myLoadDS(train_data_list, train_data_path)
    valid_dataset = ds_load.myLoadDS(test_data_list, test_data_path , ralph=train_dataset.ralph)

    tokenizer = AutoTokenizer.from_pretrained(bert_base_model, cache_dir='./cached')#, do_lower_case=config['model_bert']['do_lower_case'])

    print('Alphabet :',len(train_dataset.alph),train_dataset.alph)
    for d in [train_dataset, valid_dataset]:
        print('Dataset Size :',len(d.fns))
        print('Max LbW : ',max(list(map(len,d.tlbls))) )
        print('#Chars : ',sum([len(x) for x in d.tlbls]))
        print('Sample label :',d.tlbls[-1])
        print("Dataset :", sorted(list(map(len,d.tlbls))) )
        print('-'*80)

    if opt.num_gpu > 1:
        workers = workers * ( opt.num_gpu )

    train_loader  = torch.utils.data.DataLoader( train_dataset, batch_size=train_batch_size, shuffle=True,
                    pin_memory = True, num_workers = int(workers),
                    sampler =  None,
                    worker_init_fn = WrkSeeder,
                    collate_fn = ds_load.SameTrCollate
                )

    valid_loader  = torch.utils.data.DataLoader( valid_dataset, batch_size=val_batch_size , pin_memory=True,
                    num_workers = int(workers), sampler= None)

    # model = OrigamiNet()
    model = OrigamiNet_extended()
    model.apply(init_bn)
    model.train()

    if OnceExecWorker: import pprint;[print(k,model.layer_resize[k]) for k in sorted(model.layer_resize.keys())]

    biparams    = list(dict(filter(lambda kv: 'bias'     in kv[0], model.named_parameters())).values())
    nonbiparams = list(dict(filter(lambda kv: 'bias' not in kv[0], model.named_parameters())).values())

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=10**(-1/90000))

    model_ema = ModelEma(model)

    if continue_model != '':
        print(f'loading pretrained model from {continue_model}')
        checkpoint = torch.load(continue_model)
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        model_ema._load_checkpoint(continue_model)

    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True).to(device)
    criterion_sim = torch.nn.MSELoss(reduce=False, size_average=False)
    converter = CTCLabelConverter(train_dataset.ralph.values())

    with open(f'./saved_models/{experiment_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        opt_log += gin.operative_config_str()
        opt_file.write(opt_log)

    print(optimizer)
    print(opt_log)

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = 1e+6
    best_CER = 1e+6
    i = 0
    gAcc = 1
    epoch = 1
    btReplay = False and AMP
    max_batch_replays = 3

    titer = iter(train_loader)

    while(True):
        start_time = time.time()

        model.zero_grad()

        for j in trange(valInterval, leave=False, desc='Training'):

            try:

                image_tensors, labels = next(titer)

            except StopIteration:
                epoch += 1
                titer = iter(train_loader)
                image_tensors, labels = next(titer)

            image = image_tensors.to(device)
            text, length = converter.encode(labels)
            labels_snippets = random_select_txt_snippets(labels)
            text_snippets, length_snippets = converter.encode(labels_snippets)

            gt_sim = gt_txt_sim(text, length, text_snippets, length_snippets)
            gt_sim = torch.FloatTensor(gt_sim)

            labels_input = tokenizer(labels_snippets,
                                     return_tensors="pt",
                                     padding=True,
                                     truncation=False).to(device)

            batch_size = image.size(0)

            replay_batch = True
            maxR = 3
            while replay_batch and maxR>0:
                maxR -= 1

                preds, sim_value = model(image,labels_input)
                preds = preds.float()

                preds_size = torch.IntTensor([preds.size(1)] * batch_size).to(device)
                preds = preds.permute(1, 0, 2).log_softmax(2)

                print('Model inp : ',image.dtype,image.size())
                print('CTC inp : ',preds.dtype,preds.size(),preds_size[0])

                torch.backends.cudnn.enabled = False

                cost_ctc = criterion(preds, text.to(device), preds_size, length.to(device)).mean() / gAcc
                gt_sim = torch.FloatTensor(gt_sim)
                cost_sim = criterion_sim(sim_value.to(device), gt_sim.to(device)).sum()

                if i >= 50000:
                    cost = cost_ctc + 1000 * cost_sim
                else:
                    cost = cost_ctc + 100 * cost_sim

                torch.backends.cudnn.enabled = True

                optimizer.zero_grad()
                default_optimizer_step = optimizer.step  # added for batch replay

                cost.backward()
                replay_batch = False


            if (i+1) % gAcc == 0:

                optimizer.step()

                model.zero_grad()
                model_ema.update(model, num_updates=i/2)

                if (i+1) % (gAcc*2) == 0:
                    lr_scheduler.step()

            i += 1

        # validation part
        if True:

            elapsed_time = time.time() - start_time
            start_time = time.time()

            model.eval()
            with torch.no_grad():

                valid_loss, valid_sim_loss, current_accuracy, current_norm_ED, ted, bleu, preds, labels, infer_time = validation_single(
                    model_ema.ema, criterion, criterion_sim, valid_loader, converter, opt, bert_base_model)

            model.train()
            v_time = time.time() - start_time

            if current_norm_ED < best_norm_ED:
                best_norm_ED = current_norm_ED
                checkpoint = {
                    'model': model.state_dict(),
                    'state_dict_ema': model_ema.ema.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, f'./saved_models/{experiment_name}/best_norm_ED.pth')

            if ted < best_CER:
                best_CER = ted

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy

            out  = f'[{i}] Loss: {train_loss.avg:0.5f} time: ({elapsed_time:0.1f},{v_time:0.1f})'
            out += f' vloss: {valid_loss:0.3f}'
            out += f' sim_vloss: {valid_sim_loss:0.3f}'
            out += f' CER: {ted:0.4f} NER: {current_norm_ED:0.4f} lr: {lr_scheduler.get_lr()[0]:0.5f}'
            out += f' bAcc: {best_accuracy:0.1f}, bNER: {best_norm_ED:0.4f}, bCER: {best_CER:0.4f}, B: {bleu*100:0.2f}'
            print(out)

            with open(f'./saved_models/{experiment_name}/log_train.txt', 'a') as log: log.write(out + '\n')


        if i == num_iter:
            print('end the training')
            sys.exit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gin', default='iam/iam_r.gin', help='Gin config file')

    opt = parser.parse_args()
    gInit(opt)
    opt.manualSeed = ginM('manualSeed')

    rSeed(opt.manualSeed)

    opt.num_gpu = torch.cuda.device_count()

    train(opt)
