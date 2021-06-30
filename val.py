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

import apex
from apex.parallel import DistributedDataParallel as aDDP
from apex.fp16_utils import *
from apex import amp
from apex.multi_tensor_apply import multi_tensor_applier

import wandb
import ds_load

from utils import CTCLabelConverter, Averager, ModelEma, Metric, random_select_txt_snippets, gt_txt_sim
from cnv_model import OrigamiNet, ginM
from test import validation

from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parOptions = namedtuple('parOptions', ['DP', 'DDP', 'HVD'])
parOptions.__new__.__defaults__ = (False,) * len(parOptions._fields)

pO = None
OnceExecWorker = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_bn(model):
    if type(model) in [torch.nn.InstanceNorm2d, torch.nn.BatchNorm2d]:
        init.ones_(model.weight)
        init.zeros_(model.bias)

    elif type(model) in [torch.nn.Conv2d]:
        init.kaiming_uniform_(model.weight)

def WrkSeeder(_):
    return np.random.seed((torch.initial_seed()) % (2 ** 32))


@gin.configurable
def train(opt, AMP, WdB, train_data_path, train_data_list, test_data_path, test_data_list, experiment_name,
            train_batch_size, val_batch_size, workers, lr, valInterval, num_iter, wdbprj, bert_base_model, continue_model=''):

    HVD3P = pO.HVD or pO.DDP

    os.makedirs(f'./saved_models/{experiment_name}', exist_ok=True)

    if OnceExecWorker and WdB:
        wandb.init(project=wdbprj, name=experiment_name)
        wandb.config.update(opt)

    train_dataset = ds_load.myLoadDS(train_data_list, train_data_path)
    valid_dataset = ds_load.myLoadDS(test_data_list, test_data_path , ralph=train_dataset.ralph)

    tokenizer = AutoTokenizer.from_pretrained(bert_base_model, cache_dir='./cached')#, do_lower_case=config['model_bert']['do_lower_case'])

    if OnceExecWorker:
        print(pO)
        print('Alphabet :',len(train_dataset.alph),train_dataset.alph)
        for d in [train_dataset, valid_dataset]:
            print('Dataset Size :',len(d.fns))
            print('Max LbW : ',max(list(map(len,d.tlbls))) )
            print('#Chars : ',sum([len(x) for x in d.tlbls]))
            print('Sample label :',d.tlbls[-1])
            print("Dataset :", sorted(list(map(len,d.tlbls))) )
            print('-'*80)

    if opt.num_gpu > 1:
        workers = workers * ( 1 if HVD3P else opt.num_gpu )

    if HVD3P:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=opt.world_size, rank=opt.rank)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, num_replicas=opt.world_size, rank=opt.rank)

    train_loader  = torch.utils.data.DataLoader( train_dataset, batch_size=train_batch_size, shuffle=True if not HVD3P else False,
                    pin_memory = True, num_workers = int(workers),
                    sampler = train_sampler if HVD3P else None,
                    worker_init_fn = WrkSeeder,
                    collate_fn = ds_load.SameTrCollate
                )

    valid_loader  = torch.utils.data.DataLoader( valid_dataset, batch_size=val_batch_size , pin_memory=True,
                    num_workers = int(workers), sampler=valid_sampler if HVD3P else None)

    model = OrigamiNet()
    # print(model)
    #
    # for name, param in model.named_parameters():
    #  if param.requires_grad:
    #      print(name)
    #
    # exit()
    model.apply(init_bn)
    # model.train()

    if OnceExecWorker: import pprint;[print(k,model.lreszs[k]) for k in sorted(model.lreszs.keys())]

    biparams    = list(dict(filter(lambda kv: 'bias'     in kv[0], model.named_parameters())).values())
    nonbiparams = list(dict(filter(lambda kv: 'bias' not in kv[0], model.named_parameters())).values())

    if not pO.DDP:
        model = model.to(device)
    else:
        model.cuda(opt.rank)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=10**(-1/90000))

    if OnceExecWorker and WdB:
        wandb.watch(model, log="all")

    if pO.HVD:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
        # optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), compression=hvd.Compression.fp16)

    if pO.DDP and opt.rank!=0:
        random.seed()
        np.random.seed()

    if AMP:
        model, optimizer = amp.initialize(model, optimizer, opt_level = "O1")
    if pO.DP:
        model = torch.nn.DataParallel(model)
    elif pO.DDP:
        model = pDDP(model, device_ids=[opt.rank], output_device=opt.rank,find_unused_parameters=True)



    model_ema = ModelEma(model)

    if continue_model != '':
        if OnceExecWorker: print(f'loading pretrained model from {continue_model}')
        checkpoint = torch.load(continue_model, map_location=f'cuda:{opt.rank}' if HVD3P else None)
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        model_ema._load_checkpoint(continue_model, f'cuda:{opt.rank}' if HVD3P else None)

    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True).to(device)
    criterion_sim = torch.nn.MSELoss(reduce=False, size_average=False)
    converter = CTCLabelConverter(train_dataset.ralph.values())

    labels_snippets = """At first sight the difference between souter and the other texts is vather large. But the British text incldes the paragraph P.3-S. 1l, the Woman taken in Atdultery, and this accounts For 178 words aut of 19, which is the difference betneen the 15, bs- words of Souter's rext and the 13. Wlb of Nestle's. The omision or inclusion of this paragraph is a matter of editional decision rathe"""
    labels_input = tokenizer(labels_snippets,
                             return_tensors="pt",
                             padding=True,
                             truncation=False).to('cuda')
    min_sim = 10
    hit_str = ''
    model.eval()

    for i, (image_tensors, labels) in enumerate(valid_loader):

        image = image_tensors.to('cuda')

        with torch.no_grad():

            preds, sim_value = model_ema.ema(image, labels_input)
            # preds, sim_value = model(image, labels_input)

            preds = preds.float()
            preds_size = torch.IntTensor([preds.size(1)] * 1)
            preds = preds.permute(1, 0, 2).log_softmax(2)  # to use CTCloss format
            _, preds_index = preds.max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)
            print(sim_value)
            if sim_value < min_sim:
                min_sim = sim_value
                hit_str = preds_str

    print(8*"*")
    print(min_sim, hit_str)

    return min_sim, hit_str

def gInit(opt):
    global pO, OnceExecWorker
    gin.parse_config_file(opt.gin)
    pO = parOptions(**{ginM('dist'):True})

    if pO.HVD:
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())

    OnceExecWorker = (pO.HVD and hvd.rank() == 0) or (pO.DP)
    cudnn.benchmark = True


def rSeed(sd):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed(sd)

def launch_fn(rank, opt):
    global OnceExecWorker
    gInit(opt)
    OnceExecWorker = OnceExecWorker or (pO.DDP and rank==0)
    mp.set_start_method('fork', force=True)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(opt.port)

    dist.init_process_group("nccl", rank=rank, world_size=opt.num_gpu)

    #to ensure identical init parameters
    rSeed(opt.manualSeed)

    torch.cuda.set_device(rank)
    opt.world_size = opt.num_gpu
    opt.rank       = rank

    train(opt)


def val(model_path='saved_models/iam_gin_test4_/best_norm_ED.pth'):

    HVD3P = pO.HVD or pO.DDP

    model = OrigamiNet()
    model_ema = ModelEma(model)

    #checkpoint = torch.load(model_path, map_location='cuda:1')

    #model.load_state_dict(checkpoint['model'], strict=False)
    # model_ema._load_checkpoint(model_path, 'cuda:1')
    map_location = {'cuda:%d' % 0: 'cuda:%d' % 0}
    # model_ema._load_checkpoint(model_path, map_location=map_location)
    model_ema.load(model_path, map_location=map_location)
    test_data_list  = 'iam/val.gc'
    valid_dataset = ds_load.myLoadDS(test_data_list, '/home/zju/w4/OrigamiNet/iam_data/pargs/')
    # start_time = time.time()

    valid_loader  = torch.utils.data.DataLoader( valid_dataset, batch_size=1 , pin_memory=True,
                    num_workers = 1, sampler= None)
    labels_snippets = "Become a success with a disc and hey presto! You're a star"
    labels_input = tokenizer(labels_snippets,
                             return_tensors="pt",
                             padding=True,
                             truncation=False).to('cuda')
    min_sim = 10
    hit_str = ''

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        model.eval()

        image = image_tensors.to('cuda')

        with torch.no_grad():

            preds, sim_value = model_ema.ema(image, labels_input)

            preds = preds.float()
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.permute(1, 0, 2).log_softmax(2)  # to use CTCloss format
            _, preds_index = preds.max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)
            if sim_value < min_sim:
                min_sim = sim_value
                hit_img = image
                hit_str = preds_str

    return min_sim, hit_img, hit_str
    # v_time = time.time() - start_time



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gin', default='iam/iam.gin', help='Gin config file')

    opt = parser.parse_args()
    gInit(opt)
    opt.manualSeed = ginM('manualSeed')
    opt.port = ginM('port')

    if OnceExecWorker:
        rSeed(opt.manualSeed)

    opt.num_gpu = torch.cuda.device_count()

    mp.spawn(launch_fn, args=(opt,), nprocs=opt.num_gpu)

    # val()
