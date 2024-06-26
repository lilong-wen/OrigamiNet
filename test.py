import os
import time
import string

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.distributed as dist
import numpy as np
import editdistance
from nltk.translate.bleu_score import sentence_bleu
import horovod.torch as hvd
from utils import random_select_txt_snippets, gt_txt_sim
from transformers import AutoTokenizer
from utils import Averager
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def metric_sum_hvd(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name, average=False)
    return avg_tensor.item()

def metric_sum_ddp(tensor, av=False):
    # rt = tensor.clone()
    rt = torch.tensor(tensor).cuda()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if av:
        rt /= dist.get_world_size()
    return rt

def validation(model, criterion_ctc, criterion_sim, evaluation_loader, converter, opt, parO, bert_base_model):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    tot_ED = 0
    length_of_gt = 0
    bleu = 0.0
    infer_time = 0
    valid_loss_avg = Averager()
    valid_sim_loss_avg = Averager()
    tokenizer = AutoTokenizer.from_pretrained(bert_base_model, cache_dir='./cached')#, do_lower_case=config['model_bert']['do_lower_case'])

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        image = image_tensors.to(device)

        text_for_loss, length_for_loss = converter.encode(labels)
        labels_snippets = random_select_txt_snippets(labels)
        print("search text: ")
        print(labels_snippets)
        text_snippets_for_loss, length_snippets_for_loss = converter.encode(labels_snippets)
        gt_sim = gt_txt_sim(text_for_loss, length_for_loss, text_snippets_for_loss, length_snippets_for_loss)
        gt_sim = torch.FloatTensor(gt_sim)
        labels_input = tokenizer(labels_snippets,
                                 return_tensors="pt",
                                 padding=True,
                                 truncation=False).to(device)

        start_time = time.time()

        preds, sim_value = model(image, labels_input)
        # print(12*"@")
        print("predicted sim value")
        print(sim_value)
        print("gt sim value")
        print(gt_sim)
        preds = preds.float()
        forward_time = time.time() - start_time

        # Calculate evaluation loss for CTC deocder.
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        preds = preds.permute(1, 0, 2).log_softmax(2)  # to use CTCloss format

        # To avoid ctc_loss issue, disabled cudnn for the computation of the ctc_loss
        # https://github.com/jpuigcerver/PyLaia/issues/16
        torch.backends.cudnn.enabled = False
        cost_ctc = criterion_ctc(preds, text_for_loss, preds_size, length_for_loss).mean()
        cost_sim = 10000 * criterion_sim(sim_value.to(device), gt_sim.to(device)).sum()

        cost = cost_ctc + cost_sim
        torch.backends.cudnn.enabled = True

        _, preds_index = preds.max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
        preds_str = converter.decode(preds_index.data, preds_size.data)
        print("preds_str")
        print(preds_str)

        infer_time += forward_time
        valid_loss_avg.add(cost)
        valid_sim_loss_avg.add(cost_sim)

        # calculate accuracy.
        for pred, gt in zip(preds_str, labels):
            tmped = editdistance.eval(pred, gt)
            if pred == gt:
                n_correct += 1
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += tmped / float(len(gt))

            tot_ED += tmped
            length_of_gt += len(gt)
            bleu += sentence_bleu( [list(gt)], list(pred) ) 

    if parO.HVD:
        n_correct  = metric_sum_hvd(n_correct , 'sum_n_correct')
        tot_ED  = metric_sum_hvd(tot_ED , 'sum_tot_ED')
        length_of_gt  = metric_sum_hvd(length_of_gt , 'sum_length_of_gt')
        norm_ED = metric_sum_hvd(norm_ED, 'sum_norm_ED')
        bleu = metric_sum_hvd(bleu, 'sum_bleu')

    elif parO.DDP:
        val_loss = metric_sum_ddp(valid_loss_avg.val(), av=True)
        val_sim_loss = metric_sum_ddp(valid_sim_loss_avg.val(), av=True)
        n_correct  = metric_sum_ddp(n_correct)
        tot_ED  = metric_sum_ddp(tot_ED)
        length_of_gt  = metric_sum_ddp(length_of_gt)
        norm_ED = metric_sum_ddp(norm_ED)
        bleu = metric_sum_ddp(bleu)

    nelms = float(len(evaluation_loader.dataset))
    tot_ED = tot_ED / float(length_of_gt)
    val_loss = valid_loss_avg.val() if not parO.DDP else val_loss
    val_sim_loss = valid_sim_loss_avg.val() if not parO.DDP else val_sim_loss
    bleu /= nelms
    norm_ED /= nelms
    accuracy = n_correct / nelms * 100

    return val_loss, val_sim_loss, accuracy, norm_ED, tot_ED, bleu, preds_str, labels, infer_time


def validation_single(model, criterion_ctc, criterion_sim, evaluation_loader, converter, opt, bert_base_model):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    tot_ED = 0
    length_of_gt = 0
    bleu = 0.0
    infer_time = 0
    valid_loss_avg = Averager()
    valid_sim_loss_avg = Averager()
    tokenizer = AutoTokenizer.from_pretrained(bert_base_model, cache_dir='./cached')#, do_lower_case=config['model_bert']['do_lower_case'])

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        image = image_tensors.to(device)

        text_for_loss, length_for_loss = converter.encode(labels)
        labels_snippets = random_select_txt_snippets(labels)
        print("search text: ")
        print(labels_snippets)
        text_snippets_for_loss, length_snippets_for_loss = converter.encode(labels_snippets)
        gt_sim = gt_txt_sim(text_for_loss, length_for_loss, text_snippets_for_loss, length_snippets_for_loss)
        gt_sim = torch.FloatTensor(gt_sim)
        labels_input = tokenizer(labels_snippets,
                                 return_tensors="pt",
                                 padding=True,
                                 truncation=False).to(device)

        start_time = time.time()

        preds, sim_value = model(image, labels_input)
        # print(12*"@")
        print("predicted sim value")
        print(sim_value)
        print("gt sim value")
        print(gt_sim)
        preds = preds.float()
        forward_time = time.time() - start_time

        # Calculate evaluation loss for CTC deocder.
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        preds = preds.permute(1, 0, 2).log_softmax(2)  # to use CTCloss format

        # To avoid ctc_loss issue, disabled cudnn for the computation of the ctc_loss
        # https://github.com/jpuigcerver/PyLaia/issues/16
        torch.backends.cudnn.enabled = False
        cost_ctc = criterion_ctc(preds, text_for_loss, preds_size, length_for_loss).mean()
        cost_sim = 10000 * criterion_sim(sim_value.to(device), gt_sim.to(device)).sum()

        cost = cost_ctc + cost_sim
        torch.backends.cudnn.enabled = True

        _, preds_index = preds.max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
        preds_str = converter.decode(preds_index.data, preds_size.data)
        print("preds_str")
        print(preds_str)

        infer_time += forward_time
        valid_loss_avg.add(cost)
        valid_sim_loss_avg.add(cost_sim)

        # calculate accuracy.
        for pred, gt in zip(preds_str, labels):
            tmped = editdistance.eval(pred, gt)
            if pred == gt:
                n_correct += 1
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += tmped / float(len(gt))

            tot_ED += tmped
            length_of_gt += len(gt)
            bleu += sentence_bleu( [list(gt)], list(pred) ) 


    nelms = float(len(evaluation_loader.dataset))
    tot_ED = tot_ED / float(length_of_gt)
    val_loss = valid_loss_avg.val()
    val_sim_loss = valid_sim_loss_avg.val()
    bleu /= nelms
    norm_ED /= nelms
    accuracy = n_correct / nelms * 100

    return val_loss, val_sim_loss, accuracy, norm_ED, tot_ED, bleu, preds_str, labels, infer_time
