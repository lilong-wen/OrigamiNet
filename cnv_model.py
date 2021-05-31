import torch
import torch.nn as nn
import torch.nn.functional as F
from chk import checkpoint_sequential_step, checkpoint

import math
import numpy as np
from torchvision.utils import save_image

from transformers import AutoModel

import gin

def ginM(n): return gin.query_parameter(f'%{n}')
gin.external_configurable(nn.MaxPool2d, module='nn')
gin.external_configurable(nn.Upsample,  module='nn')


class LN(nn.Module):
    def forward(self, x):
        return F.layer_norm(x, x.size()[1:], weight=None, bias=None, eps=1e-05)

@gin.configurable
class PadPool(nn.Module):
    def forward(self, x):
        x = F.pad(x, [0, 0, 0, 1]) 
        x = F.max_pool2d(x,(2, 2), stride=(1, 2))
        return x

def pCnv(inp,out,groups=1):
  return nn.Sequential(
      nn.Conv2d(inp,out,1,bias=False,groups=groups),
      nn.InstanceNorm2d(out,affine=True)
  )
  
#regarding same padding in PT https://github.com/pytorch/pytorch/issues/3867
def dsCnv(inp,k):
  return nn.Sequential(
      nn.Conv2d(inp,inp,k,groups=inp,bias=False,padding=(k - 1) // 2),
      nn.InstanceNorm2d(inp,affine=True)
  )

ngates = 2

class Gate(nn.Module):
    def __init__(self,ifsz):
        super().__init__()
        self.ln = LN()

    def forward(self, x):
        t0,t1 = torch.chunk(x, ngates, dim=1)
        t0 = torch.tanh_(t0)
        t1.sub_(2)
        t1 = torch.sigmoid_(t1)

        return t1*t0

def customGC(module):
    def custom_forward(*inputs):
        inputs = module(inputs[0])
        return inputs
    return custom_forward

@gin.configurable
class GateBlock(nn.Module):
    def __init__(self, ifsz, ofsz, gt = True, ksz = 3, GradCheck=gin.REQUIRED):
        super().__init__()

        cfsz   = int( math.floor(ifsz/2) )
        ifsz2  = ifsz + ifsz%2

        self.sq = nn.Sequential(
          pCnv(ifsz, cfsz),
          dsCnv(cfsz,ksz),
          nn.ELU(),
          ###########
          pCnv(cfsz, cfsz*ngates),
          dsCnv(cfsz*ngates,ksz),
          Gate(cfsz),
          ###########
          pCnv(cfsz, ifsz),
          dsCnv(ifsz,ksz),
          nn.ELU(),
        )

        self.gt = gt
        self.gc = GradCheck
    

    def forward(self, x):
        if self.gc >= 1:
          y = checkpoint(customGC(self.sq), x)
        else:
          y = self.sq(x)

        out = x + y
        return out

@gin.configurable
class InitBlock(nn.Module):
    def __init__(self, fup, n_channels):
        super().__init__()

        self.n1 = LN()
        self.Initsq = nn.Sequential(
          pCnv(n_channels, fup),
          nn.Softmax(dim=1),
          dsCnv(fup,11),
          LN()
        )

    def forward(self, x):
        x  = self.n1(x)
        xt = x
        x  = self.Initsq(x)
        x  = torch.cat([x,xt],1)
        return x

'''
class Simcal(nn.Module):
    # defining the structure of the network
    def __init__(self, VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS,
                 BI_RNN, RNN_LAYERS ):
        super(Simcal, self).__init__()
        # self.embedding = nn.Embedding(VOCAB_SIZE + 1, EMBEDDING_DIM)
        self.rnn = nn.GRU(EMBEDDING_DIM, RNN_UNITS, bidirectional=BI_RNN, 
                          num_layers=RNN_LAYERS, batch_first=True)
        self.lin1 = nn.Linear(RNN_UNITS * 2, 96)
        self.lin2 = nn.Linear(96, 28)
        self.out = nn.Linear(28, 1)

    # defining steps in forward pass
    def forward(self, x1, x2):
        try:
            # x1 = self.embedding(x1)
            # x2 = self.embedding(x2)
            print(self.rnn(x1))
            x1 = self.rnn(x1)[1]
            x1 = x1.view(-1, x1.size()[1], x1.size()[2]).sum(dim=0)
            x2 = self.rnn(x2)[1]
            x2 = x2.view(-1, x2.size()[1], x2.size()[2]).sum(dim=0)
            lin = self.lin1(torch.cat((x1, x2), 1))
            lin = torch.relu(self.lin2(lin))
            pred = self.out(lin)
            return pred
        except IndexError:
            print(x1.max(), x2.max())
'''

@gin.configurable
class OrigamiNet(nn.Module):
    def __init__(self, n_channels, o_classes, wmul, lreszs, lszs, nlyrs, fup, GradCheck,
                 VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BI_RNN, RNN_LAYERS,
                 reduceAxis=3):
        super().__init__()

        self.lreszs = lreszs
        self.Initsq = InitBlock(fup)

        # TEXT Encoder
        self.bert_model = self._get_bert_basemodel('emilyalsentzer/Bio_ClinicalBERT',
                                                   [0,1,2,3,4,5])
        # projection MLP for BERT model
        self.bert_l1 = nn.Linear(768, 768) #768 is the size of the BERT embbedings
        self.bert_l2 = nn.Linear(768, 80) #768 is the size of the BERT embbedings
    
        layers = []
        isz = fup + n_channels
        osz = isz
        for i in range(nlyrs):
            osz = int( math.floor(lszs[i] * wmul) ) if i in lszs else isz    
            layers.append( GateBlock(isz, osz, True, 3) )

            if isz != osz:
              layers.append( pCnv(isz, osz) )
              layers.append( nn.ELU() )
            isz = osz

            if i in lreszs:
              layers.append( lreszs[i] )
 
        layers.append( LN() )
        self.Gatesq = nn.Sequential(*layers)
        
        self.Finsq = nn.Sequential(
          pCnv(osz, o_classes),
          nn.ELU(),
        )

        self.n1 = LN()
        self.it=0
        self.gc = GradCheck
        self.reduceAxis = reduceAxis

        # similarity measure
        self.rnn = nn.GRU(EMBEDDING_DIM, RNN_UNITS, bidirectional=BI_RNN, 
                          num_layers=RNN_LAYERS, batch_first=True)
        self.lin1 = nn.Linear(RNN_UNITS * 2, 96)
        self.lin2 = nn.Linear(96, 28)
        self.out = nn.Linear(28, 1)

    def _get_bert_basemodel(self, bert_model_name, freeze_layers):
        try:
            model = AutoModel.from_pretrained(bert_model_name, cache_dir='./cached')#, return_dict=True)
            print("Image feature extractor:", bert_model_name)
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model

    def mean_pooling(self, model_output, attention_mask):

        """
        Mean Pooling - Take attention mask into account for correct averaging
        Reference: https://www.sbert.net/docs/usage/computing_sentence_embeddings.html
        """
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def text_encoder(self, encoded_inputs):
        """
        Obter os inputs e em seguida extrair os hidden layers e fazer a media de todos os tokens
        Fontes:
        - https://github.com/BramVanroy/bert-for-inference/blob/master/introduction-to-bert.ipynb
        - Nils Reimers, Iryna Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
        https://www.sbert.net
        """
        outputs = self.bert_model(**encoded_inputs)

        with torch.no_grad():
            sentence_embeddings = self.mean_pooling(outputs, encoded_inputs['attention_mask'])
            x = self.bert_l1(sentence_embeddings)
            x = F.relu(x)
            out_emb = self.bert_l2(x)

        return out_emb
    
    def sim_measure(self, x1, x2):
        try:
            # x1 = self.embedding(x1)
            # x2 = self.embedding(x2)
            x1 = self.rnn(x1)[1]
            x1 = x1.view(-1, x1.size()[1], x1.size()[2]).sum(dim=0)
            x2 = self.rnn(x2)[1]
            x2 = x2.view(-1, x2.size()[1], x2.size()[2]).sum(dim=0)
            lin = self.lin1(torch.cat((x1, x2), 1))
            lin = torch.relu(self.lin2(lin))
            pred = torch.sigmoid(self.out(lin))
            return pred
        except IndexError:
            print(x1.max(), x2.max())

    def forward(self, x, encoded_inputs):
        x = self.Initsq(x)

        if self.gc >=2:
          x = checkpoint_sequential_step(self.Gatesq,4,x)  #slower, more memory save
          # x = checkpoint_sequential_step(self.Gatesq,8,x)  #faster, less memory save
        else:
          x = self.Gatesq(x)

        x = self.Finsq(x)
 
        x = torch.mean(x, self.reduceAxis, keepdim=False)
        x = self.n1(x)
        x = x.permute(0,2,1)

        txt_emb = self.text_encoder(encoded_inputs).unsqueeze(1)

        # print(f'input x shape {x.shape}')
        # print(f'input txt_emb shape {txt_emb.shape}')
        sim = self.sim_measure(x, txt_emb)

        return x, sim
    
