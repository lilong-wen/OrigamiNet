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

@gin.configurable
class PadPool(nn.Module):
    def forward(self, x):
        x = F.pad(x, [0, 0, 0, 1])
        x = F.max_pool2d(x,(2, 2), stride=(1, 2))
        return x

@gin.configurable
class TextEncoder(nn.Module):
    def __init__(self, bert_base_model, out_dim, freeze_layers):
        super().__init__()
        #init BERT
        self.bert_model = self._get_bert_basemodel(bert_base_model,freeze_layers)
        # projection MLP for BERT model
        self.bert_l1 = nn.Linear(768, 768) #768 is the size of the BERT embbedings
        self.bert_l2 = nn.Linear(768, out_dim) #768 is the size of the BERT embbedings

    def _get_bert_basemodel(self, bert_model_name, freeze_layers):
        try:
            model = AutoModel.from_pretrained(bert_model_name)#, return_dict=True)
            print("text feature extractor:", bert_model_name)
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model


    def mean_pooling(self, model_output, attention_mask):

        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encoder(self, encoded_inputs):

        outputs = self.bert_model(**encoded_inputs)

        with torch.no_grad():
            sentence_embeddings = self.mean_pooling(outputs, encoded_inputs['attention_mask'])
            x = self.bert_l1(sentence_embeddings)
            x = F.relu(x)
            out_emb = self.bert_l2(x)

        return out_emb

    def forward(self, encoded_inputs):

        # print(encoded_inputs)
        zls = self.encoder(encoded_inputs)

        return zls


def pCnv(input_channels, output_channels):

    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 1, bias=False),
        nn.InstanceNorm2d(output_channels, affine=True)
    )

def dsCnv(input_channels, kernel_size):

    return nn.Sequential(
        nn.Conv2d(input_channels, input_channels, kernel_size, bias=False,
                  padding=(kernel_size - 1) // 2),
        nn.InstanceNorm2d(input_channels, affine=True)
    )

@gin.configurable
class LN(nn.Module):

    def forward(self, x):

        return F.layer_norm(x, x.size()[1:], weight=None, bias=None, eps=1e-05)

@gin.configurable
class InitBlock(nn.Module):

    def __init__(self, feature_upstream, num_channels):

        super().__init__()

        self.norm1 = LN()
        self.InitSequence = nn.Sequential(
            pCnv(num_channels, feature_upstream),
            nn.Softmax(dim=1),
            dsCnv(feature_upstream, 11),
            LN()
        )

    def forward(self, x):

        x = self.norm1(x)
        xt = x
        x = self.InitSequence(x)
        x = torch.cat([x, xt], 1)

        return x

class Gate(nn.Module):

    def __init__(self, input_feature_size, num_gates=2):

        super().__init__()
        self.norm = LN()
        self.num_gates = num_gates

    def forward(self, x):

        t0, t1 = torch.chunk(x, self.num_gates, dim=1)
        t0 = torch.tanh_(t0)
        # meant to be an attention mechanism,
        # should be changed later, t1 should sub t3
        t1.sub_(2)
        t1 = torch.sigmoid_(t1)

        return t1 * t0

def custom_GradienCheck(module):

    def custom_forward(*inputs):
        inputs = module(inputs[0])
        return inputs

    return custom_forward

@gin.configurable
class GateBlock(nn.Module):

    def __init__(self, input_feature_size, output_feature_size, gt=True,
                 kernel_size=3, num_gates=2, GradCheck=gin.REQUIRED):
        super().__init__()

        middle_feature_size = int(math.floor(input_feature_size / 2))
        # input_feature_size_2 = input_feature_size + input_feature_size % 2

        self.sequnce = nn.Sequential(
            pCnv(input_feature_size, middle_feature_size),
            dsCnv(middle_feature_size, kernel_size),
            nn.ELU(), # activation function

            pCnv(middle_feature_size, middle_feature_size * num_gates),
            dsCnv(middle_feature_size * num_gates, kernel_size),
            Gate(middle_feature_size),

            pCnv(middle_feature_size, input_feature_size),
            dsCnv(input_feature_size, kernel_size),
            nn.ELU()
        )

        self.gt = gt
        self.gc = GradCheck

    def forward(self, x):

        if self.gc >= 1:
            y = checkpoint(custom_GradienCheck(self.sequnce), x)
        else:
            y = self.sequnce(x)

        out = x + y

        return out


@gin.configurable
class RNN_SIM(nn.Module):

    def __init__(self, EMBEDDING_DIM, RNN_UNITS, BI_RNN, RNN_LAYERS):
        super().__init__()

        # similarity measure
        self.rnn = nn.GRU(EMBEDDING_DIM, RNN_UNITS, bidirectional=BI_RNN,
                          num_layers=RNN_LAYERS, batch_first=True)
        self.lin1 = nn.Linear(RNN_UNITS * 2, 96)
        self.lin2 = nn.Linear(96, 28)
        self.out = nn.Linear(28, 1)

    def forward(self, x_input):
        x1, x2 = x_input[0], x_input[1]
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


@gin.configurable
class OrigamiNet_extended(nn.Module):

    def __init__(self, layer_resize, feature_upstream, num_channels, num_layers,
                 layer_sizes, wmul, output_classes, GradCheck, reduceAxis=3):

        super().__init__()

        # layer_resize
        # 0: @MaxPool2d(),
        # 2: @MaxPool2d(),
        # 4: @MaxPool2d(),
        # 6: @PadPool(),
        # 8: @PadPool(),
        # 10: @s1/Upsample(),
        # 11: @s2/Upsample()
        self.layer_resize = layer_resize
        self.InitSequence = InitBlock(feature_upstream)
        self.text_encoder = TextEncoder()

        layers = []
        input_size = feature_upstream + num_channels
        output_size = input_size
        for i in range(num_layers):
            # {0:  128, 2:  256, 4:  512, 11: 256  }
            output_size = int(math.floor(layer_sizes[i] * wmul)) if i in layer_sizes else input_size
            layers.append(GateBlock(input_size, output_size, True, 3))

            if input_size != output_size:

                layers.append(pCnv(input_size, output_size))
                layers.append(nn.ELU())

            input_size = output_size

            if i in layer_resize:
                layers.append(layer_resize[i])

        layers.append(LN())
        self.GatedSequence = nn.Sequential(*layers)

        self.FinealSequence = nn.Sequential(
            pCnv(output_size, output_classes),
            nn.ELU()
        )

        self.TextSequence = nn.Sequential(
            TextEncoder()
        )
        self.SimSequence = nn.Sequential(
            RNN_SIM()
        )

        self.norm1 = LN()
        self.checkGradent = GradCheck
        self.reduceAxis = reduceAxis

    def forward(self, x, txt_tokens):

        x = self.InitSequence(x)

        if self.checkGradent >= 2:
            x = checkpoint_sequential_step(self.GatedSequence, 4, x)
        else:
            x = self.GatedSequence(x)

        x = self.FinealSequence(x)

        x = torch.mean(x, self.reduceAxis, keepdim=False)
        x = self.norm1(x)
        x = x.permute(0, 2, 1)

        txt_emb = self.TextSequence(txt_tokens).unsqueeze(1)
        #print(txt_emb.shape)
        sim = self.SimSequence([x, txt_emb])

        return x, sim
