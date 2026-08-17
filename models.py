import csv
from torch.utils.data import Dataset
import numpy as np
import sys
import random
import torch
import linecache
import pickle
import time
from configparser import ConfigParser
import scipy.sparse as sp

import itertools
from scipy.sparse import coo_matrix
from utils import *
from torch.utils.data import DataLoader

from Dataset import Data
from layers import *

class DIG_Generator(torch.nn.Module):
    def __init__(self, num_items, seq_len, history_len, emb_dim=64, device='cpu', layer_num=4, K=5, alpha=0.1):
        super(DIG_Generator, self).__init__()

        self.device = device
        self.emb_dim = emb_dim
        self.layer_num = layer_num
        self.seq_len = seq_len
        self.history_len = history_len
        self.K = K
        self.alpha = alpha
        
        self.item_embed = nn.Embedding(num_items, emb_dim)
        self.basket_time_embed = nn.Embedding(seq_len, emb_dim) # the max len of seqs

        layers = []
        for i in range(self.layer_num):
            layers.append(HGNN_conv(in_ft=emb_dim, out_ft=emb_dim))
        self.layers = nn.ModuleList(layers)

        self.basket_encoder = nn.MultiheadAttention(embed_dim=emb_dim*2, num_heads=4, batch_first=True)

        self.group_ae = Group_AE(in_ft=history_len+1, out_ft=emb_dim, K=K)

        out_encoders = []
        for _ in range(K):
            out_encoders.append(Score_layer(emb_dim))
        self.out_encoders = nn.ModuleList(out_encoders)

        self.bce_criterion = torch.nn.BCELoss(reduction="sum") # sum or not, sum is quite important
        self.ae_criterion = nn.MSELoss(reduction="sum")
        self.reset_parameters(num_items)

    def reset_parameters(self, num_items):
        self.item_embed.weight.data.uniform_(-1./np.sqrt(num_items), 1./np.sqrt(num_items))
        self.basket_time_embed.weight.data.uniform_(-1./np.sqrt(num_items), 1./np.sqrt(num_items))

    def set_paras(self, input_x, norm_HH, norm_HG):
        self.input_x = input_x
        self.norm_HH = norm_HH
        self.norm_HG = norm_HG

    def update(self):
        input_x = self.input_x
        item_emb = self.item_embed(input_x)

        item_emb_list = [item_emb]
        hyper_emb_list = []
        for layer in self.layers:
            updated_item_emb, updated_hyper_emb = layer(item_emb, self.norm_HH, self.norm_HG)
            item_emb_list.append(updated_item_emb)
            hyper_emb_list.append(updated_hyper_emb)

        # layer_num - 1 --> 1 layer num for 0 layer
        item_emb_list = item_emb_list[:-1]
        hyper_emb_list = hyper_emb_list

        new_item_emb = torch.mean(torch.stack(item_emb_list), dim=0)
        new_hyper_emb = torch.mean(torch.stack(hyper_emb_list), dim=0)
        return new_item_emb, new_hyper_emb

    def seq_encoder(self, item_emb, hyper_emb, batch_data):
        basket_seq, basket_time, basket_mask, ui_history, all_items, target = batch_data

        # basket seq encoder
        basket_seq_emb = hyper_emb[basket_seq]
        basket_time_emb = self.basket_time_embed(basket_time)
        basket_input_emb = torch.cat([basket_seq_emb, basket_time_emb], dim=2)

        basket_lens = (self.seq_len-1) - torch.sum(basket_mask, dim=1)
        lastest_basket_input_emb = torch.cat([torch.index_select(out, dim=0, index=ind).unsqueeze(0) for out, ind in zip(basket_input_emb, basket_lens)])

        basket_seq_out, attn_output_weights = self.basket_encoder(query=lastest_basket_input_emb, key=basket_input_emb, value=basket_input_emb, 
                key_padding_mask=basket_mask)
        basket_seq_out = basket_seq_out.squeeze(1)

        return basket_seq_out

    def pattern_encoder(self, pattern):
        pattern_seq_emb = torch.sum(pattern_seq_emb, dim=2)
        return pattern_seq_emb 

    def forward(self, batch_data):
        item_emb, hyper_emb = self.update()
        basket_seq, basket_time, basket_mask, ui_history, all_items, target = batch_data
        
        # seq encoder
        basket_seq_out = self.seq_encoder(item_emb, hyper_emb, batch_data)

        # pattern encoder
        batch, item_num, ui_dim = ui_history.size()
        pattern_encoded, lamd, pattern_decoded = self.group_ae(ui_history)
        pattern_encoded = pattern_encoded.view(batch, item_num, -1)
        lamd = lamd.view(batch, item_num, -1)
        pattern_decoded = pattern_decoded.view(batch, item_num, -1)

        # score process
        candi_item_emb = item_emb[all_items]
        batch, item_num, emb_dim = candi_item_emb.size()

        interact_emb = basket_seq_out.unsqueeze(1).repeat(1, item_num, 1)
        interact_emb = torch.cat([candi_item_emb, interact_emb, pattern_encoded], dim=2)

        rec_score = []
        for i in range(self.K):
            score = self.out_encoders[i](interact_emb)
            rec_score.append(score)
        rec_score = torch.cat(rec_score, dim=2)
        rec_score = torch.sum(rec_score*lamd, dim=2)
        rec_score = torch.nn.functional.softmax(rec_score, dim=-1)

        return rec_score, pattern_decoded

    def test_score(self, test_batch_data):
        item_emb, hyper_emb = self.update()
        basket_seq, basket_time, basket_mask, ui_history, all_items = test_batch_data
        batch_data = test_batch_data + [None]
        
        # seq encoder
        basket_seq_out = self.seq_encoder(item_emb, hyper_emb, batch_data)

        # pattern encoder
        batch, item_num, ui_dim = ui_history.size()
        pattern_encoded, lamd, pattern_decoded = self.group_ae(ui_history)
        pattern_encoded = pattern_encoded.view(batch, item_num, -1)
        lamd = lamd.view(batch, item_num, -1)
        pattern_decoded = pattern_decoded.view(batch, item_num, -1)

        # score process
        candi_item_emb = item_emb[all_items]
        batch, item_num, emb_dim = candi_item_emb.size()

        interact_emb = basket_seq_out.unsqueeze(1).repeat(1, item_num, 1)
        interact_emb = torch.cat([candi_item_emb, interact_emb, pattern_encoded], dim=2)

        rec_score = []
        for i in range(self.K):
            score = self.out_encoders[i](interact_emb)
            rec_score.append(score)
        rec_score = torch.cat(rec_score, dim=2)
        rec_score = torch.sum(rec_score*lamd, dim=2)
        rec_score = torch.nn.functional.softmax(rec_score, dim=-1).squeeze()

        return rec_score

class DIG_Discriminator(torch.nn.Module):
    def __init__(self, num_items, seq_len, emb_dim=64, device='cpu', layer_num=4):
        super(DIG_Discriminator, self).__init__()

        self.device = device
        self.emb_dim = emb_dim
        self.layer_num = layer_num
        self.seq_len = seq_len
        
        self.item_embed = nn.Embedding(num_items, emb_dim)
        self.basket_time_embed = nn.Embedding(seq_len, emb_dim) # the max len of seqs

        layers = []
        for i in range(self.layer_num):
            layers.append(HGNN_conv(in_ft=emb_dim, out_ft=emb_dim))
        self.layers = nn.ModuleList(layers)

        self.basket_encoder = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=4, batch_first=True, kdim=emb_dim*2, vdim=emb_dim*2)

        self.clf_fn = nn.Linear(emb_dim*2, 1)
        self.sigmoid = nn.Sigmoid()

        self.reset_parameters(num_items)

    def reset_parameters(self, num_items):
        self.item_embed.weight.data.uniform_(-1./np.sqrt(num_items), 1./np.sqrt(num_items))
        self.basket_time_embed.weight.data.uniform_(-1./np.sqrt(num_items), 1./np.sqrt(num_items))

    def set_paras(self, input_x, norm_HH, norm_HG):
        self.input_x = input_x
        self.norm_HH = norm_HH
        self.norm_HG = norm_HG

    def update(self):
        input_x = self.input_x
        item_emb = self.item_embed(input_x)

        item_emb_list = [item_emb]
        hyper_emb_list = []
        for layer in self.layers:
            updated_item_emb, updated_hyper_emb = layer(item_emb, self.norm_HH, self.norm_HG)
            item_emb_list.append(updated_item_emb)
            hyper_emb_list.append(updated_hyper_emb)

        # layer_num - 1 --> 1 layer num for 0 layer
        item_emb_list = item_emb_list[:-1]
        hyper_emb_list = hyper_emb_list

        new_item_emb = torch.mean(torch.stack(item_emb_list), dim=0)
        new_hyper_emb = torch.mean(torch.stack(hyper_emb_list), dim=0)
        return new_item_emb, new_hyper_emb

    def seq_encoder(self, item_emb, hyper_emb, batch_data):
        basket_seq, basket_time, basket_mask, target = batch_data

        # basket seq encoder
        basket_seq_emb = hyper_emb[basket_seq]
        basket_time_emb = self.basket_time_embed(basket_time)
        basket_input_emb = torch.cat([basket_seq_emb, basket_time_emb], dim=-1)

        target_emb = torch.mean(item_emb[target], dim=0)
        lastest_basket_input_emb = target_emb.unsqueeze(0)

        basket_seq_out, attn_output_weights = self.basket_encoder(query=lastest_basket_input_emb, key=basket_input_emb, value=basket_input_emb, 
                key_padding_mask=basket_mask)
        return basket_seq_out 

    def forward(self, batch_data, item_emb, hyper_emb):
        basket_seq, basket_time, basket_mask, target = batch_data

        # seq encoder
        basket_seq_out = self.seq_encoder(item_emb, hyper_emb, batch_data)

        target_emb = torch.mean(item_emb[target], dim=0).unsqueeze(0)
        _input = torch.cat([basket_seq_out, target_emb], dim=1)
        cls_score = self.clf_fn(_input)
        cls_score = self.sigmoid(cls_score)
        return cls_score.squeeze(0)