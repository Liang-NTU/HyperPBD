import csv
from torch.utils.data import Dataset
import numpy as np
import sys
import os
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

# input: <bu, is, hs> <x0> (<xt, t> from <x0>)
# <bu> can has min length constrain
# <bu> less than min_trans filtered --> preprocessing --> when changing min_trans, delete the prep_data
class Data(Dataset):
    def __init__(self, args):

        self.min_trans = args.min_trans
        self.min_basket_start = args.min_basket_start
        self.data_name = args.train_name

        self.history_len = args.history_len
        self.seq_len = args.seq_len 

        print("================ Load raw input data ===================") 
        user2basket, basket2items, items_list, test_data = load_data(args.train_name, args.test_name, args.min_trans)
        print("user2basket:", len(user2basket), "basket2items:", len(basket2items), \
            "items_list:", len(items_list), "test_data:", len(test_data))
        print("================ End Load raw input data ===================" + "\n") 

        self.device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')

        self.user2basket = user2basket
        self.basket2items = basket2items
        self.items_list = items_list
        self.test_data = test_data

        self.num_items = len(items_list)
        self.num_baskets = len(basket2items)

        # construct hypergraph based on relations between basket and item
        self.construct_hypergraph()

        # set max basket seq len, max item seq len, and max history seq len
        print("================ Construct training_input & testing_input ===================") 
        self.max_interval = 0
        training_input = self.generate_train_input()
        self.training_input = training_input
        testing_input = self.generate_test_input()
        self.testing_input = testing_input

        print("num of training_input", len(training_input[0]), "num of testing_input", len(testing_input[0]))
        print("max interval", self.max_interval)
        print("================ End Construct training_input & testing_input ===================" + "\n") 

        # set input_x (item_ids)
        input_x = []
        for item in range(self.num_items):
            input_x.append(item)
        input_x = torch.LongTensor(input_x)
        self.input_x = input_x

    # basket & item hypergraph
    def construct_hypergraph(self):
        print("================ Construct Hypergraph ===================") 
        basket2items = self.basket2items
        item2basket = {}
        for basket_id in basket2items:
            item_list = basket2items[basket_id]
            for item in item_list:
                if item not in item2basket:
                    item2basket[item] = []
                item2basket[item].append(basket_id)
        self.item2basket = item2basket

        print("basket2items:", len(self.basket2items), "item2basket:", len(self.item2basket)) 
        
        hyper_network = np.zeros((len(self.item2basket), len(self.basket2items)))
        for item in self.item2basket:
            for basket_id in self.item2basket[item]:
                hyper_network[item][basket_id] = 1.0
        
        self.H = sp.coo_matrix(hyper_network)
        print("hypergraph for HGNN", self.H.shape)

        DH = np.sum(hyper_network, axis=0) ** (-1.0)
        DH = sp.diags(DH)
        print("DH", DH.shape)

        DvH = np.sum(hyper_network, axis=1) ** (-1/2)
        DvH = sp.diags(DvH)
        print("DvH", DvH.shape)

        norm_HH = DH * self.H.T
        norm_HG = DvH * DvH * self.H

        self.norm_HH = self.sparse_to_tensor(norm_HH) # D->H message, H*D [D*E]
        self.norm_HG = self.sparse_to_tensor(norm_HG) # H->D message, D*H [H*E]

        print("norm_HH shape:", self.norm_HH.size(), "norm_HG shape:", self.norm_HG.size())
        print("================ End Construct Hypergraph ===================" + "\n") 

    def generate_train_input(self):
        basket_seqs = []
        basket_times = []
        basket_masks = []

        ui_historys = []
        targets = [] 

        for user_id in self.user2basket:
            baskets = self.user2basket[user_id]

            repeat_items = []
            for basket_id in baskets:
                for item in self.basket2items[basket_id]:
                    repeat_items.append(item)
            repeat_items = list(set(repeat_items)) 
            repeat_basket_indicator = {}
            for i, basket_id in enumerate(baskets): 
                for item in self.basket2items[basket_id]:
                    if item not in repeat_basket_indicator:
                        repeat_basket_indicator[item] = []
                    repeat_basket_indicator[item].append(i)

            for i in range(self.min_basket_start, len(baskets)):
                label_basket = baskets[i]

                history_baskets = baskets[:i]
                history_baskets = history_baskets[-self.seq_len:]
                history_baskets_times = []
                for j, basket_id in enumerate(history_baskets):
                    history_baskets_times.append(j)

                # padding
                history_basket_masks = [False] * len(history_baskets) + [True] * (self.seq_len-len(history_baskets))
                history_baskets = history_baskets + [0] * (self.seq_len-len(history_baskets))
                history_baskets_times = history_baskets_times + [0] * (self.seq_len-len(history_baskets_times))

                historys = {}
                for item in repeat_items:
                    if item not in repeat_basket_indicator:
                        # novel item
                        # pif = 0.0
                        # gap_seq = [0] * self.history_len
                        # interval_seq = [0] * self.history_len
                        # history_vector = gap_seq + interval_seq + [pif]
                        # historys.append(history_vector)
                        continue 

                    index = np.argmax(np.array(repeat_basket_indicator[item]) >= i)
                    if np.max(np.array(repeat_basket_indicator[item])) < i:
                        index = len(repeat_basket_indicator[item])
                    input_history = repeat_basket_indicator[item][:index].copy()

                    pif = len(input_history) / len(baskets[:i])

                    input_history = input_history[-self.history_len:]
                    while len(input_history) < self.history_len:
                        input_history.insert(0, -1)
                    gap_seq = []
                    interval_seq = []
                    for x in input_history:
                        if x == -1:
                            gap_seq.append(0)
                        else:
                            gap_seq.append(i-x)
                    for j, x in enumerate(input_history[:-1]):
                        if x == -1:
                            interval_seq.append(0)
                        else:
                            interval_seq.append(input_history[j+1]-input_history[j])
                    if input_history[-1] == -1: # make up for the last one element
                        interval_seq.append(0)
                    else:
                        interval_seq.append(i-input_history[-1])
                    history_vector = gap_seq + interval_seq + [pif]

                    # historys.append(history_vector)
                    historys[item] = history_vector
                    self.max_interval = max(self.max_interval, max(interval_seq))
                
                ### set level training input
                ub_target = [0] * len(self.items_list)
                label_items = self.basket2items[label_basket]
                for position in label_items:
                    ub_target[position] = 1 

                basket_seqs.append(history_baskets)
                basket_times.append(history_baskets_times)
                basket_masks.append(history_basket_masks)

                ui_historys.append(historys)
                targets.append(ub_target)

        training_input = [basket_seqs, basket_times, basket_masks, ui_historys, targets]
        return training_input

    def generate_test_input(self):
        basket_seqs = []
        basket_times = []
        basket_masks = []

        ui_historys = []
        targets = [] 

        min_len = 100
        for user_id in self.test_data:
            baskets = self.user2basket[user_id]
            min_len = min(min_len, len(baskets))

            repeat_items = []
            for basket_id in baskets:
                for item in self.basket2items[basket_id]:
                    repeat_items.append(item)
            repeat_items = list(set(repeat_items)) 
            repeat_basket_indicator = {}
            for i, basket_id in enumerate(baskets): 
                for item in self.basket2items[basket_id]:
                    if item not in repeat_basket_indicator:
                        repeat_basket_indicator[item] = []
                    repeat_basket_indicator[item].append(i)

            history_baskets = baskets
            history_baskets = history_baskets[-self.seq_len:]
            history_baskets_times = []
            for i, basket_id in enumerate(history_baskets):
                history_baskets_times.append(i)
            
            # padding
            history_basket_masks = [False] * len(history_baskets) + [True] * (self.seq_len-len(history_baskets))
            history_baskets = history_baskets + [0] * (self.seq_len-len(history_baskets))
            history_baskets_times = history_baskets_times + [0] * (self.seq_len-len(history_baskets_times))

            historys = {}
            for item in repeat_items:
                if item not in repeat_basket_indicator:
                    # novel item
                    # pif = 0.0
                    # gap_seq = [0] * self.history_len
                    # interval_seq = [0] * self.history_len
                    # history_vector = gap_seq + interval_seq + [pif]
                    # historys.append(history_vector)
                    continue 

                index = len(repeat_basket_indicator[item])
                input_history = repeat_basket_indicator[item][:index].copy()

                pif = len(input_history) / len(baskets)

                input_history = input_history[-self.history_len:]
                while len(input_history) < self.history_len:
                    input_history.insert(0, -1)
                gap_seq = []
                interval_seq = []
                for x in input_history:
                    if x == -1:
                        gap_seq.append(0)
                    else:
                        gap_seq.append(len(baskets)-x)
                for j, x in enumerate(input_history[:-1]):
                    if x == -1:
                        interval_seq.append(0)
                    else:
                        interval_seq.append(input_history[j+1]-input_history[j])
                interval_seq.append(len(baskets)-input_history[-1])
                history_vector = gap_seq + interval_seq + [pif]

                # historys.append(history_vector)
                historys[item] = history_vector
                self.max_interval = max(self.max_interval, max(interval_seq))
            
            ub_target = [0] * len(self.items_list)
            label_items = self.test_data[user_id]
            for position in label_items:
                ub_target[position] = 1 

            basket_seqs.append(history_baskets)
            basket_times.append(history_baskets_times)
            basket_masks.append(history_basket_masks)
            
            ui_historys.append(historys)

            targets.append(ub_target)

        print("user min basket seq len", min_len)
        testing_input = [basket_seqs, basket_times, basket_masks, ui_historys, targets]
        return testing_input

    def __len__(self):
        return len(self.training_input[0])

    def __getitem__(self, idx):
        basket_seq = self.training_input[0][idx]
        basket_time = self.training_input[1][idx]
        basket_mask = self.training_input[2][idx]

        ui_history = self.training_input[3][idx]

        ### set level original input
        all_items = self.items_list
        target = self.training_input[4][idx]
        ui_history_vector = np.zeros((len(all_items), self.history_len+1))
        for item_pos in ui_history:
            ui_history_vector[item_pos] = np.array(ui_history[item_pos][self.history_len:])

        sample = {
            'basket_seq': torch.LongTensor(basket_seq),
            'basket_time': torch.LongTensor(basket_time),
            'basket_mask': torch.Tensor(basket_mask).bool(),

            'ui_history': torch.FloatTensor(ui_history_vector),
            'all_items': torch.LongTensor(all_items),

            "target": torch.FloatTensor(target)
        }
        return sample

    def sparse_to_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)