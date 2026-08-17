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
import sys

import itertools
from scipy.sparse import coo_matrix
from utils import *
from torch.utils.data import DataLoader

from Dataset import Data
from models import * 
import os

def model_train(args, epoch, batch_data, model_G, model_D, opt_G, opt_D, g_bce_criterion, g_ae_criterion, d_criterion):
    if epoch % args.train_DG[2] < args.train_DG[0]: 
        # update D 
        trainD, trainG = True, False
    else: 
        # update G
        trainD, trainG = False, True 

    # trainD, trainG = False, True

    if trainD:
        model_D.train()
        opt_D.zero_grad()
    else:
        model_D.eval()

    if trainG:
        model_G.train()
        opt_G.zero_grad()
    else:
        model_G.eval()

    basket_seq, basket_time, basket_mask, ui_history, all_items, target = batch_data
    rec_score, pattern_decoded = model_G(batch_data)

    batch_size = basket_seq.size()[0]
    pos_preds, pos_labels = [], []
    neg_preds, neg_labels = [], []
    neg_rewards = []
    item_emb, hyper_emb = model_D.update() 
    for i in range(batch_size):
        probs = rec_score[i]
        per_target = target[i]
        per_target = [x[0] for x in np.nonzero(per_target).tolist()]

        #negative aggregation
        sample_hedge = np.random.choice(np.arange(probs.shape[0]), len(per_target), p = probs.detach().cpu().numpy())

        per_target = torch.LongTensor(per_target).to(args.device)
        sample_hedge = torch.LongTensor(sample_hedge).to(args.device)

        pos_input = [basket_seq[i], basket_time[i], basket_mask[i], per_target]
        pos_logits = model_D(pos_input, item_emb, hyper_emb)
        pos_preds.append(pos_logits)
        pos_labels.append(1)

        neg_input = [basket_seq[i], basket_time[i], basket_mask[i], sample_hedge]
        neg_logits = model_D(neg_input, item_emb, hyper_emb)
        neg_preds.append(neg_logits)
        neg_labels.append(0)
        neg_rewards.append(torch.mean((2*neg_logits-1) * torch.log(probs[sample_hedge])))

    labels = pos_labels + neg_labels
    logits = pos_preds + neg_preds
    labels = torch.FloatTensor(labels).to(args.device)
    logits = torch.stack(logits)
    d_loss = d_criterion(logits.squeeze(), labels.squeeze())


    neg_rewards = torch.stack(neg_rewards)
    bce_loss = g_bce_criterion(rec_score, target)
    ae_loss = torch.sqrt(g_ae_criterion(ui_history, pattern_decoded))
    g_loss = bce_loss + args.alpha * ae_loss - (torch.mean(neg_rewards))

    if trainD:
        d_loss.backward()
        opt_D.step()
        
    if trainG:
        g_loss.backward()
        opt_G.step()

    return d_loss.item(), g_loss.item()

def model_eval(args, model_G, data, device):
    acc_2 = []
    acc_3 = []

    acc_10 = []
    acc_20 = []
    acc_30 = []
    acc_40 = []
    acc_50 = []

    for idx in range(len(data.testing_input[0])):

        basket_seq = data.testing_input[0][idx]
        basket_time = data.testing_input[1][idx]
        basket_mask = data.testing_input[2][idx]

        ui_history = data.testing_input[3][idx]
        all_items = data.items_list

        target = data.testing_input[4][idx]

        ui_history_vector = np.zeros((len(all_items), data.history_len+1))
        for item_pos in ui_history:
            ui_history_vector[item_pos] = np.array(ui_history[item_pos][data.history_len:])


        basket_seq = torch.LongTensor(basket_seq).unsqueeze(0).to(device)
        basket_time = torch.LongTensor(basket_time).unsqueeze(0).to(device)
        basket_mask = torch.Tensor(basket_mask).bool().unsqueeze(0).to(device)

        ui_history = torch.FloatTensor(ui_history_vector).unsqueeze(0).to(device)
        all_items = torch.LongTensor(all_items).unsqueeze(0).to(device)

        test_batch_data = [basket_seq, basket_time, basket_mask, ui_history, all_items]
        scores = model_G.test_score(test_batch_data).cpu().data.numpy().tolist()

        all_items_list = data.items_list
        result = {}
        for index in range(len(all_items_list)):
            item = all_items_list[index]
            result[item] = scores[index]

        result = sorted(result.items(), key=lambda item: item[1], reverse=True)
        topN = [int(item[0]) for item in result]

        target = torch.FloatTensor(target)
        ground_truths = []
        x = np.nonzero(target).tolist()
        for gt in x:
            ground_truths.append(gt[0])

        ground_truths_set = set(ground_truths)



        # small for subset 
        # 2-size bundle design
        if len(ground_truths_set) >= 2:
            design_bundle = topN[:2]
            design_bundle_set = set(design_bundle)

            if design_bundle_set.issubset(ground_truths_set):
                per_acc = 1 
            else:
                per_acc = 0
            acc_2.append(per_acc)

        # 3-size bundle design
        if len(ground_truths_set) >=3:
            design_bundle = topN[:3]
            design_bundle_set = set(design_bundle)

            if design_bundle_set.issubset(ground_truths_set):
                per_acc = 1 
            else:
                per_acc = 0
            acc_3.append(per_acc)




        # large for superset
        # 10-size bundle design 
        if len(ground_truths_set) <= 10:
            design_bundle = topN[:10]
            design_bundle_set = set(design_bundle)

            if ground_truths_set.issubset(design_bundle_set):
                per_acc = 1
            else:
                per_acc = 0
            acc_10.append(per_acc)

        # 20-size bundle design 
        if len(ground_truths_set) <= 20:
            design_bundle = topN[:20]
            design_bundle_set = set(design_bundle)

            if ground_truths_set.issubset(design_bundle_set):
                per_acc = 1
            else:
                per_acc = 0
            acc_20.append(per_acc)


        # 30-size bundle design 
        if len(ground_truths_set) <= 30:
            design_bundle = topN[:30]
            design_bundle_set = set(design_bundle)

            if ground_truths_set.issubset(design_bundle_set):
                per_acc = 1
            else:
                per_acc = 0
            acc_30.append(per_acc)


        # 40-size bundle design 
        if len(ground_truths_set) <= 40:
            design_bundle = topN[:40]
            design_bundle_set = set(design_bundle)

            if ground_truths_set.issubset(design_bundle_set):
                per_acc = 1
            else:
                per_acc = 0
            acc_40.append(per_acc)


        # 50-size bundle design 
        if len(ground_truths_set) <= 50:
            design_bundle = topN[:50]
            design_bundle_set = set(design_bundle)

            if ground_truths_set.issubset(design_bundle_set):
                per_acc = 1
            else:
                per_acc = 0
            acc_50.append(per_acc)



    acc_2 = np.array(acc_2)
    acc_3 = np.array(acc_3)

    acc_10 = np.array(acc_10)
    acc_20 = np.array(acc_20)
    acc_30 = np.array(acc_30)
    acc_40 = np.array(acc_40)
    acc_50 = np.array(acc_50)

    return np.mean(acc_2, axis=0), np.mean(acc_3, axis=0), np.mean(acc_10, axis=0), np.mean(acc_20, axis=0), \
        np.mean(acc_30, axis=0), np.mean(acc_40, axis=0), np.mean(acc_50, axis=0)
 