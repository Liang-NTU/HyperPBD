import csv 
import numpy as np
import pickle
import os
import pandas as pd
import torch

def load_data(train_name, test_name, min_trans):
    train_file = open(train_name, "r")
    
    # filter user less than min_trans trans
    csvreader = csv.reader(train_file)
    header = next(csvreader)

    user2trans = {}
    for row in csvreader:
        user = int(row[0])
        basket = int(row[1])
        if user not in user2trans:
            user2trans[user] = set()
        user2trans[user].add(basket)
    selected_users = set()
    for user in user2trans:
        trans = len(user2trans[user])
        if trans <= min_trans:
            continue
        selected_users.add(user)
    train_file.seek(0)
    print("raw users", len(user2trans), "selected users", len(selected_users))

    # to ids
    # user and item to ids
    csvreader = csv.reader(train_file)
    header = next(csvreader)

    user_id_mapper = {}
    item_id_mapper = {}
    
    user2basket = {}
    basket2items ={}
    for row in csvreader:
        user = int(row[0])
        basket = int(row[1])
        item = int(row[2])
        if user not in selected_users:
            continue

        if user not in user_id_mapper:
            user_id_mapper[user] = len(user_id_mapper)
        if item not in item_id_mapper:
            item_id_mapper[item] = len(item_id_mapper)

        basket_id = str(user_id_mapper[user]) + "_" + str(basket) 
        if user_id_mapper[user] not in user2basket:
            user2basket[user_id_mapper[user]] = []
        user2basket[user_id_mapper[user]].append(basket_id)
        if basket_id not in basket2items:
            basket2items[basket_id] = []
        basket2items[basket_id].append(item_id_mapper[item])

    # basket to ids
    basket_id_mapper = {}
    tuple_id_mapper = {}

    for basket_id in basket2items:
        item_list = basket2items[basket_id]
        item_tuple = tuple(item_list)
        if item_tuple not in tuple_id_mapper:
            tuple_id_mapper[item_tuple] = len(tuple_id_mapper)
        basket_id_mapper[basket_id] = tuple_id_mapper[item_tuple]

    # clear the raw data with unified ids, u2b, b2i, is
    clear_user2basket = {}
    clear_basket2items = {}
    for basket_id in basket2items:
        mapper_id = basket_id_mapper[basket_id]
        clear_basket2items[mapper_id] = basket2items[basket_id]

    for user_id in user2basket:
        baskets = user2basket[user_id]
        pure_basket = []
        _index = 0
        while _index < len(baskets):
            basket_id = baskets[_index]
            mapper_id = basket_id_mapper[basket_id]

            basket_size = len(clear_basket2items[mapper_id])
            _index += basket_size
            pure_basket.append(mapper_id)
        clear_user2basket[user_id] = pure_basket

    clear_items_list = []
    clear_items_set = set()
    for mapper_id in clear_basket2items:
        item_list = clear_basket2items[mapper_id]
        for item in item_list:
            if item not in clear_items_set:
                clear_items_list.append(item)
                clear_items_set.add(item)
        
    test_file = open(test_name, "r")
    csvreader = csv.reader(test_file)
    header = next(csvreader)
    test_user2items = {}
    for row in csvreader:
        user = int(row[0])
        basket = int(row[1])
        item = int(row[2])
        if user not in test_user2items:
            test_user2items[user] = []
        test_user2items[user].append(item)

    clear_test_data = {}
    for user in test_user2items:
        if user not in user_id_mapper:
            continue
        item_list = test_user2items[user]
        
        clear_user_id = user_id_mapper[user]
        clear_items = []
        for item in item_list:
            if item not in item_id_mapper:
                continue
            clear_items.append(item_id_mapper[item])
        
        if len(clear_items) < 1:
            continue
        clear_test_data[clear_user_id] = clear_items

    # save mappers
    file_path = "./mapper/" + train_name + "/ui_mapper.pkl"
    if not os.path.exists(os.path.dirname(file_path)):
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, "wb") as file:
        pickle.dump((user_id_mapper, item_id_mapper), file)

    return clear_user2basket, clear_basket2items, clear_items_list, clear_test_data

def recall_k(y_true, y_pred, k):
    a = len(set(y_pred[:k]).intersection(set(y_true)))
    b = len(set(y_true))
    return a/b

def ndcg_k(y_true, y_pred, k):
    a = 0
    for i,x in enumerate(y_pred[:k]):
        if x in y_true:
            a += 1/np.log2(i+2)
    
    b = 0
    # for i in range(k): # range(min(k,len(set(y_true)))):
    for i in range(len(set(y_true))):
        b += 1/np.log2(i+2)
    return a/b