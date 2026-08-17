import torch
import pandas as pd
import pickle

def load_mapper(mapper_path, user_path, item_path):
    with open(mapper_path, "rb") as file:
        ui_mapper = pickle.load(file)

    user_id_mapper, item_id_mapper = ui_mapper

    user_info = torch.load(user_path)
    item_info = torch.load(item_path)
    item2feat = {}
    for feat_dict in item_info:
        item_id = feat_dict["item_id"]
        feat = feat_dict["feature"]
        item2feat[item_id] = {"feature": feat}
    item_info = item2feat
    print("type of item info:", type(item_info))
    print("item_info content:", item_info[0])

    new_item_info = {}
    item_error = 0
    item_error_set = []
    for item_id in item_id_mapper:
        new_item_id = item_id_mapper[item_id]
        try:
            feats = item_info[item_id]
            new_item_info[new_item_id] = feats
        except Exception as e:
            item_error += 1
            item_error_set.append(item_id)
    print("Item inde error:", item_error, "*********** Consistent with Mengkun ************")

    return new_item_info

def load_inference_results(path):
    inference_results = torch.load(path)
    return inference_results

def semantic_match(inference_results, item2feat):

    total_match = []
    feat_match = []
    for user_inference_result in inference_results:
        history = user_inference_result[0]
        topN = user_inference_result[1]

        feat_history = []
        for trans in history:
            trans_feat = []
            for item in trans:
                if item not in item2feat:
                    # print("Item map error:", item)
                    continue
                feat = item2feat[item]
                feat["new_item_id"] = item
                trans_feat.append(feat)
            if trans_feat:
                feat_history.append(trans_feat)

        feat_topN = []
        for item in topN:
            if item not in item2feat:
                # print("Item map error:", item)
                continue
            feat = item2feat[item]
            feat["new_item_id"] = item
            feat_topN.append(feat)

        if feat_history and feat_topN:
            feat_match.append([feat_history, feat_topN])

        total_match.append([feat_history, feat_topN])

    print("len of feat_match:", len(feat_match))
    print("len of total_match:", len(total_match))

    return feat_match, total_match

if __name__ == '__main__':
    mapper_path = "./data/cosmetics_history.csv/ui_mapper.pkl"
    user_path = "./data/cosmfeat_history.csv/user_info.pt"
    item_path = "./data/cosmfeat_history.csv/item_info.pt"

    item2feat = load_mapper(mapper_path, user_path, item_path)

    infer_path = "./inference_results_1.pt"
    inference_results = load_inference_results(infer_path)

    feat_match, total_match = semantic_match(inference_results, item2feat)

    output_path = "./dig_feat_match.pt"
    torch.save(total_match, output_path)

