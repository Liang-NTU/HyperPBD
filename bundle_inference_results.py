from mapper.mapper_analysis import load_mapper
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import os
# from tqdm import tqdm
import utils
from training import model_train, model_eval
import parsers
import pickle as pkl
from Dataset import *
from models import *

def train(args):
    train_DG = args.train_DG.split(":")

    args.train_DG = [int(train_DG[0]), int(train_DG[1]), int(train_DG[0]) + int(train_DG[1])]
    os.makedirs(
        f"./logs/{args.train_name}/{args.train_name}_{args.train_DG[0]}_{args.train_DG[1]}_{args.alpha}_{args.n_layers}_{args.K}_{args.emb_dim}",
        exist_ok=True)

    if args.fix_seed:
        np.random.seed(0)
        torch.manual_seed(0)
        print("seed fixed for reproducibility")
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
    args.device = device

    data = Data(args)
    loader = DataLoader(data, batch_size=args.bs, shuffle=True)
    print("Total number of items:", len(data.items_list)) # 2798

    # set model
    model_G = DIG_Generator(num_items=data.num_items, seq_len=data.seq_len, history_len=data.history_len,
                            emb_dim=args.emb_dim, device=device, layer_num=args.n_layers, K=args.K)

    model_D = DIG_Discriminator(num_items=data.num_items, seq_len=data.seq_len, emb_dim=args.emb_dim,
                                device=device, layer_num=args.n_layers)

    model_G = model_G.to(device)
    opt_G = torch.optim.Adam(model_G.parameters(), lr=args.lr, weight_decay=1e-5)

    model_D = model_D.to(device)
    opt_D = torch.optim.Adam(model_D.parameters(), lr=args.lr, weight_decay=1e-5)

    input_x = data.input_x.to(device)
    norm_HH = data.norm_HH.to(device)
    norm_HG = data.norm_HG.to(device)
    model_G.set_paras(input_x, norm_HH, norm_HG)
    model_D.set_paras(input_x, norm_HH, norm_HG)

    PATH = f"./models/{args.train_name}/{args.train_name}_{args.train_DG[0]}_{args.train_DG[1]}_{args.alpha}_{args.n_layers}_{args.K}_{args.emb_dim}"
    j = 1
    if os.path.exists(PATH + "/model_G" + "_" + str(j) + ".pt"):
        model_G.load_state_dict(torch.load(PATH + "/model_G" + "_" + str(j) + ".pt", map_location=device))

        # model_G.eval()
        # acc_2, acc_3, acc_10, acc_20, acc_30, acc_40, acc_50 = model_eval(args, model_G, data, device)
        # print("Test Acc_2:", acc_2, "Test Acc_3:", acc_3, "Test Acc_10:", acc_10, "Test Acc_20:", acc_20,
        #       "Test Acc_30:", acc_30, "Test Acc_40:", acc_40, "Test Acc_50:", acc_50)
        # print("----------------------------------------------------------------------")

    model_G.eval()
    inference_results = []
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

        history = []
        for basket_id in basket_seq[~basket_mask]:
            trans = data.basket2items[basket_id.item()]
            history.append(trans)
        user_inference_result = [history, topN[:10]]
        inference_results.append(user_inference_result)

    output_path = "./mapper/inference_results_" + str(j) + ".pt"
    torch.save(inference_results, output_path)

if __name__ == "__main__":
    args = parsers.parse_args()
    train(args)