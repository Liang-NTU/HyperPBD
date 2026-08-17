from copyreg import pickle
import torch 
import torch.nn as nn
import numpy as np
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

    args.train_DG = [int(train_DG[0]), int(train_DG[1]), int(train_DG[0])+int(train_DG[1])]
    os.makedirs(f"./logs/{args.train_name}/{args.train_name}_{args.train_DG[0]}_{args.train_DG[1]}_{args.alpha}_{args.n_layers}_{args.K}_{args.emb_dim}", exist_ok=True)

    if args.fix_seed:
        np.random.seed(0)
        torch.manual_seed(0)
        print("seed fixed for reproducibility")
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
    args.device = device

    best_acc_2_all = []
    best_acc_3_all = []

    best_acc_10_all = []
    best_acc_20_all = []
    best_acc_30_all = []
    best_acc_40_all = []
    best_acc_50_all = []

    for j in range(args.exp_num): 
        best_acc_2 = 0 
        best_acc_3 = 0

        best_acc_10 = 0
        best_acc_20 = 0
        best_acc_30 = 0
        best_acc_40 = 0
        best_acc_50 = 0

        f_log = open(f"./logs/{args.train_name}/{args.train_name}_{args.train_DG[0]}_{args.train_DG[1]}_{args.alpha}_{args.n_layers}_{args.K}_{args.emb_dim}/exp_{j}.txt", "w")
        f_log.write(f"args: {args}\n")
        
        data = Data(args)
        loader = DataLoader(data, batch_size=args.bs, shuffle=True)

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
        if os.path.exists(PATH + "/model_G" + "_" + str(j) + ".pt"):
            model_G.load_state_dict(torch.load(PATH + "/model_G" + "_" + str(j) + ".pt", map_location=device))

            model_G.eval()
            acc_2, acc_3, acc_10, acc_20, acc_30, acc_40, acc_50 = model_eval(args, model_G, data, device)
            print("Test Acc_2:", acc_2, "Test Acc_3:", acc_3, "Test Acc_10:", acc_10, "Test Acc_20:", acc_20, "Test Acc_30:", acc_30, "Test Acc_40:", acc_40, "Test Acc_50:", acc_50)
            print("----------------------------------------------------------------------")
    
            best_acc_2_all.append(acc_2)
            best_acc_3_all.append(acc_3)

            best_acc_10_all.append(acc_10)
            best_acc_20_all.append(acc_20)
            best_acc_30_all.append(acc_30)
            best_acc_40_all.append(acc_40)
            best_acc_50_all.append(acc_50)            
            continue


        g_bce_criterion = torch.nn.BCELoss(reduction="sum")
        g_ae_criterion = nn.MSELoss(reduction="sum")

        d_criterion = nn.BCELoss()
        for epoch in range(args.epochs):     
       
            d_loss_sum, g_loss_sum = 0.0, 0.0
            for i_batch, sample_batched in enumerate(loader):
                basket_seq = sample_batched['basket_seq'].to(device)
                basket_time = sample_batched['basket_time'].to(device)
                basket_mask = sample_batched['basket_mask'].to(device)

                ui_history = sample_batched['ui_history'].to(device)
                all_items = sample_batched['all_items'].to(device)

                target = sample_batched['target'].to(device)

                batch_data = [basket_seq, basket_time, basket_mask, ui_history, all_items, target]

                d_loss, g_loss = model_train(args, epoch, batch_data, model_G, model_D, opt_G, opt_D, g_bce_criterion, g_ae_criterion, d_criterion)
                d_loss_sum += d_loss
                g_loss_sum += g_loss

            print("**********************************************************************")
            print("epoch num:", epoch, "g_loss:", g_loss_sum/(len(data)/args.bs), "d_loss", d_loss_sum/(len(data)/args.bs), g_loss_sum)
            
            f_log.write(f"*******************************************************************\n")
            f_log.write(f"epoch {epoch} g_loss: {g_loss_sum/(len(data)/args.bs)} d_loss: {d_loss_sum/(len(data)/args.bs)}\n")
            f_log.flush()
            
            if epoch >= args.min_epoch and epoch % args.interval_epoch == 0:
                model_G.eval()
                acc_2, acc_3, acc_10, acc_20, acc_30, acc_40, acc_50 = model_eval(args, model_G, data, device)
                print("epoch:", epoch, "Test Acc_2:", acc_2, "Test Acc_3:", acc_3, "Test Acc_10:", acc_10, "Test Acc_20:", acc_20, "Test Acc_30:", acc_30, "Test Acc_40:", acc_40, "Test Acc_50:", acc_50)
                print("----------------------------------------------------------------------")

                f_log.write(f"epoch {epoch} Test Acc_2: {acc_2} Test Acc_3: {acc_3} Test Acc_10: {acc_10} Test Acc_20: {acc_20} Test Acc_30: {acc_30} Test Acc_40: {acc_40} Test Acc_50: {acc_50}\n")
                f_log.write(f"-----------------------------------------------------------------\n")
                f_log.flush()

                if acc_2 > best_acc_2:
                    best_acc_2 = acc_2 
                if acc_3 > best_acc_3:
                    best_acc_3 = acc_3

                if acc_10 > best_acc_10:
                    best_acc_10 = acc_10
                    best_acc_20 = acc_20
                    best_acc_30 = acc_30
                    best_acc_40 = acc_40
                    best_acc_50 = acc_50

                    PATH = f"./models/{args.train_name}/{args.train_name}_{args.train_DG[0]}_{args.train_DG[1]}_{args.alpha}_{args.n_layers}_{args.K}_{args.emb_dim}"
                    os.makedirs(PATH, exist_ok=True)
                    torch.save(model_G.state_dict(), PATH + "/model_G" + "_" + str(j) + ".pt")

                # if acc_20 > best_acc_20:
                #     best_acc_20 = acc_20
                # if acc_30 > best_acc_30:
                #     best_acc_30 = acc_30
                # if acc_40 > best_acc_40:
                #     best_acc_40 = acc_40
                # if acc_50 > best_acc_50:
                #     best_acc_50 = acc_50

        f_log.close()

        best_acc_2_all.append(best_acc_2)
        best_acc_3_all.append(best_acc_3)

        best_acc_10_all.append(best_acc_10)
        best_acc_20_all.append(best_acc_20)
        best_acc_30_all.append(best_acc_30)
        best_acc_40_all.append(best_acc_40)
        best_acc_50_all.append(best_acc_50)

    with open(f"./logs/{args.train_name}/{args.train_name}_{args.train_DG[0]}_{args.train_DG[1]}_{args.alpha}_{args.n_layers}_{args.K}_{args.emb_dim}/exp_all.txt", "w") as e_log:  
        e_log.write(f"{best_acc_2_all[0]}\n{best_acc_3_all[0]}\n")
        e_log.write(  " | Acc_2"+ ", mean: " + "{:.6f}".format(np.mean(best_acc_2_all))  + "   std:"+ "{:.6f}\n".format(np.std(best_acc_2_all)))
        e_log.write(  " | Acc_3@"+ ", mean: " + "{:.6f}".format(np.mean(best_acc_3_all))  + "   std:"+ "{:.6f}\n".format(np.std(best_acc_3_all)))

        e_log.write(  " | Acc_10@"+ ", mean: " + "{:.6f}".format(np.mean(best_acc_10_all))  + "   std:"+ "{:.6f}\n".format(np.std(best_acc_10_all)))
        e_log.write(  " | Acc_20@"+ ", mean: " + "{:.6f}".format(np.mean(best_acc_20_all))  + "   std:"+ "{:.6f}\n".format(np.std(best_acc_20_all)))
        e_log.write(  " | Acc_30@"+ ", mean: " + "{:.6f}".format(np.mean(best_acc_30_all))  + "   std:"+ "{:.6f}\n".format(np.std(best_acc_30_all)))
        e_log.write(  " | Acc_40@"+ ", mean: " + "{:.6f}".format(np.mean(best_acc_40_all))  + "   std:"+ "{:.6f}\n".format(np.std(best_acc_40_all)))
        e_log.write(  " | Acc_50@"+ ", mean: " + "{:.6f}".format(np.mean(best_acc_50_all))  + "   std:"+ "{:.6f}\n".format(np.std(best_acc_50_all)))    

if __name__ == "__main__":
    args = parsers.parse_args()
    train(args)