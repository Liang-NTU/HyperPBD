import argparse
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser()
    
    ##### training hyperparameter #####
    parser.add_argument("--train_name", type=str, default='./data/cosmetics_history.csv', help='dataset name')
    parser.add_argument("--test_name", type=str, default='./data/cosmetics_future.csv', help='dataset name')
    parser.add_argument("--seed", dest='fix_seed', action='store_const', default=True, const=True, help='Fix seed for reproducibility and fair comparison.')
    parser.add_argument("--gpu", type=int, default=0, help='gpu number. -1 if cpu else gpu number')
    parser.add_argument("--exp_num", default=5, type=int, help='number of experiments')
    parser.add_argument("--epochs", default=100, type=int, help='number of epochs')
    parser.add_argument("--bs", default=512, type=int, help='batch size')
    parser.add_argument("--train_DG", default="1:4", type=str, help='update ratio in epochs (D updates:G updates)')
    parser.add_argument("--lr", default=0.001, type=float, help='learning rate')
    parser.add_argument("--emb_dim", default = 64, type=int, help='dimension')

    #hgnn parameters
    parser.add_argument("--n_layers", default=3, type=int, help='number of layers')
    parser.add_argument("--K", default=5, type=int, help='number of layers')
    parser.add_argument("--alpha", default=0.1, type=float)
    parser.add_argument("--history_len", default=10, type=int)
    parser.add_argument("--seq_len", default=10, type=int)
    parser.add_argument("--min_trans", default=0, type=int)
    parser.add_argument("--min_basket_start", default=1, type=int)
    parser.add_argument("--min_epoch", default=0, type=int)
    parser.add_argument("--interval_epoch", default=5, type=int)
    opt = parser.parse_known_args()[0]

    return opt
     
 