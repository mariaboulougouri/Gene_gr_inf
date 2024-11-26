
# libraries
import numpy as np
from utils_multiclass import trainGCN, PatientDataset # custom dataset and trainer
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold
import wandb
import argparse

# reproducibility
pl.seed_everything(1)

# hyperparams
tot_epochs = 100
dropout_val = 0.0
num_inp_ft = 1

# command line arguments 
parser = argparse.ArgumentParser() 
parser.add_argument("--adj_init", type=str, default='Spearman', help="Adjacency matrix initialization")
parser.add_argument("--corrth", type=int, default=1, help="No. of stdevs to threshold adj. matrix")
parser.add_argument("--ffun_alg", type=str, default='mlp', help="Graph function algorithm - classification")
parser.add_argument("--gfun_alg", type=str, default='gcn', help="Graph function algorithm - diffusion")
parser.add_argument("--bs", type=int, default=1, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--num_nodes", type=int, default=32, help="Number of nodes")
parser.add_argument("--cancer_type", type=str, default='coadread', help="what cancer type to use") # coadred, brca, luad
parser.add_argument("--label", type=str, default='pathological_stage', help="which label to classify") # follow_up, venous, lymph, history
parser.add_argument("--edge_perc", type=float, default=12.5, help="percentage edges to keep globally")
args = parser.parse_args()

corrth = args.corrth
bs = args.bs
lr = args.lr
adj_init = args.adj_init # None, Pearson, Spearman, Zeros, Ones
ffun_alg = args.ffun_alg # mlp, gcn, gat
gfun_alg = args.gfun_alg # sage, gcn, gat
label_arg = args.label
num_nodes = args.num_nodes

gcn_layers = [64, 32]

dataset = PatientDataset(adj_init, label_arg, args.cancer_type, corrth)
# get first sample
sample = dataset[0]
num_genes = sample.x.shape[1]
num_outs = sample.num_outs
initial_adj = sample.edge_index
weights = sample.weights

# # split into train and test in 80-20 ratio
# train_ds, test_ds = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset) - int(0.8*len(dataset))])

# do kfold cross validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

f1_foldwise = []
acc_foldwise = []

out_file_name = '/rcp/boulougo/DGM/dgm_out/DGM_' + str(args.cancer_type) + '_' + str(args.label) + '_' + str(args.adj_init) + '.txt'
f = open(out_file_name, 'w')

logger_name = 'CT: ' + str(args.cancer_type) + ' Label: ' + label_arg  + ' Init: ' + str(adj_init) + ' Params: [LR: ' + str(lr) + ', FF: ' + str(ffun_alg) + ', GF: ' + str(gfun_alg) + ', BS: ' + str(bs) + ', EP: ' + str(args.edge_perc) + ']'
wandb_logger = WandbLogger(log_model="all", project="DGM", name=logger_name)

# split train, test with straitfy
fold_num = 1
for train_idx, test_idx in kfold.split(np.zeros(len(dataset)), dataset.labels):
    # split train_idx into train and validation in 90-10 ratio
    train_idx = list(train_idx)
    np.random.shuffle(train_idx)
    val_idx = train_idx[:int(0.1*len(train_idx))]
    train_idx = train_idx[int(0.1*len(train_idx)):]

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    print("Fold Number: ", str(fold_num))

    # start a new wandb run to track this script
    model_name = 'CT: ' + str(args.cancer_type) + ' Label: ' + label_arg  + ' Init: ' + str(adj_init) + ' Params: [LR: ' + str(lr) + ', FF: ' + str(ffun_alg) + ', GF: ' + str(gfun_alg) + ', BS: ' + str(bs) + ', FN: ' + str(fold_num) + ', EP: ' + str(args.edge_perc) + ']'
    # model parameters
    save_loc = '/rcp/boulougo/DGM/saved_models/' + model_name

    # train model
    model, result = trainGCN(num_nodes, tot_epochs, bs, lr, save_loc, wandb_logger, train_ds, val_ds, test_ds, dropout_val, ffun_alg, gfun_alg, fold_num, initial_adj, args.edge_perc, label_arg, int(dataset.num_outs), adj_init, dataset.weights, dataset.num_genes, args.cancer_type, f)

    f1_foldwise.append(result['test'][0]['test_f1'])
    acc_foldwise.append(result['test'][0]['test_acc'])

    fold_num += 1

wandb.finish()
f.write("Num Fold with Test F1 Scores: " + str(len(f1_foldwise)) + " out of 5")
f.write("Final F1 scores (Mean +- STD): " + str(np.mean(f1_foldwise)) + " +- " + str(np.std(f1_foldwise)))
f.write("Final Acc scores (Mean +- STD): " + str(np.mean(acc_foldwise)) + " +- " + str(np.std(acc_foldwise)))
f.close()