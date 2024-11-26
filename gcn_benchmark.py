from gnn_utils import trainGCN # custom dataset and trainer
import pytorch_lightning as pl
import numpy as np
from utils_binary import PatientDataset
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold
import argparse
print("Libraries loaded")

gcn_layers = [64, 32]
batch_size = 16
lr = 1e-4
algo = 'GCN'
dropout_val = 0.1
seed = 1
num_inp_ft = 1
print("Parameters loaded")

# reproducibility
pl.seed_everything(seed)
tot_epochs = 100
adj_init = 'Spearman'
# cancer_type = 'kipan'
# label_arg = 'neoplasm'
parser = argparse.ArgumentParser()
parser.add_argument("--cancer_type", type=str, help="cancer_type")
parser.add_argument("--label_arg", type=str, help="metadata")
args = parser.parse_args()

cancer_type = args.cancer_type
label_arg = args.label_arg

dataset = PatientDataset(adj_init, label_arg, cancer_type)
sample = dataset[0]
num_genes = sample.x.shape[1]
initial_adj = sample.edge_index

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
f1_foldwise = []
acc_foldwise = []
fold_num = 1
out_file_name = '/rcp/boulougo/DGM/dgm_out/GCN_' + str(cancer_type) + str(label_arg) + '_Spearman' '.txt'
f = open(out_file_name, 'w')
for train_idx, test_idx in kfold.split(np.zeros(len(dataset)), dataset.labels):
    train_idx = list(train_idx)
    np.random.shuffle(train_idx)
    val_idx = train_idx[:int(0.1*len(train_idx))]
    train_idx = train_idx[int(0.1*len(train_idx)):]

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    print("Fold Number: ", str(fold_num))
    print("samples in train dataset: ", len(train_ds))
    print("samples in validation dataset: ", len(val_ds))
    print("samples in test dataset: ", len(test_ds))

    full_labels = []
    for k in range(len(test_ds)):
        full_labels.append(test_ds[k].y)
    for k in range(len(val_ds)):
        full_labels.append(val_ds[k].y)
    for k in range(len(train_ds)):
        full_labels.append(train_ds[k].y)

    print("percentage of samples in dataset that are positive: ", sum(full_labels)/len(full_labels))
    pos_weight = (1 / (sum(full_labels)/len(full_labels)))
    print("Positive weight: ", pos_weight)

    model_name = 'GCN'
    save_loc = 'saved_models_gcn/' + model_name

    model, result = trainGCN(gcn_layers, tot_epochs, batch_size, lr, train_ds, val_ds, test_ds, dropout_val, num_inp_ft, algo, seed, save_loc, pos_weight, num_genes)

    f1_foldwise.append(result['test'][0]['test_f1'])
    acc_foldwise.append(result['test'][0]['test_acc'])
    fold_num += 1

f.write("Num Fold with Test F1 Scores: " + str(len(f1_foldwise)) + " out of 5")
f.write("Final F1 scores (Mean +- STD): " + str(np.mean(f1_foldwise)) + " +- " + str(np.std(f1_foldwise)))
f.write("Final Acc scores (Mean +- STD): " + str(np.mean(acc_foldwise)) + " +- " + str(np.std(acc_foldwise)))
f.close()