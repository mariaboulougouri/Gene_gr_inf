# libraries
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Dataset, Data
import os
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score
import lightning as L
import torchmetrics
from scipy import sparse
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.nn import GATConv, EdgeConv,  GCNConv, GATConv, GINEConv
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch.nn import ModuleList

from layers import *
if (not os.environ.get("USE_KEOPS")) or os.environ.get("USE_KEOPS")=="False":
    from layers_dense import *

class PatientDataset(Dataset):
    def __init__(self, adj_init, label_arg, cancer_type):
        super().__init__()

        print("loading dataset")

        if cancer_type == 'coadread':
            folder = 'TCGA_COADREAD'
        elif cancer_type == 'brca':
            folder = 'TCGA_BRCA'
        elif cancer_type == 'hnsc':
            folder = 'TCGA_HNSC'
        elif cancer_type == 'kipan':
            folder = 'TCGA_KIPAN'
        elif cancer_type == 'luad':
            folder = 'TCGA_LUAD'
        elif cancer_type == 'stes':
            folder = 'TCGA_STES'
        elif cancer_type == 'gbmlgg':
            folder = 'TCGA_GBMLGG'

        if label_arg == 'vital_status':
            metadata = 'patient.vital_status' # one example
        elif label_arg == 'follow_up':
            metadata = 'patient.follow_ups.follow_up.vital_status' # one example
        elif label_arg == 'neoplasm':
            metadata = 'patient.person_neoplasm_cancer_status' # one example

        # data_path = 'data_dgm/' + folder + '/'
        data_path = '/rcp/boulougo/DGM/Raw_data/' + folder + '/'
        df_labels = pd.read_csv(data_path + "filtered_metaclinical.txt", sep= '\t', index_col=0)   

        df = pd.read_csv(data_path + "filtered_rnaseq_zerosandlowvar_removed.txt", sep= '\t', index_col=0, header=0)

        df_labels = df_labels[metadata]
        df_labels = df_labels.dropna()
        df.index = df.index.str[:12].str.lower()
        df = df.loc[df_labels.index]

        # convert df_labels into a tensor with 0,1
        df_labels = df_labels.astype('category')
        df_labels = df_labels.cat.codes

        self.labels = np.asarray(df_labels.values)

        print("Num Unique Labels: ", len(np.unique(self.labels)))

        self.num_outs = len(list(set(list(self.labels))))
        print("Num Unique Labels: ", self.num_outs)

        # number of 0, and 1 from self.labels to make self.weights
        num_0 = np.sum(self.labels == 0) / len(self.labels)
        num_1 = np.sum(self.labels == 1) / len(self.labels)
        self.weights = [1/ num_0, 1/ num_1]
        self.pos_weight = [1/ num_0]

        # convert df into a feature matrix 
        self.features = np.asarray(df.values)
        self.num_genes = len(self.features[0])

        if adj_init == 'Pearson':
            # load adj matrix
            pearson_adj_matrix = df.corr(method='pearson')
            new_adj_matrix = pearson_adj_matrix.dropna(axis=0, how='all')
            new_adj_matrix = new_adj_matrix.dropna(axis=1, how='all')
            p_mat = new_adj_matrix.to_numpy().flatten()
            new_adj_matrix = new_adj_matrix.to_numpy()
            mean = np.mean(p_mat) 
            st = np.std(p_mat)
            lth = mean-st 
            rth = mean+st
            new_adj_matrix[(new_adj_matrix < lth) | (new_adj_matrix > rth)] = 1
            new_adj_matrix[(new_adj_matrix >= lth) & (new_adj_matrix <= rth)] = 0
            pearson_adj_matrix_sparse = sparse.csr_matrix(new_adj_matrix)
            self.ei_pea, self.ew_pea = from_scipy_sparse_matrix(pearson_adj_matrix_sparse)
        elif adj_init == 'Spearman':
            spearman_adj_matrix = df.corr(method='spearman')
            new_adj_matrix = spearman_adj_matrix.dropna(axis=0, how='all')
            new_adj_matrix = new_adj_matrix.dropna(axis=1, how='all')
            p_mat = new_adj_matrix.to_numpy().flatten()
            new_adj_matrix = new_adj_matrix.to_numpy()
            mean = np.mean(p_mat) 
            st = np.std(p_mat)
            lth = mean-st 
            rth = mean+st
            new_adj_matrix[(new_adj_matrix < lth) | (new_adj_matrix > rth)] = 1
            new_adj_matrix[(new_adj_matrix >= lth) & (new_adj_matrix <= rth)] = 0
            spearman_adj_matrix_sparse = sparse.csr_matrix(new_adj_matrix)
            self.ei_sp, self.ew_sp = from_scipy_sparse_matrix(spearman_adj_matrix_sparse)
        elif adj_init == 'None':
            pass
        elif adj_init == 'Ones':
            # make an adj matrix full of ones
            adj_matrix = np.ones((self.num_genes, self.num_genes))
            adj_matrix_sparse = sparse.csr_matrix(adj_matrix)
            self.ei_ones, self.ew_ones = from_scipy_sparse_matrix(adj_matrix_sparse)
        elif adj_init == 'Zeros':
            # make an adj matrix full of zeros
            adj_matrix = np.zeros((self.num_genes, self.num_genes))
            adj_matrix_sparse = sparse.csr_matrix(adj_matrix)
            self.ei_zeros, self.ew_zeros = from_scipy_sparse_matrix(adj_matrix_sparse)

        # make 
        self.adj_init = adj_init
        print("Num Genes: ", self.num_genes)

    def len(self):
        return len(self.features)
    
    def get(self, idx):
        y_out = torch.zeros(self.num_outs)
        y_out[self.labels[idx]] = 1

        data = Data(x = torch.tensor(self.features[idx], dtype = torch.float).unsqueeze(1), y = y_out)
        data.x = data.x.unsqueeze(0)

        data.num_outs = self.num_outs
        data.weights = self.weights
        data.pos_weight = self.pos_weight

        if self.adj_init == 'Pearson':
            # pearson init
            data.edge_index = self.ei_pea
        elif self.adj_init == 'Spearman':
            # spearman init
            data.edge_index = self.ei_sp
        elif self.adj_init == 'None':
            pass
        elif self.adj_init == 'Ones':
            data.edge_index = self.ei_ones
        elif self.adj_init == 'Zeros':
            data.edge_index = self.ei_zeros

        return data

class DGM_Model_Lightning(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        
        self.model_params=hparams
        self.final_adj = None
        fold_num = hparams['fold_num']
        conv_layers = hparams['conv_layers']
        fc_layers = hparams['fc_layers']
        dgm_layers = hparams['dgm_layers']
        fold_num = hparams['fold_num']
        self.edge_perc = hparams['edge_perc']
        self.label_arg = hparams['label_arg']
        self.pos_weight = hparams['pos_weight']
        self.adj_init = hparams['adj_init']
        self.num_genes = hparams['num_genes'] #
        self.cancer_type = hparams['cancer_type']
        self.out_file = hparams['out_file']
        self.gfun = hparams['gfun']#
        self.ffun = hparams['ffun']#
        self.best_edges_file_name = hparams['edges_files_name']#
        self.adj_mat_file_name = hparams['adj_mat_files_name']#
            
        self.graph_f = ModuleList() 
        self.node_g = ModuleList() 
        for i,(dgm_l,conv_l) in enumerate(zip(dgm_layers,conv_layers)):
            if len(dgm_l)>0:
                if 'ffun' not in list(hparams.keys()) or hparams['ffun'] == 'gcn':
                    self.graph_f.append(DGM_d(GCNConv(dgm_l[0], dgm_l[-1]), k=hparams['k'], distance=hparams['distance'], perc = self.edge_perc))
                if hparams['ffun'] == 'sage':
                    self.graph_f.append(DGM_d(SAGEConv(dgm_l[0], dgm_l[-1]), k=hparams['k'], distance=hparams['distance'], perc = self.edge_perc))
                if hparams['ffun'] == 'gat':
                    self.graph_f.append(DGM_d(GATConv(dgm_l[0], dgm_l[-1]), k=hparams['k'], distance=hparams['distance'], perc = self.edge_perc))
                if hparams['ffun'] == 'mlp':
                    self.graph_f.append(DGM_d(MLP(dgm_l), k=hparams['k'], distance=hparams['distance'], perc = self.edge_perc))
                if hparams['ffun'] == 'knn':
                    self.graph_f.append(DGM_d(Identity(retparam=0), k=hparams['k'], distance=hparams['distance'], perc = self.edge_perc))
            else:
                self.graph_f.append(Identity())
            
            if hparams['gfun'] == 'edgeconv':
                conv_l=conv_l.copy()
                conv_l[0]=conv_l[0]*2
                self.node_g.append(EdgeConv(MLP(conv_l), hparams.pooling))
            if hparams['gfun'] == 'gcn':
                self.node_g.append(GCNConv(conv_l[0],conv_l[1]))
            if hparams['gfun'] == 'gat':
                self.node_g.append(GATConv(conv_l[0],conv_l[1]))
            if hparams['gfun'] == 'sage':
                self.node_g.append(SAGEConv(conv_l[0],conv_l[1]))
        
        self.fc = MLP(fc_layers, final_activation=False)
        if hparams['pre_fc'] is not None and len(hparams['pre_fc'])>0:
            self.pre_fc = MLP(hparams['pre_fc'], final_activation=True)
        self.avg_accuracy = None
        
        #torch lightning specific
        self.automatic_optimization = True
        self.debug=False

        self.final_linear = nn.Linear(hparams['num_genes'], 1)

        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.preds_list = []
        self.true_list = []
        self.best_val_f1 = 0
        self.best_edges = None
        self.fold_num = fold_num
        self.edges = hparams['initial_adj'].to(torch.device('cuda:0'))
        
    def forward(self, x, edges, mode = 'train'):
        if self.model_params['pre_fc'] is not None and len(self.model_params['pre_fc'])>0:
            x = self.pre_fc(x)
            
        graph_x = x.detach()
        for f,g in zip(self.graph_f, self.node_g):
            if mode == 'train':
                graph_x, self.edges = f(graph_x, self.edges)
                
                b,n,d = x.shape
                
                x = torch.nn.functional.relu(g(torch.dropout(x.view(-1,d), self.model_params['dropout'], train=self.training), self.edges)).view(b,n,-1) 

                # if self.gfun == 'gat':
                #     x = x.view(b, n, 4, -1).mean(dim=2)
            else:
                graph_x, _ = f(graph_x, edges)

                b,n,d = x.shape

                x = torch.nn.functional.relu(g(x.view(-1,d), edges)).view(b,n,-1) 

                # if self.gfun == 'gat':
                #     x = x.view(b, n, 4, -1).mean(dim=2)

        out = self.fc(x)

        # flatten output
        out = torch.flatten(out)

        # linear out
        out = self.final_linear(out)                
        
        return out
   
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.model_params['lr'])
        # cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.model_params['num_epochs'], eta_min=0)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        # set model to train
        self.train()
        optimizer = self.optimizers(use_pl_optimizer=True)
        optimizer.zero_grad()
        
        data = train_batch
        X = data.x
        y = data.y
        y = y.unsqueeze(0)
        
        pred = self(X, self.edges, mode='train')

        w_p = torch.FloatTensor([self.pos_weight]).cuda()

        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred,y, pos_weight=w_p)
        loss.backward()

        # calculate f1 score
        # threshold the predictions
        if pred > 0.5:
            pred_thresh = 1
        else:
            pred_thresh = 0
        train_f1 = f1_score(y.detach().cpu(), [pred_thresh], average='weighted')
        acc = self.accuracy(pred, y)
            
        optimizer.step()

        self.log('train_loss', loss.detach().cpu())
        self.log('train_perf', train_f1)
        self.log('train_acc', acc)
    
    def validation_step(self, train_batch, batch_idx):
        # set model to eval
        self.eval()
        data = train_batch
        X = data.x
        y = data.y
        y = y.unsqueeze(0)
        
        pred = self(X, self.edges, mode='val')
        w_p = torch.FloatTensor([self.pos_weight]).cuda()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred,y, pos_weight=w_p)
        if pred > 0.5:
            pred_thresh = 1
        else:
            pred_thresh = 0

        self.preds_list.append(pred_thresh)
        self.true_list.append(int(y.detach().cpu().numpy()[0]))
        
        self.log('val_loss', loss.detach())

    def on_validation_epoch_end(self):
        # make confusion matrix 
        cr = classification_report(self.true_list, self.preds_list, output_dict=True)
        f1_val = cr['weighted avg']['f1-score']
        # acc = cr['accuracy']
        acc = balanced_accuracy_score(self.true_list, self.preds_list)

        self.out_file.write("Fold Number: " + str(self.fold_num) + " Epoch: " + str(self.current_epoch) + " Val F1: " + str(f1_val) + " Val Acc: " + str(acc) + "\n")
        print("Fold Number: " + str(self.fold_num) + " Epoch: " + str(self.current_epoch) + " Val F1: " + str(f1_val) + " Val Acc: " + str(acc) + "\n")

        if f1_val > self.best_val_f1:
            self.best_val_f1 = f1_val
            self.log('best_val_f1', self.best_val_f1)
            self.best_edges = self.edges.detach().clone()

            torch.save(self.best_edges, "DGM/adj_matrices/" + str(self.cancer_type) + '/best_edges_' + str(self.label_arg) + "_" + str(self.adj_init) + "_" + str(self.fold_num) + ".pt")

            self.final_adj = torch.sparse_coo_tensor(self.edges.detach().cpu(), torch.ones(self.edges.shape[1]), (self.num_genes, self.num_genes)).to_dense()
            path = "DGM/adj_matrices/" + str(self.cancer_type) + '/best_adj_' + str(self.label_arg) + "_" + str(self.adj_init) + "_" + str(self.fold_num) + ".pt"
            torch.save(self.final_adj, path)

        # Here I wan't to save all adj. matrices in separate files
        # torch.save(self.edges, "adj_matrices/" + str(self.cancer_type) + '/edges_' + str(self.label_arg) + "_" + str(self.adj_init) + "_" + str(self.fold_num) + str(self.current_epoch) + ".pt") ### 
        self.final_adj = torch.sparse_coo_tensor(self.edges.detach().cpu(), torch.ones(self.edges.shape[1]), (self.num_genes, self.num_genes)).to_dense()
        path = "DGM/adj_matrices/" + str(self.cancer_type) + '/adj_' + str(self.label_arg) + "_" + str(self.adj_init) + "_" + str(self.fold_num) + str(self.current_epoch) + ".pt"
        torch.save(self.final_adj, path)

        self.true_list = []
        self.preds_list = []

    def test_step(self, train_batch, batch_idx):
        # set model to eval
        self.eval()
        data = train_batch
        X = data.x
        y = data.y
        y = y.unsqueeze(0)

        self.best_edges = torch.load("DGM/adj_matrices/" + str(self.cancer_type) + '/best_edges_' + str(self.label_arg) + "_" + str(self.adj_init) + "_" + str(self.fold_num) + ".pt", weights_only=True).to(torch.device('cuda:0'))
     
        pred = self(X, self.best_edges, mode='test')
        w_p = torch.FloatTensor([self.pos_weight]).cuda()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred,y, pos_weight=w_p)
        if pred > 0.5:
            pred_thresh = 1
        else:
            pred_thresh = 0

        self.preds_list.append(pred_thresh)
        self.true_list.append(int(y.detach().cpu().numpy()[0]))

        self.log('test_loss', loss.detach().cpu())

    def on_test_epoch_end(self):
        # make confusion matrix 
        cr = classification_report(self.true_list, self.preds_list, output_dict=True)
        f1_test = cr['weighted avg']['f1-score']
        acc = balanced_accuracy_score(self.true_list, self.preds_list)

        # make confusion matrix 
        print("Test: ", cr)
        self.best_edges = torch.load("DGM/adj_matrices/" + str(self.cancer_type) + '/best_edges_' + str(self.label_arg) + "_" + str(self.adj_init) + "_" + str(self.fold_num) + ".pt", weights_only=True).to(torch.device('cuda:0'))
        print("Test: ", self.best_edges)

        self.out_file.write("Fold: " + str(self.fold_num) + " Test F1: " + str(f1_test) + " Test Acc: " + str(acc) + "\n")
        print("Fold: " + str(self.fold_num) + " Test F1: " + str(f1_test) + " Test Acc: " + str(acc) + "\n")
        
        self.log('test_f1', f1_test)
        self.log('test_acc', acc)

        self.best_val_f1 = 0
        self.best_edges = None
        self.true_list = []
        self.preds_list = []


def trainGCN(num_nodes, num_epochs, bs, lr, save_loc, wandb_logger, train_loader, val_loader, test_loader, dropout_val, ffun_alg, gfun_alg, fold_num, initial_adj, num_k, edge_perc, label_arg, pos_weight, adj_init, num_genes, cancer_type, out_file):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        default_root_dir=save_loc,
        accelerator="auto",
        devices=1,
        accumulate_grad_batches=bs,
        max_epochs=num_epochs,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(dirpath=save_loc,
                monitor='val_f1',
                save_top_k=1,
                mode='max'),
            L.pytorch.callbacks.LearningRateMonitor("epoch"),
            L.pytorch.callbacks.EarlyStopping(monitor="val_f1", patience=10),
        ],
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Training
    # Check whether pretrained model exists. If yes, load it and skip training
    hparams = {}    
    hparams['conv_layers'] = [[num_nodes,num_nodes],[num_nodes,num_nodes],[num_nodes,num_nodes]]
    hparams['fc_layers'] = [num_nodes,num_nodes,1]
    hparams['dgm_layers'] = [[num_nodes,num_nodes,num_nodes],[],[]]
    hparams['ffun'] = ffun_alg
    hparams['gfun'] = gfun_alg
    hparams['pre_fc'] = [1, num_nodes]
    hparams['dropout'] = dropout_val
    hparams['lr'] = lr
    hparams['test_eval'] = 10
    hparams['k'] = num_k
    hparams['pooling'] = 'add'
    hparams['distance'] = 'euclidean'
    hparams['num_epochs'] = num_epochs
    hparams['fold_num'] = fold_num
    hparams['initial_adj'] = initial_adj
    hparams['edge_perc'] = edge_perc
    hparams['label_arg'] = label_arg
    hparams['pos_weight'] = pos_weight
    hparams['adj_init'] = adj_init
    hparams['num_genes'] = num_genes
    hparams['cancer_type'] = cancer_type
    hparams['out_file'] = out_file
    hparams['edges_files_name'] = "adj_matrices/" + str(cancer_type) + '/best_edges_' + str(label_arg) + "_" + str(adj_init) + "_" + str(fold_num) + '_' + str(lr) + '_' + str(ffun_alg) + '_' + str(gfun_alg) + '_' + str(bs) + '_' + str(edge_perc) + '_' + str(num_nodes) + '.pt'
    hparams['adj_mat_files_name'] = "adj_matrices/" + str(cancer_type) + '/final_adj_mat_' + str(label_arg) + "_" + str(adj_init) + "_" + str(fold_num) + '_' + str(lr) + '_' + str(ffun_alg) + '_' + str(gfun_alg) + '_' + str(bs) + '_' + str(edge_perc) + '_' + str(num_nodes) + '.pt'
    model = DGM_Model_Lightning(hparams)

    # # fit trainer
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader)

    # Test best model on test set
    # get the ckpt path of the best model
    test_result = trainer.test(model, dataloaders = test_loader, verbose = False, ckpt_path = "best")
    result = {"test": test_result}

    return model, result
    