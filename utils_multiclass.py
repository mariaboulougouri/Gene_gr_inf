# libraries
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Dataset, Data
import os
import lightning as L 
import torchmetrics
from scipy import sparse
from torch_geometric.nn import GATConv, EdgeConv, GCNConv, GATConv
from torch_geometric.nn.conv import SAGEConv
from scipy import sparse
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch.nn import ModuleList
from sklearn.metrics import f1_score, classification_report, balanced_accuracy_score

from layers import *
if (not os.environ.get("USE_KEOPS")) or os.environ.get("USE_KEOPS")=="False":
    from layers_dense import *

class PatientDataset(Dataset):
    def __init__(self, adj_init, label_arg, cancer_type):
        super().__init__()

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

        if label_arg == 'pathological_stage':
            metadata = 'patient.stage_event.pathologic_stage' # one example
        elif label_arg == 'pathologic_t':
            if cancer_type == 'hnsc':
                metadata = 'patient.stage_event.tnm_categories.clinical_categories.clinical_t'
            else:
                metadata = 'patient.stage_event.tnm_categories.pathologic_categories.pathologic_t' # one example
        elif label_arg == 'pathologic_n':
            if cancer_type == 'hnsc':
                metadata = 'patient.stage_event.tnm_categories.clinical_categories.clinical_n'
            else:
                metadata = 'patient.stage_event.tnm_categories.pathologic_categories.pathologic_n' # one example
        elif label_arg == 'pathologic_m':
            if cancer_type == 'hnsc':
                metadata = 'patient.stage_event.tnm_categories.clinical_categories.clinical_m'
            else:
                metadata = 'patient.stage_event.tnm_categories.pathologic_categories.pathologic_m'

        data_path = '/rcp/boulougo/DGM/Raw_data/' + folder + '/'
        
        df = pd.read_csv(data_path + "filtered_rnaseq_zerosandlowvar_removed.txt", sep='\t', index_col=0, header=0)
        df_labels = pd.read_csv(data_path + "filtered_metaclinical.txt", sep= '\t', index_col=0)

        df_labels = df_labels[metadata]
        df_labels = df_labels.dropna()
        df.index = df.index.str[:12].str.lower()
        df = df.loc[df_labels.index]

        if label_arg == 'pathological_stage':
            df_labels.replace('stage ia', 'stage i', inplace=True)
            df_labels.replace('stage ib', 'stage i', inplace=True)
            df_labels.replace('stage ic', 'stage ii', inplace=True)
            df_labels.replace('stage iia', 'stage ii', inplace=True)
            df_labels.replace('stage iib', 'stage ii', inplace=True)
            df_labels.replace('stage iic', 'stage ii', inplace=True)
            df_labels.replace('stage iiia', 'stage iii', inplace=True)
            df_labels.replace('stage iiib', 'stage iii', inplace=True)
            df_labels.replace('stage iiic', 'stage iii', inplace=True)
            df_labels.replace('stage iva', 'stage iv', inplace=True)
            df_labels.replace('stage ivb', 'stage iv', inplace=True)
            df_labels.replace('stage ivc', 'stage iv', inplace=True)
            df_labels.drop(df_labels[df_labels == 'stage x'].index, inplace = True)
        elif label_arg == 'pathologic_t':
            df_labels.replace('t1a', 't1', inplace=True)
            df_labels.replace('t1b', 't1', inplace=True)
            df_labels.replace('t1c', 't1', inplace=True)
            df_labels.replace('t1d', 't1', inplace=True)
            df_labels.replace('t2a', 't2', inplace=True)
            df_labels.replace('t2b', 't2', inplace=True)
            df_labels.replace('t2c', 't2', inplace=True)
            df_labels.replace('t2d', 't2', inplace=True)
            df_labels.replace('t3a', 't3', inplace=True)
            df_labels.replace('t3b', 't3', inplace=True)
            df_labels.replace('t3c', 't3', inplace=True)
            df_labels.replace('t3d', 't3', inplace=True)
            df_labels.replace('t4a', 't4', inplace=True)
            df_labels.replace('t4b', 't4', inplace=True)
            df_labels.replace('t4c', 't4', inplace=True)
            df_labels.replace('t4d', 't4', inplace=True)
            df_labels.drop(df_labels[df_labels == 'tx'].index, inplace = True)
            df_labels.drop(df_labels[df_labels == 'tis'].index, inplace = True)
        elif label_arg == 'pathologic_n':
            df_labels.replace('n0 (mol+)', 'n0', inplace=True)
            df_labels.replace('n0 (i+)', 'n0', inplace=True)
            df_labels.replace('n0 (i-)', 'n0', inplace=True)
            df_labels.replace('n1a', 'n1', inplace=True)
            df_labels.replace('n1b', 'n1', inplace=True)
            df_labels.replace('n1c', 'n1', inplace=True)
            df_labels.replace('n1d', 'n1', inplace=True)
            df_labels.replace('n1mi', 'n1', inplace=True)
            df_labels.replace('n2a', 'n2', inplace=True)
            df_labels.replace('n2b', 'n2', inplace=True)
            df_labels.replace('n2c', 'n2', inplace=True)
            df_labels.replace('n2d', 'n2', inplace=True)
            df_labels.replace('n3a', 'n3', inplace=True) 
            df_labels.replace('n3b', 'n3', inplace=True)
            df_labels.replace('n3c', 'n3', inplace=True)
            df_labels.replace('n3d', 'n3', inplace=True)
            df_labels.drop(df_labels[df_labels == 'nx'].index, inplace = True)
        elif label_arg == 'pathologic_m':
            df_labels.replace('m1a', 'm1', inplace=True)
            df_labels.replace('m1b', 'm1', inplace=True)
            df_labels.replace('m1c', 'm1', inplace=True)
            df_labels.replace('m1d', 'm1', inplace=True)
            df_labels.drop(df_labels[df_labels == 'mx'].index, inplace = True)
            df_labels.drop(df_labels[df_labels  == 'cm0 (i+)'].index, inplace = True)

        df_labels.dropna()
        df = df.loc[df_labels.index]

        # convert df_labels into a tensor with 0,1
        df_labels = df_labels.astype('category')
        df_labels = df_labels.cat.codes

        self.labels = np.asarray(df_labels.values)

        self.num_outs = len(list(set(list(self.labels))))

        print("Num Unique Labels: ", self.num_outs)

        # # number of 0, and 1 from self.labels to make self.weights
        self.weights = []
        for k in range(self.num_outs):
            num_k = np.sum(self.labels == k) / len(self.labels)
            self.weights.append(1/num_k)

        # convert df into a feature matrix 
        self.features = np.asarray(df.values)
        self.num_genes = len(self.features[0])

        if adj_init == 'Pearson':
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
        self.num_classes = hparams['num_classes']
        self.adj_init = hparams['adj_init']
        self.class_weights = hparams['class_weights']
        self.num_genes = hparams['num_genes']
        self.cancer_type = hparams['cancer_type']
        self.out_file = hparams['out_file']
        self.ffun = hparams['ffun']
        self.gfun = hparams['gfun']
        self.best_edges_files_name = hparams['best_edges_files_name']
        self.final_adj_mat_file_name = hparams['final_adj_files_name']
            
        self.graph_f = ModuleList() 
        self.node_g = ModuleList() 
        for i,(dgm_l,conv_l) in enumerate(zip(dgm_layers,conv_layers)):
            if len(dgm_l)>0:
                if 'ffun' not in list(hparams.keys()) or hparams['ffun'] == 'gcn':
                    self.graph_f.append(DGM_d(GCNConv(dgm_l[0],dgm_l[-1]),k=hparams['k'],distance=hparams['distance'], perc = self.edge_perc))
                if hparams['ffun'] == 'sage':
                    self.graph_f.append(DGM_d(SAGEConv(dgm_l[0], dgm_l[-1]), k=hparams['k'], distance=hparams['distance'], perc = self.edge_perc))
                if hparams['ffun'] == 'gat':
                    self.graph_f.append(DGM_d(GATConv(dgm_l[0],dgm_l[-1]),k=hparams['k'],distance=hparams['distance'], perc = self.edge_perc))
                if hparams['ffun'] == 'mlp':
                    self.graph_f.append(DGM_d(MLP(dgm_l),k=hparams['k'],distance=hparams['distance'], perc = self.edge_perc))
                if hparams['ffun'] == 'knn':
                    self.graph_f.append(DGM_d(Identity(retparam=0),k=hparams['k'],distance=hparams['distance'], perc = self.edge_perc))
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
        self.debug = False

        self.final_linear = nn.Linear(self.num_genes, self.num_classes)

        self.accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes = self.num_classes, average = 'weighted')
        self.weighted_f1 = torchmetrics.classification.MulticlassF1Score(num_classes = self.num_classes, average = 'weighted')
        self.preds_list = []
        self.true_list = []
        self.best_val_f1 = 0
        self.best_edges = None
        self.fold_num = fold_num
        self.edges = hparams['initial_adj'].to(torch.device('cuda:0'))
        self.loss = torch.nn.CrossEntropyLoss(weight = torch.tensor(self.class_weights).to(torch.device('cuda:0')).to(torch.float32))
        
    def forward(self, x, edges=None, mode='train'):
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
        out = torch.flatten(out)
        out = self.final_linear(out)                
        return out
   
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.model_params['lr'])
        # cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.model_params['num_epochs'], eta_min=0)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        self.train()
        
        optimizer = self.optimizers(use_pl_optimizer=True)
        optimizer.zero_grad()
        
        data = train_batch
        X = data.x
        y = data.y
        y = y.unsqueeze(0)
        y = y.long()
        y = torch.argmax(y, dim = 1)

        # print("Train Step Edges: ", self.edges)
        
        pred = self(X, self.edges, mode='train')
        pred = pred.unsqueeze(0)

        loss = self.loss(input = pred, target = y)
        loss.backward()

        # calculate f1 score
        # threshold the predictions
        pred = torch.argmax(pred, dim = 1)
        train_f1 = self.weighted_f1(pred, y)
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
        y = y.long()
        y = torch.argmax(y, dim = 1)

        # print("Val Step Edges: ", self.edges)
        
        pred = self(X, self.edges, mode='val')
        pred = pred.unsqueeze(0)

        loss = self.loss(input = pred, target = y)

        pred = torch.argmax(pred, dim = 1)
        self.preds_list.append(int(pred.detach().cpu().numpy()))
        self.true_list.append(int(y.detach().cpu().numpy()))
        
        self.log('val_loss', loss.detach())

    def on_validation_epoch_end(self):
        # get metrics
        cr = classification_report(self.true_list, self.preds_list, output_dict=True)
        print("Validation: ", cr)
        print("Validation: ", self.edges)
        f1_val = cr['weighted avg']['f1-score']
        acc_val = balanced_accuracy_score(self.true_list, self.preds_list)
        
        # self.log('val cr', cr)
        self.log('val_f1', f1_val)
        self.log('val_acc', acc_val)

        self.out_file.write("Fold Number: " + str(self.fold_num) + " Epoch: " + str(self.current_epoch) + " Val F1: " + str(f1_val) + " Val Acc: " + str(acc_val) + "\n")
        print("Fold Number: " + str(self.fold_num) + " Epoch: " + str(self.current_epoch) + " Val F1: " + str(f1_val) + " Val Acc: " + str(acc_val) + "\n")


        if f1_val > self.best_val_f1:
            self.best_val_f1 = f1_val
            self.log('best_val_f1', self.best_val_f1)
            self.best_edges = self.edges.detach().clone()

            torch.save(self.best_edges, "DGM/adj_matrices/" + str(self.cancer_type) + '/best_edges_' + str(self.label_arg) + "_" + str(self.adj_init) + "_" + str(self.fold_num) + ".pt")

            self.final_adj = torch.sparse_coo_tensor(self.edges.detach().cpu(), torch.ones(self.edges.shape[1]), (self.num_genes, self.num_genes)).to_dense()
            path = "DGM/adj_matrices/" + str(self.cancer_type) + '/best_adj_' + str(self.label_arg) + "_" + str(self.adj_init) + "_" + str(self.fold_num) + ".pt"
            torch.save(self.final_adj, self.final_adj_mat_file_name)

        # Here I wan't to save all adj. matrices in separate files
        # torch.save(self.edges, "adj_matrices/" + str(self.cancer_type) + '/edges_' + str(self.label_arg) + "_" + str(self.adj_init) + "_" + str(self.fold_num) + str(self.current_epoch) + ".pt") ### 
        self.final_adj = torch.sparse_coo_tensor(self.edges.detach().cpu(), torch.ones(self.edges.shape[1]), (self.num_genes, self.num_genes)).to_dense()
        path = "DGM/adj_matrices/" + str(self.cancer_type) + '/adj_' + str(self.label_arg) + "_" + str(self.adj_init) + "_" + str(self.fold_num) + "_" + str(self.current_epoch) + ".pt"
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
        y = y.long()
        y = torch.argmax(y, dim = 1)

        self.best_edges = torch.load("DGM/adj_matrices/" + str(self.cancer_type) + '/best_edges_' + str(self.label_arg) + "_" + str(self.adj_init) + "_" + str(self.fold_num) + ".pt", weights_only=True).to(torch.device('cuda:0'))

        pred = self(X, self.best_edges, mode='test')
        pred = pred.unsqueeze(0)

        # categorical cross entropy
        loss = self.loss(input = pred, target = y)

        pred = torch.argmax(pred, dim = 1)
        self.preds_list.append(int(pred.detach().cpu().numpy()))
        self.true_list.append(int(y.detach().cpu().numpy()))

        self.log('test_loss', loss.detach().cpu())

    def on_test_epoch_end(self):
        cr = classification_report(self.true_list, self.preds_list, output_dict=True)
        f1_test = cr['weighted avg']['f1-score']
        acc_test = balanced_accuracy_score(self.true_list, self.preds_list)

        # make confusion matrix 
        print("Test: ", cr)
        self.best_edges = torch.load("DGM/adj_matrices/" + str(self.cancer_type) + '/best_edges_' + str(self.label_arg) + "_" + str(self.adj_init) + "_" + str(self.fold_num) + ".pt", weights_only=True).to(torch.device('cuda:0'))
        print("Test: ", self.best_edges)

        self.out_file.write("Fold: " + str(self.fold_num) + " Test F1: " + str(f1_test) + " Test Acc: " + str(acc_test) + "\n")
        print("Fold: " + str(self.fold_num) + " Test F1: " + str(f1_test) + " Test Acc: " + str(acc_test) + "\n")
        
        # self.log('test cr', cr)
        self.log('test_f1', f1_test)
        self.log('test_acc', acc_test)

        self.best_val_f1 = 0
        self.best_edges = None
        self.true_list = []
        self.preds_list = []

def trainGCN(num_nodes, num_epochs, bs, lr, save_loc, wandb_logger, train_loader, val_loader, test_loader, dropout_val, ffun_alg, gfun_alg, fold_num, initial_adj, num_k, edge_perc, label_arg, num_classes, adj_init, class_weights, num_genes, cancer_type, out_file):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        default_root_dir=save_loc,
        accelerator="auto",
        devices=1,
        accumulate_grad_batches=bs,
        max_epochs=num_epochs,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(dirpath=save_loc,
                monitor='val_f1',
                save_top_k=1,
                mode='max'),
            L.pytorch.callbacks.LearningRateMonitor("epoch"),
            L.pytorch.callbacks.EarlyStopping(monitor="val_f1", patience=20),
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
    hparams['num_classes'] = num_classes
    hparams['adj_init'] = adj_init
    hparams['class_weights'] = class_weights
    hparams['num_genes'] = num_genes
    hparams['cancer_type'] = cancer_type
    hparams['out_file'] = out_file
    hparams['best_edges_files_name'] = "DGM/adj_matrices/" + str(cancer_type) + '/best_edges_' + str(label_arg) + "_" + str(adj_init) + "_" + str(fold_num) + "_" + str(lr) + "_" + str(ffun_alg) + "_" + str(gfun_alg) + "_" + str(bs) + "_" + str(edge_perc) + "_" + str(num_nodes) + ".pt"
    hparams['final_adj_files_name'] = "DGM/adj_matrices/" + str(cancer_type) + '/final_adj_mat_' + str(label_arg) + "_" + str(adj_init) + "_" + str(fold_num) + "_" + str(lr) + "_" + str(ffun_alg) + "_" + str(gfun_alg) + "_" + str(bs) + "_" + str(edge_perc) + "_" + str(num_nodes) + ".pt"
    model = DGM_Model_Lightning(hparams)

    # # fit trainer
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader)
    # Test best model on test set
    test_result = trainer.test(model, dataloaders = test_loader, verbose = False, ckpt_path = "best")
    result = {"test": test_result}
    # print(result)

    return model, result
    