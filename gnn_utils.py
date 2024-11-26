# libraries
import numpy as np
import torch
import torch.nn as nn
import lightning as L
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.conv import GCNConv
from torch.autograd import Variable
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score, precision_recall_curve


class GCNOnly(L.LightningModule):                                            
    def __init__(self, gcn_layers, dropout_val, num_epochs, bs, lr, num_inp_ft, algo, pos_weight, num_genes):
        super().__init__()

        self.module_list = nn.ModuleList()
        self.module_list.append(GCNConv(num_inp_ft, gcn_layers[0]))
        for i in range(len(gcn_layers)-1):
            self.module_list.append(GCNConv(gcn_layers[i], gcn_layers[i+1]))
        self.dropout = nn.Dropout(dropout_val)
        self.linear = nn.Linear(num_genes*32, 1)
        self.num_genes = num_genes
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        self.pos_weight = pos_weight
        self.sigmoid = nn.Sigmoid()
        self.lr = lr
        self.bs = bs
        self.num_epochs = num_epochs
        self.algo = algo

        self.true_list = []
        self.preds_list = []

    def forward(self, batch):

        x = batch.x
        ei = batch.edge_index
        
        for i in range(len(self.module_list)):
            x = self.module_list[i](x, ei)
            x = self.relu(x)
            x = self.dropout(x)

        x = x.view(-1, self.num_genes*32)
        out = self.linear(x)
        out = out.squeeze(0)
        # # softmax out
        # out = self.sigmoid(out)

        return out
    
    def _get_loss(self, batch):
        y = batch.y
        y = y.unsqueeze(0)

        y_pred = self.forward(batch)

        w_p = torch.FloatTensor([self.pos_weight]).cuda()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y, pos_weight=w_p)

        return loss, y, y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        
        monitor = 'val_loss'
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}
    
    def training_step(self, batch):

        loss, y, y_pred = self._get_loss(batch)

        self.log('train_loss', loss, batch_size=self.bs)

        return loss
    
    def validation_step(self, batch):
        loss, y, y_pred = self._get_loss(batch)
        print(self.sigmoid(y_pred))

        self.preds_list.append(y_pred.detach().cpu().numpy()[0])
        self.true_list.append(int(y.detach().cpu().numpy()[0]))

        self.log('val_loss', loss)

        return loss
    
    def on_validation_epoch_end(self):
        precision, recall, thresholds = precision_recall_curve(self.true_list, self.preds_list)
        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.argmax(fscore)
        best_thresh = thresholds[ix]
        # if best_thresh < 0:
        #     best_thresh = 0.5
        print(f"Best Threshold: {best_thresh}")
        self.preds_list = [1 if i >= best_thresh else 0 for i in self.preds_list]
        # make confusion matrix 
        cr = classification_report(self.true_list, self.preds_list, output_dict=True)
        f1_val = cr['weighted avg']['f1-score']
        acc = balanced_accuracy_score(self.true_list, self.preds_list)

        print(f"Validation F1: {f1_val}, Validation Accuracy: {acc}")
        self.true_list = []
        self.preds_list = []
    
    def test_step(self, batch):
        loss, y, y_pred = self._get_loss(batch)

        self.preds_list.append(y_pred.detach().cpu().numpy()[0])
        self.true_list.append(int(y.detach().cpu().numpy()[0]))

        self.log('test_loss', loss)

        return loss
    
    def on_test_epoch_end(self):
        precision, recall, thresholds = precision_recall_curve(self.true_list, self.preds_list)
        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.argmax(fscore)
        best_thresh = thresholds[ix]
        print(f"Best Threshold: {best_thresh}")
        self.preds_list = [1 if i >= best_thresh else 0 for i in self.preds_list]
        # make confusion matrix 
        cr = classification_report(self.true_list, self.preds_list, output_dict=True)
        f1_test = cr['weighted avg']['f1-score']
        acc = balanced_accuracy_score(self.true_list, self.preds_list)

        print(f"Test F1: {f1_test}, Test Accuracy: {acc}")

        self.log('test_f1', f1_test)
        self.log('test_acc', acc)
        self.true_list = []
        self.preds_list = []

def trainGCN(gcn_layers, num_epochs, bs, lr, train_loader, val_loader, test_loader, dropout_val, num_inp_ft, algo, seed, save_loc, pos_weight, num_genes):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        default_root_dir=save_loc,
        accelerator="auto",
        devices=1,
        accumulate_grad_batches=bs,
        max_epochs=num_epochs,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(dirpath=save_loc,
                monitor='val_loss',
                save_top_k=2),
            L.pytorch.callbacks.LearningRateMonitor("epoch"),
            L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=10),
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # # Training
    # # Check whether pretrained model exists. If yes, load it and skip training
    model = GCNOnly(gcn_layers=gcn_layers, dropout_val=dropout_val, num_epochs=num_epochs, bs=bs, lr=lr, num_inp_ft=num_inp_ft, algo=algo, pos_weight=pos_weight, num_genes=num_genes)

    # number of trainable parameters in the model
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # # fit trainer
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader)
    # Test best model on test set
    test_result = trainer.test(model, dataloaders = test_loader, verbose = False, ckpt_path = "best")
    result = {"test": test_result}

    # # Testing
    # # pretrained model loading
    # model = GCNOnly.load_from_checkpoint(save_loc + '/best.ckpt', gcn_layers=gcn_layers, dropout_val=dropout_val, num_epochs=num_epochs, bs=bs, lr=lr, num_inp_ft=num_inp_ft, algo=algo, pos_weight=pos_weight, num_genes=num_genes)
    # model.eval()
    # # test on this model
    # result = trainer.test(model, dataloaders=test_loader, verbose=False)
    # print(result)

    return model, result