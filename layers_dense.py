import torch
import numpy as np
from layers import *

from torch.nn import Module, ModuleList, Sequential
from torch import nn
from scipy import sparse
from torch_geometric.utils.convert import from_scipy_sparse_matrix


class DGM_d(nn.Module):
    def __init__(self, embed_f, distance=pairwise_euclidean_distances, sparse=True, perc=25):
        super(DGM_d, self).__init__()
        
        self.sparse=sparse
        
        self.temperature = nn.Parameter(torch.tensor(1. if distance=="hyperbolic" else 4.).float())
        self.embed_f = embed_f
        self.centroid=None
        self.scale=None
        self.perc = perc
        
        self.debug=False
        if distance == 'euclidean':
            self.distance = pairwise_euclidean_distances
        else:
            self.distance = pairwise_poincare_distances
        
    def forward(self, x, A):
        x = self.embed_f(x,A)  
        
        if self.training:
            D, _x = self.distance(x)
            edges_hat = self.topPercEdges(D)

            return x, edges_hat
                
        else:
            with torch.no_grad():           
                return x, A
    

    def topPercEdges(self, D):
        D = torch.squeeze(D, dim=0)
        D = D.detach().cpu().numpy()
        
        # least 25% percentile value
        q_val = np.percentile(D, 100-self.perc)
        D = np.where(D >= q_val, 1, 0)

        # convert this into an edge list
        adj_mat = sparse.csr_matrix(D)
        edges, edge_w = from_scipy_sparse_matrix(adj_mat)
        edges = edges.long().to(torch.device('cuda'))

        return edges