import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.GAT.layers import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, output, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, output, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, state, adj):
        x = F.dropout(state, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        gat_state = F.elu(self.out_att(x, adj))
        return gat_state


