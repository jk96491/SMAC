import math
import torch
import torch.nn as nn
torch.set_default_tensor_type(torch.FloatTensor)

class GINGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, state_dim, hypernet_embed, weights_operation=None):
        super(GINGraphConvolution, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.state_dim = state_dim
        self.weights_operation = weights_operation
        
        self.hidden_features = int((in_features + out_features) / 2)
        
        # breaking the MLP to hypernetworks for deriving the weights and biases
        self.w1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                nn.ReLU(),
                                nn.Linear(hypernet_embed, in_features * self.hidden_features))
        self.b1 = nn.Linear(self.state_dim, self.hidden_features)
        
        self.w2 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                nn.ReLU(),
                                nn.Linear(hypernet_embed, self.hidden_features * out_features))
        self.b2 = nn.Linear(self.state_dim, out_features)
        
    def forward(self, input_features, adj, states):
        
        aggregated_input = torch.matmul(adj, input_features)
        
        batch_size = aggregated_input.size(0)
        
        w1 = self.w1(states).view(-1, self.in_features, self.hidden_features)
        w2 = self.w2(states).view(-1, self.hidden_features, self.out_features)
        
        if self.weights_operation == 'abs':
            w1 = torch.abs(w1)
            w2 = torch.abs(w2)
        elif self.weights_operation == 'clamp':
            w1 = nn.ReLU()(w1)
            w2 = nn.ReLU()(w2)
        elif self.weights_operation is None:
            pass
        else:
            raise NotImplementedError('The operation {} on the weights not implemented'.format(self.weights_operation))
            
        b1 = self.b1(states).view(batch_size, 1, -1).repeat(1, aggregated_input.size(1), 1)
        b2 = self.b2(states).view(batch_size, 1, -1).repeat(1, aggregated_input.size(1), 1)
        
        output1 = torch.nn.LeakyReLU()(torch.matmul(aggregated_input, w1) + b1)
        output = torch.matmul(output1, w2) + b2
        
        return output
