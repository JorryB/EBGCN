import sys,os
sys.path.append(os.getcwd())
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import copy
from torch.nn import BatchNorm1d
from collections import OrderedDict

# Top-Down Graph Convolutional Networks
class TDrumorGCN(th.nn.Module):
    def __init__(self,args):
        super(TDrumorGCN, self).__init__()
        self.args = args
        
        # first layer: input features are the word embedding of the tweet contents, output is the number of neurons in the layer
        self.conv1 = GCNConv(args.input_features, args.hidden_features)
        # second layer: input dimension is the word embedding features + the output dimension from the first layer, output the class probabilities
        self.conv2 = GCNConv(args.input_features + args.hidden_features, args.output_features)
        self.device = args.device
        # store the number of hidden feature number as a list: 64 ---> [64]
        self.num_features_list = [args.hidden_features * r for r in [1]]

        def creat_network(self, name):
            # in OrderedDict() the order is important and will be kept if any item inserted into this dict
            layer_list = OrderedDict()
            # for each layer:
            for l in range(len(self.num_features_list)):
                # store the convolutional layer in the corresponding key
                # the layer is a 1D convolutional operator with padding and stride = 1, kernel/sliding window size = 1 X 1
                # input channels is the hidden features and output the same dimension
                # no bias in this case, so the weight in the sliding windows is the only trainable weight in this layer
                layer_list[name + 'conv{}'.format(l)] = th.nn.Conv1d(
                    in_channels=args.hidden_features,
                    out_channels=args.hidden_features,
                    kernel_size=1,
                    bias=False)
                # do normalization on the output of the previous layer
                layer_list[name + 'norm{}'.format(l)] = th.nn.BatchNorm1d(num_features=args.hidden_features)
                # output LeakyReLu of the output from the previous layer
                layer_list[name + 'relu{}'.format(l)] = th.nn.LeakyReLU()
            # 1 X 1 sliding windows to scan all the hidden features into a probability (true or false)
            layer_list[name + 'conv_out'] = th.nn.Conv1d(in_channels=args.hidden_features,
                                                         out_channels=1,
                                                         kernel_size=1)
            return layer_list
            
        ### create Sequential network with the network structure order described in the function "create_network"
        self.sim_network = th.nn.Sequential(creat_network(self, 'sim_val'))
        mod_self = self
        mod_self.num_features_list = [args.hidden_features]
        self.W_mean = th.nn.Sequential(creat_network(mod_self, 'W_mean'))
        self.W_bias = th.nn.Sequential(creat_network(mod_self, 'W_bias'))
        self.B_mean = th.nn.Sequential(creat_network(mod_self, 'B_mean'))
        self.B_bias = th.nn.Sequential(creat_network(mod_self, 'B_bias'))
        ###

        ### define fully connected neural network function with input dimension = hidden feature and output the dimension of the number of edge
        self.fc1 = th.nn.Linear(args.hidden_features, args.edge_num, bias=False)
        self.fc2 = th.nn.Linear(args.hidden_features, args.edge_num, bias=False)
        ###
        # define random dropout function with proportion = args.dropout
        self.dropout = th.nn.Dropout(args.dropout)
        # loss = y_true * (log y_true - log y_pred) / input_size
        self.eval_loss = th.nn.KLDivLoss(reduction='batchmean')
        # normalization on the previous layer with dimension = hidden_features + word_embedding size
        self.bn1 = BatchNorm1d(args.hidden_features + args.input_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # store a copy of the original input x where the dimension = number of nodes X number of word_embedding features
        x1 = copy.copy(x.float())
        # input the adjacency matrix and X's word embedding features, return the output with dimension = (Number of nodes X hidden_dimension)          
        x = self.conv1(x, edge_index)
        # save a copy of the output from the first conv1 layer
        x2 = copy.copy(x)

        if self.args.edge_infer_td:
            edge_loss, edge_pred = self.edge_infer(x, edge_index)
        else:
            edge_loss, edge_pred = None, None

        rootindex = data.rootindex
        ## len(data.batch): the number of data points in the batch which is represented by N, size(1): number of features
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(self.device)
        ## extract the max number of graphs from the dataset batches
        batch_size = max(data.batch) + 1

        ## from 0 to (the max number of graphs)
        for num_batch in range(batch_size):
            ## look for the batch index where the num_batch is equal to the number of graphs in that batch
            index = (th.eq(data.batch, num_batch))
            ## 
            root_extend[index] = x1[rootindex[num_batch]]
            
        # combine the two matrix by column
        # x: (N X input_features), root_extend: (N X hidden_features)
        # x will be converted into (N X (input_features + hidden_features))
        x = th.cat((x, root_extend), 1)
        x = self.bn1(x)
        x = F.relu(x)

        # edge_pred: (N X 1)
        # A: N X N
        # the 1st row of A: 1 X N
        # edge_weight.T: 1 X N, updated A1: A1 point_wise_multiplication with edge_weight
        # go through all N rows
        x = self.conv2(x, edge_index, edge_weight=edge_pred)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(self.device)
        # second update
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        # updated x's dim = (N X (input_features + hidden_features + hidden_features))
        x = th.cat((x, root_extend), 1)
        # ______question________
        x = scatter_mean(x, data.batch, dim=0)
        return x, edge_loss

    def edge_infer(self, x, edge_index):
        ## edge_index[0]: [node1, node3] ---> edge_index[1]: [node2, node4]
        ## X is of dimension (N X F)
        row, col = edge_index[0], edge_index[1]
        # convert (N X F) into a column （N X F X 1)
        x_i = x[row - 1].unsqueeze(2)
        # convert (N X F) into a column （N X 1 X F)
        x_j = x[col - 1].unsqueeze(1)
        # take absolute difference between x_i and x_j, dimension = (N X F X F)      ____________question_____________
        x_ij = th.abs(x_i - x_j)
        # go through the network architecture (Conv1D, normalization, LeakyReLu, Conv1D) and output the result with (N X 1 X F)
        sim_val = self.sim_network(x_ij)
        # dimension will be converted from (N X 1 X F) into (N X 1 X edge_num)
        edge_pred = self.fc1(sim_val)
        # convert the values to probabilities (N X 1 X edge_num)
        edge_pred = th.sigmoid(edge_pred)
        ## The likelihood of latent relations from the l-th layer based on node embeddings (P)
        
        ## convert the input (N X F X F) into (N X 1 X F)
        w_mean = self.W_mean(x_ij)
        w_bias = self.W_bias(x_ij)
        b_mean = self.B_mean(x_ij)
        b_bias = self.B_bias(x_ij)
        ##
        ## logit_mean and logit_var refer to the Factorized Guassian distribution (u and var), dim = (N X 1 X F)
        logit_mean = w_mean * sim_val + b_mean
        logit_var = th.log((sim_val ** 2) * th.exp(w_bias) + th.exp(b_bias))
        ##

        # the value of edge weight follow a normal distribution with dim = (N X 1 X F)
        edge_y = th.normal(logit_mean, logit_var)
        # convert the values from normal distribution to probabilities (N X 1 X F)
        edge_y = th.sigmoid(edge_y)
        # input the probabilities (N X 1 X F), return the output (N X 1 X edge_num), (Q)
        edge_y = self.fc2(edge_y)
        # take the log of the softmax value of edge_pred, output dim = (N X 1 X edge_num), (P)
        logp_x = F.log_softmax(edge_pred, dim=-1) 
        # take softmax value of edge_y, output dim = (N X 1 X edge_num)
        p_y = F.softmax(edge_y, dim=-1)
        # calculate the loss of two distributions, return an average loss # KL D loss between P and Q
        edge_loss = self.eval_loss(logp_x, p_y) 
        # return a loss value, and the mean edge_pred (N X 1 X 1)
        return edge_loss, th.mean(edge_pred, dim=-1).squeeze(1)

class BUrumorGCN(th.nn.Module):
    def __init__(self,args):
        super(BUrumorGCN, self).__init__()
        self.args = args
        self.conv1 = GCNConv(args.input_features, args.hidden_features)
        self.conv2 = GCNConv(args.input_features + args.hidden_features, args.output_features)
        self.device = args.device
        self.num_features_list = [args.hidden_features * r for r in [1]]
        def creat_network(self, name):
            layer_list = OrderedDict()
            for l in range(len(self.num_features_list)):
                layer_list[name + 'conv{}'.format(l)] = th.nn.Conv1d(
                    in_channels=args.hidden_features,
                    out_channels=args.hidden_features,
                    kernel_size=1,
                    bias=False)
                layer_list[name + 'norm{}'.format(l)] = th.nn.BatchNorm1d(num_features=args.hidden_features)
                layer_list[name + 'relu{}'.format(l)] = th.nn.LeakyReLU()
            layer_list[name + 'conv_out'] = th.nn.Conv1d(in_channels=args.hidden_features,
                                                         out_channels=1,
                                                         kernel_size=1)
            return layer_list
        self.sim_network = th.nn.Sequential(creat_network(self, 'sim_val'))
        mod_self = self
        mod_self.num_features_list = [args.hidden_features]  #
        self.W_mean = th.nn.Sequential(creat_network(mod_self, 'W_mean'))
        self.W_bias = th.nn.Sequential(creat_network(mod_self, 'W_bias'))
        self.B_mean = th.nn.Sequential(creat_network(mod_self, 'B_mean'))
        self.B_bias = th.nn.Sequential(creat_network(mod_self, 'B_bias'))
        self.fc1 = th.nn.Linear(args.hidden_features, args.edge_num, bias=False)
        self.fc2 = th.nn.Linear(args.hidden_features, args.edge_num, bias=False)
        self.dropout = th.nn.Dropout(args.dropout)
        self.eval_loss = th.nn.KLDivLoss(reduction='batchmean')  # mean
        self.bn1 = BatchNorm1d(args.hidden_features + args.input_features)

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)

        if self.args.edge_infer_bu:
            edge_loss, edge_pred = self.edge_infer(x, edge_index)
        else:
            edge_loss, edge_pred = None, None

        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(self.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_pred)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(self.device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x= scatter_mean(x, data.batch, dim=0)
        return x, edge_loss

    def edge_infer(self, x, edge_index):
        row, col = edge_index[0], edge_index[1]
        x_i = x[row - 1].unsqueeze(2)
        x_j = x[col - 1].unsqueeze(1)
        x_ij = th.abs(x_i - x_j)
        sim_val = self.sim_network(x_ij)
        edge_pred = self.fc1(sim_val)
        edge_pred = th.sigmoid(edge_pred)

        w_mean = self.W_mean(x_ij)
        w_bias = self.W_bias(x_ij)
        b_mean = self.B_mean(x_ij)
        b_bias = self.B_bias(x_ij)
        logit_mean = w_mean * sim_val + b_mean
        logit_var = th.log((sim_val ** 2) * th.exp(w_bias) + th.exp(b_bias))

        edge_y = th.normal(logit_mean, logit_var)
        edge_y = th.sigmoid(edge_y)
        edge_y = self.fc2(edge_y)

        logp_x = F.log_softmax(edge_pred, dim=-1)
        p_y = F.softmax(edge_y, dim=-1)
        edge_loss = self.eval_loss(logp_x, p_y)
        return edge_loss, th.mean(edge_pred, dim=-1).squeeze(1)

class EBGCN(th.nn.Module):
    def __init__(self, args):
        super(EBGCN, self).__init__()
        self.args = args
        self.TDrumorGCN = TDrumorGCN(args)
        self.BUrumorGCN = BUrumorGCN(args)
        self.fc = th.nn.Linear((args.hidden_features + args.output_features)*2, args.num_class)

    def forward(self, data):
        TD_x, TD_edge_loss = self.TDrumorGCN(data)
        BU_x, BU_edge_loss = self.BUrumorGCN(data)

        self.x = th.cat((BU_x,TD_x), 1)
        out = self.fc(self.x)
        out = F.log_softmax(out, dim=1)
        return out,  TD_edge_loss, BU_edge_loss
        
