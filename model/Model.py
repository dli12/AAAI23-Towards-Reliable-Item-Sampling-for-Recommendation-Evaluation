import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sparse
from sklearn.metrics.pairwise import cosine_similarity

class itemKNN(nn.Module):
    
    def __init__(self, URM_train):
        
        super().__init__()
        
        self.URM_train = URM_train
        
        self.num_user, self.num_item = self.URM_train.shape
        
        self.noise = 1e-5 * np.random.randn(self.num_item, self.num_item)
        
        self.similarity = cosine_similarity(URM_train.transpose())
        
#     def fit(self, topk):
        
#         row, col = [], []
#         index = np.argsort(-self.similarity)[:, 1:topk + 1]
#         for i, j in enumerate(index):

#             row.extend([i] * topk)
#             col.extend(j)

#         mask = sparse.csr_matrix((np.ones_like(row), (row, col)), shape = (self.num_item, self.num_item)).toarray()
        
#         w = self.similarity * mask + self.noise
        
#         w = w.astype(np.float32)
        
#         self.KNN = self.URM_train.dot(w)
        
    def fit(self, topk):
        
        row, col = [], []
        index = np.argsort(-self.similarity)[:, 1:topk + 1]
        for i, j in enumerate(index):

            row.extend([i] * topk)
            col.extend(j)

        mask = sparse.csr_matrix((np.ones_like(row), (row, col)), shape = (self.num_item, self.num_item)).toarray()
        
        w = self.similarity * mask + self.noise
        
        self.KNN = np.matmul(self.URM_train.toarray(), w)
        
    def evaluate_user(self, user_id):
        
        item_scores = self.KNN[user_id]
        
        return torch.tensor(item_scores).float()
    
    

class MultiVAE(nn.Module):
    
    def __init__(self, p_dims, q_dims = None, dropout = 0.5):
        super().__init__()
        
        self.p_dims = p_dims
        if q_dims is None:
            self.q_dims = p_dims[::-1]
        else:
            assert q_dims[0] == p_dims[-1], "Input and output dimension must equivalent for autoencoders."
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims
            
        self.drop = nn.Dropout(dropout)# by default, contain self.training
        self.tanh = nn.Tanh()
        
        self.temp_q = self.q_dims[:-1] + [self.q_dims[-1] * 2]# eg:[10,20,30] -> [10, 20, 60]
        
        
        
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for 
                                       d_in, d_out in zip(self.temp_q[:-1], self.temp_q[1:])])
        
        self.q_len = len(self.q_layers)
        
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])
        
        self.p_len = len(self.p_layers)
        self.init_weights()
        
        
    def init_layer(self, layers):
        
        for m in layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                    # not exactly same as author
        
    def init_weights(self):
        
        self.init_layer(self.q_layers)
        self.init_layer(self.p_layers)
        
            
    def forward(self, x):
        
        mu, logvar = self.encoder(x)
        z = self.reparameter(mu, logvar)# dimension (-1, k)
        output = self.decoder(z)
        
        return output, mu, logvar
    
    
    def encoder(self, x):
        
        x = F.normalize(x)
        x = self.drop(x)
        
        for i, layer in enumerate(self.q_layers):
            x = layer(x)
            if i != self.q_len - 1:
                x = self.tanh(x)
            else:
                mu = x[:, :self.q_dims[-1]]
                logvar = x[:, self.q_dims[-1]:]
                
        return mu, logvar
    
    def decoder(self, x):
        
        for i, layer in enumerate(self.p_layers):
            x = layer(x)
            if i != self.p_len - 1:
                x = self.tanh(x)
                
        return x
    
    def reparameter(self, mu, logvar):
        
        if self.training:# training flag in nn.Module
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu 
        else:
            return mu
            
    
    def loss(self, recon_x, x, mu, logvar, anneal = 1.0):
        
        BCE = - (F.log_softmax(recon_x, dim = 1) * x).sum(1).mean(-1) 
        KLD = - 0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean(-1)
        loss = BCE + anneal * KLD
        
        return loss 
    
    

class EASE(nn.Module):
    
    def __init__(self, URM_train, device):
        
        super().__init__()
        
        self.URM_train = URM_train # sparse csr matrix
        self.XtX = np.asarray(URM_train.T.dot(URM_train).todense(), dtype = np.float32)
        self.ii_diag = np.diag_indices(self.XtX.shape[0])
        
        self.device = device
        
    def fit(self, l2_norm = 1e-3):

        t_XtX = torch.tensor(self.XtX).to(self.device)

        t_XtX[self.ii_diag] += l2_norm

        P = torch.inverse(t_XtX)

        self.B = P / (-torch.diag(P))
        self.B[self.ii_diag] = 0.0
        
        return self.B
        
    def evaluate_user(self, user_id):
        
        user_profile = torch.tensor(self.URM_train[user_id].toarray()).to(self.device)
        item_scores = user_profile.float().view(-1)@(self.B)
        
        return item_scores
    
    

class GMF(nn.Module):
    def __init__(self, user_size, item_size, embed_size):
        
        super().__init__()
        
        self.user_size = user_size
        self.item_size = item_size
        self.embed_size = embed_size
        
        self.U = nn.Embedding(self.user_size, self.embed_size)
        nn.init.normal_(self.U.weight, std = 0.01)
        
        self.I = nn.Embedding(self.item_size, self.embed_size)
        nn.init.normal_(self.I.weight, std = 0.01)
                
        self.fc = nn.Linear(self.embed_size, 1)
        nn.init.kaiming_uniform_(self.fc.weight, a=1, nonlinearity='sigmoid')
        self.fc.bias.data.zero_()
              
        self.loss_fn = nn.BCEWithLogitsLoss()
          
    def forward(self, user, item):
        
        user_vec = self.U(user)
        item_vec = self.I(item)

        dot = self.fc(item_vec * user_vec).view(-1)
        
        #dot = (user_vec * item_vec).sum(1)
                
        return dot

    def loss(self, user, item, label):

        dot = self.forward(user, item)
        total_loss = self.loss_fn(dot, label)

        return total_loss
    
    def evaluate_user(self, user_id):
        
        user_vec = self.U(user_id)# 1, k
        item_vec = self.I.weight # n, k
        
        dot  = self.fc(item_vec * user_vec).view(-1) #n
        #dot = (user_vec * item_vec).sum(1)
        
        return dot


class MLP(nn.Module):
    
    def __init__(self, user_size, item_size, embed_size, layers, drop = 0):
        
        super().__init__()
        
        self.user_size = user_size
        self.item_size = item_size
        self.embed_size = embed_size
        
        self.U = nn.Embedding(self.user_size, self.embed_size)
        self.I = nn.Embedding(self.item_size, self.embed_size)
        self.drop = nn.Dropout(drop)
        module_list = []
        
        for i in range(1, len(layers)):
            module_list.append(self.drop)
            module_list.append(nn.Linear(layers[i-1], layers[i]))
            module_list.append(nn.ReLU())
                
        self.mlp = nn.Sequential(*module_list)
        self.output = nn.Linear(layers[-1], 1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        self._init_weight()
        
    def _init_weight(self):
        
        nn.init.normal_(self.U.weight, std = 0.01)
        nn.init.normal_(self.I.weight, std = 0.01)
        
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                
        nn.init.kaiming_uniform_(self.output.weight, a = 1, nonlinearity = 'sigmoid')
        
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
        
    def forward(self, user, item):
        
        user_vec = self.U(user)
        item_vec = self.I(item)
        uv = torch.cat([user_vec, item_vec], dim = -1)
        #uv = self.drop(uv)
        dot = self.mlp(uv)
        dot = self.output(dot).view(-1)
        #dot = self.fc(item_vec * user_vec).view(-1)
                
        return dot

    def loss(self, user, item, label):

        dot = self.forward(user, item)
        total_loss = self.loss_fn(dot, label)

        return total_loss
    
    def evaluate_user(self, user_id):
        
        user_vec = self.U(user_id).repeat(self.item_size, 1)# n, k
        item_vec = self.I.weight # n, k
        
        uv = torch.cat([user_vec, item_vec], dim = -1)
        #uv = self.drop(uv)
        dot = self.mlp(uv)
        dot = self.output(dot).view(-1)
        
        return dot

class NeuMF(nn.Module):
    
    def __init__(self, user_size, item_size, embed_size, layers, drop = 0):
        
        super().__init__()
        
        self.drop = nn.Dropout(drop)
        self.layers = layers
        self.user_size = user_size
        self.item_size = item_size
        self.GMF_embed_size = embed_size
        self.MLP_embed_size = self.layers[0]//2
        
        self.GMF_U = nn.Embedding(self.user_size, self.GMF_embed_size)
        self.GMF_I = nn.Embedding(self.item_size, self.GMF_embed_size)
        
        self.MLP_U = nn.Embedding(self.user_size, self.MLP_embed_size)
        self.MLP_I = nn.Embedding(self.item_size, self.MLP_embed_size)

        module_list = []
        
        for i in range(1, len(self.layers)):
            module_list.append(self.drop)
            module_list.append(nn.Linear(self.layers[i-1], self.layers[i]))
            module_list.append(nn.ReLU())
                
        self.mlp = nn.Sequential(*module_list)
        
        self.output = nn.Linear(self.layers[-1] + self.GMF_embed_size, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        #self.drop = nn.Dropout(0.1)
        
    def load_pretrain(self, GMF_model, MLP_model):
        
        self.MLP_U.weight.data.copy_(MLP_model.U.weight)
        self.MLP_I.weight.data.copy_(MLP_model.I.weight)
        
        self.GMF_U.weight.data.copy_(GMF_model.U.weight)
        self.GMF_I.weight.data.copy_(GMF_model.I.weight)

        for (m1, m2) in zip(
                self.mlp, MLP_model.mlp):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)
        
            
        out_weight = torch.cat([GMF_model.fc.weight, MLP_model.output.weight], -1)
        out_bias = GMF_model.fc.bias + MLP_model.output.bias
        
        self.output.weight.data.copy_(0.5 * out_weight)
        self.output.bias.data.copy_(0.5 * out_bias)
        
    def forward(self, user, item):
        
        GMF_U = self.GMF_U(user)
        GMF_I = self.GMF_I(item)
        
        MLP_U = self.MLP_U(user)
        MLP_I = self.MLP_I(item)
        
        GMF_UV = GMF_U * GMF_I # B , K1
        MLP_UV = self.mlp(torch.cat([MLP_U , MLP_I], -1)) # B, K2
        
        UV = torch.cat([GMF_UV, MLP_UV], dim = -1) # B, K1+K2

        #dot = self.drop(UV)
        dot = self.output(UV).view(-1)
                
        return dot
    
    def evaluate_user(self, user):
        
        GMF_U = self.GMF_U(user)#1,k
        GMF_I = self.GMF_I.weight #n, k
        MLP_U = self.MLP_U(user).repeat(self.item_size, 1)# 1, k
        MLP_I = self.MLP_I.weight#n, k
        
        GMF_UV = GMF_U * GMF_I # n , k
        MLP_UV = self.mlp(torch.cat([MLP_U , MLP_I], -1)) # n, K
        
        UV = torch.cat([GMF_UV, MLP_UV], dim = -1) # B, K1+K2

        #dot = self.drop(UV)
        dot = self.output(UV).view(-1)
                
        return dot
        

    def loss(self, user, item, label):

        dot = self.forward(user, item)
        total_loss = self.loss_fn(dot, label)

        return total_loss