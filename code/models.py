import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from numpy.linalg import norm

class NMF(nn.Module):
    """
    Neural Matrix Factorizaion model implementation
    """

    def __init__(self, num_genes, num_spots, params, device:str = 'cpu'):
        super().__init__()
        # Get relevant hyperparameters from the params dict
        latent_dim = params['latent_dim']

        # Initialize embedding layers for the users and for the items
        self.embedding_genes = nn.Embedding(num_embeddings=num_genes+1, embedding_dim=latent_dim).to(device)
        self.embedding_spots = nn.Embedding(num_embeddings=num_spots+1, embedding_dim=latent_dim).to(device)
        self.embedding_genes.weight = nn.Parameter(torch.normal(0, .1,
                                                                self.embedding_genes.weight.shape).to(device))
        self.embedding_spots.weight = nn.Parameter(torch.normal(0, .1,
                                                                self.embedding_spots.weight.shape).to(device))
        # self.linear = 

    def forward(self, gene_indices, spot_indices):
        # Get the gene and spot vector using the embedding layers
        gene_embedding = self.embedding_genes(gene_indices)
        spot_embedding = self.embedding_spots(spot_indices)

        # Calculate the expression for the gene-spot combination
        output = (gene_embedding * spot_embedding).sum(1)
        
        return output


class PMF(nn.Module):

    def __init__(self, num_genes, num_spots, params, device, lam_u=0.3, lam_v=0.3):
        super().__init__()
        self.lam_u = lam_u
        self.lam_v = lam_v

        latent_dim = params['latent_dim']

        self.embedding_genes = nn.Embedding(num_embeddings=num_genes+1, embedding_dim=latent_dim).to(device)
        self.embedding_spots = nn.Embedding(num_embeddings=num_spots+1, embedding_dim=latent_dim).to(device)
    
    def forward(self, matrix, gene_indices, spot_indices):
        
        gene_embedding = self.embedding_genes(gene_indices)
        spot_embedding = self.embedding_spots(spot_indices)

        predicted = torch.sigmoid(torch.matmul(gene_embedding, spot_embedding.t()))
        
        diff = (matrix - predicted)**2
        prediction_error = torch.sum(diff)

        u_regularization = self.lam_u * torch.sum(gene_embedding.norm(dim=1))
        v_regularization = self.lam_v * torch.sum(spot_embedding.norm(dim=1))
        
        return prediction_error + u_regularization + v_regularization


class NNMF(nn.Module):

    def __init__(self, num_genes, num_spots, params, device):
        super().__init__()

        latent_dim = params['latent_dim']

        self.embedding_genes = nn.Embedding(num_embeddings=num_genes+1, embedding_dim=latent_dim ,device=device)
        self.embedding_spots = nn.Embedding(num_embeddings=num_spots+1, embedding_dim=latent_dim, device=device)
        
        self.embedding_genes.weight = nn.Parameter(torch.normal(0, .1,
                                                                self.embedding_genes.weight.shape).to(device))
        self.embedding_spots.weight = nn.Parameter(torch.normal(0, .1,
                                                                self.embedding_spots.weight.shape).to(device))

        linear_layers = []
        for layer in params['layers_sizes']:
            linear_layers.append(nn.Linear(layer[0], layer[1]))
            linear_layers.append(nn.ReLU(inplace=True))
        linear_layers.append(nn.Linear(layer[1], 1).to(device))
        
        self.linear_layers = nn.Sequential(*linear_layers)
        
    def forward(self, gene_indices, spot_indices):
        
        gene_embedding = self.embedding_genes(gene_indices).to(torch.float)
        spot_embedding = self.embedding_spots(spot_indices).to(torch.float)

        x = torch.mul(gene_embedding, spot_embedding)
            
        output = self.linear_layers(x)

        return output


class NeuMF(nn.Module):
    def __init__(self, num_genes, num_spots, params, device):
        super(NeuMF, self).__init__()
        self.num_genes = num_genes
        self.num_spots = num_spots
        self.factor_num_mf = params['latent_dim']
        self.factor_num_mlp =  int(params['layers_sizes'][0]/2)
        self.layers = params['layers_sizes']
        # self.dropout = args.dropout

        self.embedding_user_mlp = nn.Embedding(num_embeddings=self.num_genes+1, embedding_dim=self.factor_num_mlp)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=self.num_spots+1, embedding_dim=self.factor_num_mlp)

        self.embedding_user_mf = nn.Embedding(num_embeddings=self.num_genes+1, embedding_dim=self.factor_num_mf)
        self.embedding_item_mf = nn.Embedding(num_embeddings=self.num_spots+1, embedding_dim=self.factor_num_mf)

        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(params['layers_sizes'][:-1], params['layers_sizes'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.ReLU())

        self.affine_output = nn.Linear(in_features=params['layers_sizes'][-1] + self.factor_num_mf, out_features=1)
        self.logistic = nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.embedding_user_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_user_mf.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mf.weight, std=0.01)
        
        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                
        nn.init.xavier_uniform_(self.affine_output.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, gene_indices, spot_indices):
        user_embedding_mlp = self.embedding_user_mlp(gene_indices)
        item_embedding_mlp = self.embedding_item_mlp(spot_indices)

        user_embedding_mf = self.embedding_user_mf(gene_indices)
        item_embedding_mf = self.embedding_item_mf(spot_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        mf_vector =torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating.squeeze()


class EdgeDetectNN(nn.Module):
    def __init__(self, params, device:str = 'cpu'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(24010, 1024)
        self.fc2 = nn.Linear(24010, 4)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0],-1)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x)
        x = self.fc2(x)
        return x


class NeighborsDetectNN(nn.Module):
    def __init__(self, params, device:str = 'cpu'):
        super().__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv1_2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv1_2_drop = nn.Dropout2d()
        self.fc1_1 = nn.Linear(10580, 1024)
        self.fc1_2 = nn.Linear(1024, 40)

        self.conv2_1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2_2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_2_drop = nn.Dropout2d()
        self.fc2_1 = nn.Linear(10580, 1024)
        self.fc2_2 = nn.Linear(1024, 40)

        self.cos = torch.nn.CosineSimilarity(dim=1)
        
    def forward(self, x1, x2):
        x1 = F.relu(F.max_pool2d(self.conv1_1(x1),2))
        x1 = F.relu(F.max_pool2d(self.conv1_2_drop(self.conv1_2(x1)), 2))
        x1 = x1.view(x1.shape[0],-1)
        x1 = F.relu(self.fc1_1(x1))
        x1 = F.dropout(x1)
        # x1 = self.fc1_2(x1)

        x2 = F.relu(F.max_pool2d(self.conv2_1(x2),2))
        x2 = F.relu(F.max_pool2d(self.conv2_2_drop(self.conv2_2(x2)), 2))
        x2 = x2.view(x2.shape[0],-1)
        x2 = F.relu(self.fc2_1(x2))
        x2 = F.dropout(x2)
        # x2 = self.fc2_2(x2)        

        output = torch.round(torch.sigmoid(self.cos(x1, x2)))
        # similarity = np.dot(x1,x2)/(norm(x1)*norm(x2))
        
        return output