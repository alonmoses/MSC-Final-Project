import torch
from torch import nn


class NMF(nn.Module):
    """
    Neural Matrix Factorizaion model implementation
    """

    def __init__(self, num_genes, num_spots, params, device):
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
        # out_size = params['layers_sizes'][0][0]

#         self.linear_layers = nn.ModuleList().to(device)

#         for (in_size, out_size) in features_sizes:
#             self.linear_layers.append(nn.Linear(in_size, out_size).to(device))

        self.out_layer = nn.Linear(latent_dim, 1).to(device)
        self.relu = nn.ReLU().to(device)
    
    def forward(self, gene_indices, spot_indices):
        
        gene_embedding = self.embedding_genes(gene_indices).to(torch.float)
        spot_embedding = self.embedding_spots(spot_indices).to(torch.float)

        x = torch.mul(gene_embedding, spot_embedding)
        # x = torch.cat([gene_embedding, spot_embedding], 1)

        # for idx, _ in enumerate(self.linear_layers):
        #     x = self.linear_layers[idx](x)
            
        output = self.out_layer(x)

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
