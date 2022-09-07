from cmath import nan
from dataclasses import dataclass, field
from email.mime import image
from os import path
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.image as img

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

import stlearn as st

from load_data import load_edge_detection_data, load_visium_data, TilesData
from image_data import generate_tiles
from models import EdgeDetectNN, NMF
from losses import RMSELossWithoutZeros

@dataclass
class engine:
    model: nn.Module
    model_name: str
    params: dict
    dl_train: DataLoader
    dl_test: DataLoader
    verbose: str = field(default=True)
    epochs: int = field(default = 20)
    criterion: nn = field(default_factory=nn.MSELoss)
    device: str = field(default='cpu')

    def __post_init__(self):
        self.model = self.model.to(self.device)
        self.set_optimizer()

    def execute(self) -> tuple([np.array, np.array]):
        train_epochs_losses, train_batch_losses = np.array([]), np.array([])
        test_epochs_losses, test_batch_losses = np.array([]), np.array([])

        for epoch in range(self.epochs): # Iterate over all epochs
            pbar = tqdm(iter(self.dl_train),
                        desc=f'Train epoch {epoch}/{self.epochs}',
                        disable=not self.verbose)
            for batch in pbar: # Iterate over all batches
                train_batch_loss = self.train_batch(batch)
                train_batch_losses = np.append(train_batch_losses, [train_batch_loss])

            for batch in self.dl_test: # Iterate over all batches
                test_batch_loss = self.eval_batch(batch)
                test_batch_losses = np.append(test_batch_losses, [test_batch_loss])                

            train_epoch_loss = np.sqrt(np.nanmean(train_batch_losses))
            test_epoch_loss = np.sqrt(np.nanmean(test_batch_losses))

            print(f'Epoch #{epoch} Train Loss: {train_epoch_loss}')
            print(f'Epoch #{epoch} Test Loss: {test_epoch_loss}')

            train_epochs_losses = np.append(train_epochs_losses, [train_epoch_loss])
            test_epochs_losses = np.append(test_epochs_losses, [test_epoch_loss])
            
            if (len(test_epochs_losses) >=3 and (test_epochs_losses[-2] - test_epochs_losses[-1]) < 0.01 and (test_epochs_losses[-3] - test_epochs_losses[-2]) < 0.01):
                print("Early stopping")
                break

        return train_epochs_losses, test_epochs_losses

    def train_batch(self, batch):
        genes, spots, y = batch
        genes = genes.to(self.device)
        spots = spots.to(self.device)
        y = y.float().to(self.device)

        # Set model to train
        self.model.train()

        if self.model_name == 'NMF' or self.model_name == 'NNMF' or self.model_name == 'NeuMF':
            y_pred = self.model(genes, spots).to(self.device) # Predict
            loss = self.criterion(y_pred, y)#torch.mean((y - y_pred) ** 2) # Calculate loss

        if self.model_name == 'PMF':
            loss = self.model(y, genes, spots) # Calculate loss
        self.optimizer.zero_grad() # Zero gradients
        loss.backward() # Backpropagate
        self.optimizer.step() # Optimizer step
        
        return loss.item()

    def eval_batch(self, batch):

        # Set model to eval
        self.model.eval()
        
        with torch.no_grad():
            genes, spots, y = batch
            genes = genes.to(self.device)
            spots = spots.to(self.device)
            y = y.float().to(self.device)

            if self.model_name == 'NMF' or self.model_name == 'NNMF' or self.model_name == 'NeuMF':
                y_pred = self.model(genes, spots).to(self.device) # Predict
                # loss = torch.mean((y - y_pred) ** 2) # Calculate loss
                loss = self.criterion(y_pred, y)

            if self.model_name == 'PMF':
                loss = self.model(y, genes, spots) # Calculate loss
            
            return loss.item()

    def set_optimizer(self):
        print(self.model)
        self.optimizer = getattr(optim, self.params['optimizer'])(self.model.parameters(), lr=self.params['learning_rate'])

        
    def create_reconstructed_data(model, dataset, data, device):
        model.eval()
        all_gens = []
        all_spots = []
        expressions_pred = []
        expressions_true = []

        with torch.no_grad():
            for set_dl in [dataset.dl_train, dataset.dl_valid, dataset.dl_test]:
                for batch in set_dl:
                    genes, spots, y = batch
                    genes = genes.to(device)
                    spots = spots.to(device)
                    y_pred = model(genes, spots).to(device)
                    y_pred = np.clip(a=y_pred.cpu(), a_min=0, a_max=None)

                    all_gens.extend(genes.tolist())
                    all_spots.extend(spots.tolist())
                    expressions_pred.extend(y_pred.tolist())
                    expressions_true.extend(y.tolist())
                    
        df_expressions_preds = pd.DataFrame({'gene': all_gens, 'spot': all_spots, 'expression': expressions_pred})
        df_expressions_preds[['gene']] = dataset.oe_genes.inverse_transform(df_expressions_preds[['gene']].values)
        df_expressions_preds[['spot']] = dataset.oe_spots.inverse_transform(df_expressions_preds[['spot']].values)
        df_expressions_true = df_expressions_preds.copy()
        df_expressions_true['expression'] = expressions_true
        
        df_expressions_preds_matrix = df_expressions_preds.pivot(index='spot', columns='gene', values='expression')
        df_expressions_true_matrix = df_expressions_true.pivot(index='spot', columns='gene', values='expression')
        
        new_data = deepcopy(data)
        tmp_genes_locations = [data.var.index.get_loc(key=gene_key) for gene_key in df_expressions_true_matrix.columns]
        print(new_data.X[:, tmp_genes_locations], df_expressions_preds_matrix.values)
        new_data.X[:, tmp_genes_locations] = df_expressions_preds_matrix.values
        
        return df_expressions_preds, df_expressions_true, new_data
    
    def cluster_reconstructed_data(data, plot=True):
        new_data_clusters = deepcopy(data)
        st.pp.normalize_total(new_data_clusters)
        # run PCA for gene expression data
        st.em.run_pca(new_data_clusters, n_comps=50)
        # K-means clustering
        st.tl.clustering.kmeans(new_data_clusters, n_clusters=7, use_data="X_pca", key_added="X_pca_kmeans")
        if plot:
            st.pl.cluster_plot(new_data_clusters, use_label="X_pca_kmeans")


@dataclass
class EdgeClassifyEngine:
    model: nn.Module
    params: dict
    dl_train: DataLoader
    dl_test: DataLoader
    verbose: str = field(default=True)
    epochs: int = field(default = 20)
    criterion: nn = field(default_factory=nn.CrossEntropyLoss)
    device: str = field(default='cpu')

    def __post_init__(self):
        print(self.model)
        self.model = self.model.to(self.device)
        self.set_optimizer()

    def set_optimizer(self):
        self.optimizer = getattr(optim, self.params['optimizer'])(self.model.parameters(), lr=self.params['learning_rate'])
    
    def execute(self) -> tuple([np.array, np.array]):
        train_epochs_losses, train_batch_losses = np.array([]), np.array([])
        test_epochs_losses, test_batch_losses = np.array([]), np.array([])
        train_epoch_accuracies, train_batch_accuracies = np.array([]), np.array([])
        test_epoch_accuracies, test_batch_accuracies = np.array([]), np.array([])


        for epoch in range(self.epochs): # Iterate over all epochs
            pbar = tqdm(iter(self.dl_train),
                        desc=f'Train epoch {epoch}/{self.epochs}',
                        disable=not self.verbose)

            for batch in pbar: # Iterate over all batches
                train_batch_loss, train_batch_accuracy = self.train_batch(batch)
                train_batch_losses = np.append(train_batch_losses, [train_batch_loss])
                train_batch_accuracies = np.append(train_batch_accuracies, [train_batch_accuracy])

            for batch in self.dl_test: # Iterate over all batches
                test_batch_loss, test_batch_accuracy = self.eval_batch(batch)
                test_batch_losses = np.append(test_batch_losses, [test_batch_loss])
                test_batch_accuracies = np.append(test_batch_accuracies, [test_batch_accuracy])  

            train_epoch_loss = np.sqrt(np.nanmean(train_batch_losses))
            train_epoch_accuracy = np.mean(train_batch_accuracies)

            test_epoch_loss = np.sqrt(np.nanmean(test_batch_losses))
            test_epoch_accuracy = np.mean(test_batch_accuracies)

            print(f'Epoch #{epoch} Train Loss: {train_epoch_loss}, Accuracy: {train_epoch_accuracy}')
            print(f'Epoch #{epoch} Test Loss: {test_epoch_loss}, Accuracy: {test_epoch_accuracy}')


            train_epochs_losses = np.append(train_epochs_losses, [train_epoch_loss])
            test_epochs_losses = np.append(test_epochs_losses, [test_epoch_loss])

            if (len(test_epochs_losses) >=3 
                and ((test_epochs_losses[-2] - test_epochs_losses[-1]) < 0.001 and (test_epochs_losses[-3] - test_epochs_losses[-2]) < 0.001)
                and ((train_epochs_losses[-2] - train_epochs_losses[-1]) < 0.001 and (train_epochs_losses[-3] - train_epochs_losses[-2]) < 0.001)):
                print("Early stopping")
                break

    def train_batch(self, batch):
        # Set model to train
        self.model.train()
        images, y = batch
        images.to(self.device)
        y = y.to(self.device)
        images_num, correct_predictions = 0, 0

        self.optimizer.zero_grad() # Zero gradients
        y_pred = self.model(images).to(self.device) # Predict
        loss = self.criterion(y_pred, y) # Calculate loss
        loss.backward() # Backpropagate
        self.optimizer.step() # Optimizer step

        _, predicted = torch.max(y_pred.data, 1)
        images_num = y.size(0)
        correct_predictions = correct_predictions + (predicted == y).sum().item() / images_num * 100
        
        return loss.item(), correct_predictions

    def eval_batch(self, batch):
        # Set model to eval
        self.model.eval()
        images_num, correct_predictions = 0, 0

        with torch.no_grad():
            images, y = batch
            images.to(self.device)
            y = y.to(self.device)

            y_pred = self.model(images).to(self.device) # Predict
            loss = self.criterion(y_pred, y) # Calculate loss

            _, predicted = torch.max(y_pred.data, 1)
            images_num = y.size(0)
            correct_predictions = correct_predictions + (predicted == y).sum().item() / images_num * 100

            return loss.item(), correct_predictions
    
    def generate_labeled_data(self, dataset_name:str):
        batch_idx = 0
        adata = generate_tiles(dataset_name)
        tiles_data = TilesData(dataset=dataset_name, inference=True)
        _, infer_preds_dl = tiles_data.set_dataloaders(adata.obs, adata.obs)
        adata.obs['has_edge'] = None
        self.model.eval()
        with torch.no_grad():
            for batch in infer_preds_dl:
                pred = self.model(batch).to(self.device)
                _, predicted = torch.max(pred.data, 1)
                adata.obs.loc[batch_idx: batch_idx+8,'has_edge'] = predicted
                batch_idx += 8
        return adata
            

def train_tiles_for_edges(k_fold=0):
    dataset_name = 'Visium_Mouse_Olfactory_Bulb'
    params = {'learning_rate': 0.001,
              'optimizer': "Adam"}

    edge_detect_model = EdgeDetectNN(params)

    if k_fold:
        tiles_data = TilesData(dataset=f'/FPST/data/{dataset_name}')
        train_data, test_data = tiles_data.read_data(data_path=f'/FPST/data/{dataset_name}/adata.csv')
        
        k_fold = KFold(n_splits=k_fold, shuffle=True,)
        for fold, (train_indices, valid_indices) in enumerate(k_fold.split(train_data)):
            print(f'Fold : {fold}')
            # create samplers
            train_sampler = SubsetRandomSampler(train_indices) 
            valid_sampler = SubsetRandomSampler(valid_indices)
            train_dl, valid_dl = tiles_data.set_dataloaders(train_data, test_data, k_fold=k_fold, train_sampler=train_sampler, valid_sampler=valid_sampler)
            classifier_execute = EdgeClassifyEngine(model= edge_detect_model,
                                                    params = params,
                                                    epochs = 20,
                                                    dl_train = train_dl,
                                                    dl_test = valid_dl)
            classifier_execute.execute()
        adata = classifier_execute.generate_labeled_data(f'/FPST/data/{dataset_name}')
        adata.obs.to_csv(f'/FPST/data/{dataset_name}/adata_final.csv')
    else:
        train_dl, test_dl = load_edge_detection_data(dataset_name=dataset_name)
        classifier_execute = EdgeClassifyEngine(model= edge_detect_model,
                                                params = params,
                                                epochs = 20,
                                                dl_train = train_dl,
                                                dl_test = test_dl)

    classifier_execute.execute()

def train_data_for_imputation(dataset_name:str = 'Visium_Mouse_Olfactory_Bulb', data_type:str = 'random_data', debug_mode:bool = False):
    dataset, data = load_visium_data(dataset_name, data_type, min_cells=177, min_counts=10, debug_mode=debug_mode)
    dl_train = dataset.dl_train
    dl_test = dataset.dl_test


    params = {'learning_rate': 0.001,
              'optimizer': "RMSprop",
              'latent_dim': 20,
              'batch_size': 32}

    nmf_model = NMF(dataset.genes_num, dataset.spots_num, params=params)
    nmf_execute = engine(model = nmf_model,
                         model_name = 'NMF',
                         params = params,
                         epochs = 2,
                         criterion = RMSELossWithoutZeros(),
                         dl_train = dl_train,
                         dl_test = dl_test,
                         device = 'cpu')
    nmf_train_losses, nmf_test_losses = nmf_execute.execute()
    df_expressions_preds, df_expressions_true, reconstructed_data = engine.create_reconstructed_data(nmf_model, dataset, data, 'cpu')
    engine.cluster_reconstructed_data(reconstructed_data)


if __name__ == '__main__':
    # train_tiles_for_edges(k_fold=5)
    train_data_for_imputation(dataset_name='/FPST/data/Visium_Mouse_Olfactory_Bulb', data_type='spots_data', debug_mode=True)