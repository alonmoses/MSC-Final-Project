from dataclasses import dataclass, field
from email.mime import image
from os import path
import numpy as np
import pandas as pd
from copy import deepcopy

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import stlearn as st

from load_data import main as load_data_main
from models import EdgeDetectNN

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

        return train_epochs_losses, test_epochs_losses

    def train_batch(self, batch):
        genes, spots, y = batch
        genes.to(self.device)
        spots.to(self.device)
        y = y.float().to(self.device)
        # batch = batch[0].to(self.device)

        # Set model to train
        self.model.train()

        if self.model_name == 'NMF' or self.model_name == 'NNMF' or self.model_name == 'NeuMF':
            y_pred = self.model(genes, spots).to(self.device) # Predict
            loss = torch.mean((y - y_pred) ** 2) # Calculate loss

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
            genes.to(self.device)
            spots.to(self.device)
            y = y.float().to(self.device)

            if self.model_name == 'NMF' or self.model_name == 'NNMF' or self.model_name == 'NeuMF':
                y_pred = self.model(genes, spots) # Predict
                loss = torch.mean((y - y_pred) ** 2) # Calculate loss

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
            for set_dl in [dataset.dl_train_top_genes, dataset.dl_valid_top_genes, dataset.dl_test_top_genes]:
                for batch in set_dl:
                    gens, spots, y = batch
                    gens.to(device)
                    spots.to(device)
                    y_pred = model(gens, spots)
                    y_pred = np.clip(a=y_pred, a_min=0, a_max=None)

                    all_gens.extend(gens.tolist())
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

    def train_batch(self, batch):
        images, y = batch
        images.to(self.device)
        y = y.to(self.device)
        images_num, correct_predictions = 0, 0

        # Set model to train
        self.model.train()

        self.optimizer.zero_grad() # Zero gradients
        y_pred = self.model(images).to(self.device) # Predict
        loss = self.criterion(y_pred, y) # Calculate loss
        loss.backward() # Backpropagate
        self.optimizer.step() # Optimizer step

        _, predicted = torch.max(y_pred.data, 1)
        images_num += y.size(0)
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
            images_num += y.size(0)
            correct_predictions = correct_predictions + (predicted == y).sum().item() / images_num * 100

            return loss.item(), correct_predictions

def main():
    dataset_name = 'Visium_Mouse_Olfactory_Bulb'
    params = {'learning_rate': 0.001,
              'optimizer': "Adam"}

    edge_detect_model = EdgeDetectNN(params)

    train_dl, test_dl = load_data_main(dataset_name=dataset_name)
    classifier_execute = EdgeClassifyEngine(model= edge_detect_model,
                                            params = params,
                                            epochs = 20,
                                            dl_train = train_dl,
                                            dl_test = test_dl)

    classifier_execute.execute()

if __name__ == '__main__':
    main()