from dataclasses import dataclass, field
from os import path
import numpy as np

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm


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

            # Print loss after each 10 epochs
            # if epoch % 10 == 0: 
            print(f'Epoch #{epoch} Train Loss: {train_epoch_loss}')
            print(f'Epoch #{epoch} Test Loss: {test_epoch_loss}')

            train_epochs_losses = np.append(train_epochs_losses, [train_epoch_loss])
            test_epochs_losses = np.append(test_epochs_losses, [test_epoch_loss])

        return train_epochs_losses, test_epochs_losses

    def train_batch(self, batch):
        # genes, spots, y = batch
        # genes.to(self.device)
        # spots.to(self.device)
        # y = y.float().to(self.device)
        batch = batch[0].to(self.device)

        # Set model to train
        self.model.train()

        if self.model_name == 'NMF' or self.model_name == 'NNMF' or self.model_name == 'NeuMF':
            # Predict
            y_pred = self.model(batch[:,0], batch[:,1])

            # Calculate loss
            loss = torch.mean((batch[:, 2] - y_pred) ** 2)

        if self.model_name == 'PMF':
            # Calculate loss
            loss = self.model(y, batch[:,0], batch[:,1])

        # Zero gradients
        self.optimizer.zero_grad()

        # Backpropagate
        loss.backward()

        # Optimizer step
        self.optimizer.step()
        
        return loss.item()

    def eval_batch(self, batch):

        # Set model to eval
        self.model.eval()
        
        with torch.no_grad():
            # genes, spots, y = batch
            # genes.to(self.device)
            # spots.to(self.device)
            # y = y.float().to(self.device)
            batch = batch[0].to(self.device)
            
            if self.model_name == 'NMF' or self.model_name == 'NNMF' or self.model_name == 'NeuMF':
                # Predict
                y_pred = self.model(batch[:,0], batch[:,1])
                
                # Calculate loss
                loss = torch.mean((batch[:, 2] - y_pred) ** 2)

            if self.model_name == 'PMF':
                loss = self.model(y, batch[:,0], batch[:,1])
            
            return loss.item()

    def set_optimizer(self):
        print(self.model)
        self.optimizer = getattr(optim, self.params['optimizer'])(self.model.parameters(), lr=self.params['learning_rate'])

        

    