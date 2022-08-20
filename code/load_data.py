import scanpy as sc
import stlearn as st
import pandas as pd
import numpy as np
import os

from dataclasses import dataclass, field
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.image as img
import torchvision.transforms as transforms



class ExpressionDataset(Dataset):
    """
    Generate expression dataset to use in the our models, where each sample should be a tuple of (gene, spot, expression)
    """

    def __init__(self, df, device):
        self.num_samples = len(df)
        self.genes = torch.tensor(df['gene'].values).to(device)
        self.spots = torch.tensor(df['spot'].values).to(device)
        self.labels = torch.tensor(df['expression'].values)
        self.num_genes = df['gene'].max()
        self.num_spots = df['spot'].max()

    def __getitem__(self, index):
        gene = self.genes[index]
        spot = self.spots[index]
        label = self.labels[index].item()
        return gene, spot, label

    def __len__(self):
        return self.num_samples

    def get_all_data(self):
        return self.genes, self.spots, self.labels


@dataclass
class Data:
    dataset: str
    batch_size: int = 128
    num_workers: int = 1
    device: str = field(default = 'cpu')

    def set_dataloaders(self, x, data, top_expressed_genes_number=100):
        x = np.log(x+1)
        
        spots_values = data.obs.index.values
        genes_values = data.var.index.values
        df_expressions_matrix = pd.DataFrame(x, columns=genes_values, index=spots_values)
        df_expressions = df_expressions_matrix.stack().reset_index()
        df_expressions.columns = ['spot', 'gene', 'expression']
        
        # Ordinal encoding the genes and spots for supported type
        self.oe_genes = OrdinalEncoder()
        df_expressions[['gene']] = self.oe_genes.fit_transform(df_expressions[['gene']].values)
        self.oe_spots = OrdinalEncoder()
        df_expressions[['spot']] = self.oe_spots.fit_transform(df_expressions[['spot']].values)
        df_expressions[['spot', 'gene']] = df_expressions[['spot', 'gene']].astype(int)
        print(df_expressions.head())
        # Creating DataLoaders - all genes 
        self.dl_train, self.dl_val, self.dl_test = self.prepare_dl(df_expressions, split_ratio = 0.1)
        
        genes_expressed = np.sum(x, axis=0) / (np.count_nonzero(x, axis=0) + 1)
        top_genes_indices = genes_expressed.argsort()[-top_expressed_genes_number:][::-1]
        self.top_genes_names = data.var.index[top_genes_indices]
        top_genes_codes = self.oe_genes.transform(X=pd.DataFrame(np.array(self.top_genes_names)).values)[:, 0]
        mask = df_expressions['gene'].isin(top_genes_codes)
        df_expressions_top_genes = df_expressions.loc[mask]

        # Creating DataLoaders- top N expressed genes
        self.dl_train_top_genes, self.dl_valid_top_genes, self.dl_test_top_genes = self.prepare_dl(df_expressions_top_genes, split_ratio = 0.1)
    
    def prepare_dl(self, expression_df, split_ratio = 0.1):
        # split data to train, validaion and test sets
        df_train, df_test = train_test_split(expression_df, test_size=split_ratio)
        df_train, df_valid = train_test_split(df_train, test_size=split_ratio)
        
        # Ceating Datasets- top N expressed genes
        ds_train = ExpressionDataset(df=df_train, device=self.device)
        ds_valid = ExpressionDataset(df=df_valid, device=self.device)
        ds_test = ExpressionDataset(df=df_test, device=self.device)        

        # Creating DataLoaders- top N expressed genes
        dl_train = DataLoader(dataset=ds_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        dl_valid = DataLoader(dataset=ds_valid, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        dl_test = DataLoader(dataset=ds_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        
        return dl_train, dl_valid, dl_test

class TilesDataset(Dataset):
    def __init__(self, data, transform):
        super().__init__()
        self.data = data.values
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, label = self.data[index]
        img_path = os.path.join(img_name)
        image = img.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


@dataclass
class TilesData:
    dataset: str
    batch_size: int = 8
    num_workers: int = 1
    device: str = field(default = 'cpu')
    
    def set_dataloaders(self, data_path:str):
        data = pd.read_csv(data_path)
        train_data, test_data = train_test_split(data, stratify=data.has_edge, test_size=0.2, random_state=1)

        train_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize([0,0,0],[1,1,1])])
        test_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize([0,0,0],[1,1,1])])

        train_dataset = TilesDataset(train_data, train_transform)
        test_dataset = TilesDataset(test_data, test_transform) 
        
        train_loader = DataLoader(dataset=train_dataset, batch_size = self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size = self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, test_loader


def main(dataset_name):
    tiles_data = TilesData(dataset=f'/FPST/data/{dataset_name}')
    train_dl, test_dl = tiles_data.set_dataloaders(data_path=f'/FPST/data/{dataset_name}/adata.csv')
    return train_dl, test_dl

if __name__ == '__main__':
    dataset_name = 'Visium_Mouse_Olfactory_Bulb'
    main(dataset_name)

