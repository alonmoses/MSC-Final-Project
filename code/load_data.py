from distutils.log import debug
import scanpy as sc
import stlearn as st
import pandas as pd
import numpy as np
import os
import random

from dataclasses import dataclass, field
from typing import Tuple

from sklearn.model_selection import train_test_split, KFold
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
    dataset: str = ''
    batch_size: int = 128
    num_workers: int = 1
    device: str = field(default = 'cpu')
    top_expressed_genes_number: int = 100

    def prepare_top_genes_data(self, x, data, smooth_type:str = ''):
        
        x = np.log(x+1)
        
        self.spots_num = data.n_obs
        self.genes_num = data.n_vars
        
        spots_values = data.obs.index.values
        genes_values = data.var.index.values
        df_expressions_matrix = pd.DataFrame(x, columns=genes_values, index=spots_values)
        df_expressions = df_expressions_matrix.stack().reset_index()
        df_expressions.columns = ['spot', 'gene', 'expression']
        

        if smooth_type:
            df_expressions = self.change_expression_based_on_neighbors(df_expressions, data, smooth_type)

        # Ordinal encoding the genes and spots for supported type
        self.oe_genes = OrdinalEncoder()
        df_expressions[['gene']] = self.oe_genes.fit_transform(df_expressions[['gene']].values)
        self.oe_spots = OrdinalEncoder()
        df_expressions[['spot']] = self.oe_spots.fit_transform(df_expressions[['spot']].values)
        df_expressions[['spot', 'gene']] = df_expressions[['spot', 'gene']].astype(int)


        # Creating DataLoaders - all genes 
        self.dl_train, self.dl_valid, self.dl_test = self.prepare_dl(df_expressions, split_ratio=0.1)
        
        #### get only top 100 expressed genes - NOT IN USE
        # genes_expressed = np.sum(x, axis=0) / (np.count_nonzero(x, axis=0) + 1)
        # top_genes_indices = genes_expressed.argsort()[-self.top_expressed_genes_number:][::-1]
        # self.top_genes_names = data.var.index[top_genes_indices]
        # top_genes_codes = self.oe_genes.transform(X=pd.DataFrame(np.array(self.top_genes_names)).values)[:, 0]
        # mask = df_expressions['gene'].isin(top_genes_codes)
        # df_expressions_top_genes = df_expressions.loc[mask]
        # return df_expressions_top_genes

        return df_expressions

    def change_expression_based_on_neighbors(self, df_expressions:pd.DataFrame, data, smooth_type):
        unique_spots = pd.unique(df_expressions.spot.values)
        for spot in unique_spots:
            spot_data = data.obs.loc[spot]
            spot_expression_with_neighbors = self.get_neighbors_expressions(df_expressions, data, spot, spot_data, smooth_type)
            df_expressions = self.calculate_smoothed_expression(df_expressions, spot, spot_expression_with_neighbors, smooth_type)
        
        return df_expressions

    @staticmethod
    def get_neighbors_expressions(df_expressions, data, spot, spot_data, smooth_type):
        if smooth_type in ['mean', 'mean_wrt_distance']:
            neighbors_list = [(-1,-1), (-1,0),  (-1,1),  (0,-1),  (0,1),  (1,-1),  (1,0), (1,1)]
        elif smooth_type in ['mean_wrt_distance_x2', 'mean_wrt_edge', 'mean_wrt_to_neighbors_tiles_means',
                             'mean_wrt_to_neighbors_tiles_means_remove_highest_distance']:
            neighbors_list = [(-2,-2), (-2,-1), (-2,0), (-2,1), (-2,2),
                              (-1,-2), (-1,-1), (-1,0),  (-1,1), (-1,2),
                              (0,-2), (0,-1),  (0,1), (0,2),
                              (1,-2), (1,-1), (1,0), (1,1), (1,2), 
                              (2,-2), (2,-1), (2,0), (2,1), (2,2)]
        array_row, array_col = spot_data['array_row'], spot_data['array_col']
        spot_expression = df_expressions.loc[df_expressions.spot == spot]

        for row_ind, col_ind in neighbors_list:
            neighbor_name = data.obs[(data.obs['array_row'] == array_row+row_ind) & (data.obs['array_col'] == array_col+col_ind)].index
            # is_neighbor = // TODO: 1 if the spot is a neighbor or 0 if not
            if len(neighbor_name) == 1:
                top_left_spot_data_expressions = df_expressions.loc[df_expressions.spot == neighbor_name[0]]
                spot_expression[f'{neighbor_name[0]}_{row_ind}_{col_ind}'] = top_left_spot_data_expressions['expression'].values.astype(float)
        
        return spot_expression

    def calculate_smoothed_expression(self, df_expressions, spot, spot_expression_with_neighbors, smooth_type):
        expression_column_names = spot_expression_with_neighbors.columns[2:]
        if smooth_type in ['mean', 'mean_wrt_distance_x2']:
            spot_expression_with_neighbors['expression_mean'] = spot_expression_with_neighbors[expression_column_names].mean(axis=1)
       
        elif smooth_type in ['mean_wrt_distance']:
            neighbors_names = expression_column_names[1:]
            spot_expression_with_neighbors['neighbors_mean'] = spot_expression_with_neighbors[neighbors_names].mean(axis=1)
            spot_expression_with_neighbors['expression_mean'] = (spot_expression_with_neighbors['expression'] + spot_expression_with_neighbors['neighbors_mean']) / 2
        
        elif smooth_type in ['mean_wrt_edge']:
            edge_data = pd.read_csv(f'{self.dataset}/edge_side_final.csv')
            spot_edge_location = edge_data[edge_data['spot'] == spot].has_edge.values[0]
            for neighbor_and_location in expression_column_names[1:]:
                neighbor_name, relative_row, relative_col = neighbor_and_location.split('_')
                neighbor_edge_location = edge_data[edge_data['spot'] == neighbor_name].has_edge.values[0]
                if neighbor_edge_location != spot_edge_location:
                    spot_expression_with_neighbors.drop(columns=[neighbor_and_location], inplace=True)
                    expression_column_names = expression_column_names.drop(neighbor_and_location)
            spot_expression_with_neighbors['expression_mean'] = spot_expression_with_neighbors[expression_column_names].mean(axis=1)
       
        elif smooth_type in ['mean_wrt_to_neighbors_tiles_means']:
            tiles_pixels_mean = pd.read_csv(f'{self.dataset}/tiles_pixels_mean_final.csv')
            spot_pixels_mean = tiles_pixels_mean[tiles_pixels_mean['spot'] == spot].tile_mean.values[0]
            for neighbor_and_location in expression_column_names[1:]:
                neighbor_name, _, _ = neighbor_and_location.split('_')
                neighbor_pixels_mean = tiles_pixels_mean[tiles_pixels_mean['spot'] == neighbor_name].tile_mean.values[0]
                if abs(spot_pixels_mean - neighbor_pixels_mean) > 25:
                    spot_expression_with_neighbors.drop(columns=[neighbor_and_location], inplace=True)
                    expression_column_names = expression_column_names.drop(neighbor_and_location)
            spot_expression_with_neighbors['expression_mean'] = spot_expression_with_neighbors[expression_column_names].mean(axis=1)

        elif smooth_type in['mean_wrt_to_neighbors_tiles_means_remove_highest_distance']:
            tiles_pixels_mean = pd.read_csv(f'{self.dataset}/tiles_pixels_mean_final.csv')
            spot_pixels_mean = tiles_pixels_mean[tiles_pixels_mean['spot'] == spot].tile_mean.values[0]
            max_distance = 0
            for neighbor_and_location in expression_column_names[1:]:
                neighbor_name, _, _ = neighbor_and_location.split('_')
                neighbor_pixels_mean = tiles_pixels_mean[tiles_pixels_mean['spot'] == neighbor_name].tile_mean.values[0]
                if abs(spot_pixels_mean - neighbor_pixels_mean) > max_distance:
                    neighbor_to_remove = neighbor_and_location
            spot_expression_with_neighbors.drop(columns=[neighbor_to_remove], inplace=True)
            expression_column_names = expression_column_names.drop(neighbor_to_remove)
            spot_expression_with_neighbors['expression_mean'] = spot_expression_with_neighbors[expression_column_names].mean(axis=1)

        spot_expression_with_neighbors.drop(columns=spot_expression_with_neighbors.columns[2:-1], inplace = True)
        df_expressions.loc[df_expressions.spot == spot, ['expression']] = spot_expression_with_neighbors['expression_mean'].astype(np.float32)

        return df_expressions


    def set_dataloaders(self, x, data, debug_mode:bool = False, smooth_type:str = ''):
        df_expressions_top_genes = self.prepare_top_genes_data(x, data, smooth_type)
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


@dataclass
class SpotsData(Data):
    def set_dataloaders(self, x, data, debug_mode:bool = False, smooth_type:str = ''):
        df_expressions = self.prepare_top_genes_data(x, data, smooth_type)
        # Creating DataLoaders- top N expressed genes
        if debug_mode:
            self.dl_train, self.dl_valid, self.dl_test = self.prepare_dl_spots(df_expressions, test_size=11850)
        else:
            self.dl_train, self.dl_valid, self.dl_test = self.prepare_dl_spots(df_expressions)
    
    def prepare_dl_spots(self, expression_df, test_size:int = 1124304):

        # group the data by spots in order to train batched per spot
        shuffled_data = [df for _, df in expression_df.groupby('spot')]
        random.shuffle(shuffled_data)
        shuffled_dataframe = pd.concat(shuffled_data)

        # split data to train, validaion and test sets
        df_train, df_test = train_test_split(shuffled_dataframe, test_size=test_size, shuffle=False)
        df_train, df_valid = train_test_split(df_train, test_size=test_size, shuffle=False)
        
        # Ceating Datasets- top N expressed genes
        ds_train = ExpressionDataset(df=df_train, device=self.device)
        ds_valid = ExpressionDataset(df=df_valid, device=self.device)
        ds_test = ExpressionDataset(df=df_test, device=self.device)        

        # Creating DataLoaders- top N expressed genes
        dl_train = DataLoader(dataset=ds_train, batch_size=self.top_expressed_genes_number, num_workers=self.num_workers, shuffle=False)
        dl_valid = DataLoader(dataset=ds_valid, batch_size=self.top_expressed_genes_number, num_workers=self.num_workers, shuffle=False)
        dl_test = DataLoader(dataset=ds_test, batch_size=self.top_expressed_genes_number, num_workers=self.num_workers, shuffle=False)
        
        return dl_train, dl_valid, dl_test


class TilesDataset(Dataset):
    def __init__(self, data, transform, inference:bool = False):
        super().__init__()
        self.data = data.values
        self.transform = transform
        self.inference = inference
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.inference:
            img_name = self.data[index][-1]
        else:
            img_name, label = self.data[index]
        img_path = os.path.join(img_name)
        image = img.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        if self.inference:
            return image
        return image, label


@dataclass
class TilesData:
    dataset: str
    batch_size: int = 8
    num_workers: int = 1
    device: str = field(default = 'cpu')
    inference: bool = False
    
    def read_data(self, data_path:str):
        data = pd.read_csv(data_path)
        train_data, test_data = train_test_split(data, stratify=data.has_edge, test_size=0.2, random_state=42)
        return train_data, test_data
    
    def set_dataloaders(self, train_data, test_data, k_fold=0, train_sampler=None, valid_sampler=None):
        train_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize([0,0,0],[1,1,1])])
        test_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize([0,0,0],[1,1,1])])

        if k_fold:
            train_dataset = TilesDataset(train_data, train_transform)
            valid_dataset = TilesDataset(train_data, train_transform)
            
            train_loader = DataLoader(dataset=train_dataset, batch_size = self.batch_size, shuffle=False, num_workers=self.num_workers, sampler=train_sampler)
            valid_loader = DataLoader(dataset=valid_dataset, batch_size = self.batch_size, shuffle=False, num_workers=self.num_workers, sampler=valid_sampler) 

            return train_loader, valid_loader

        else:
            train_dataset = TilesDataset(train_data, train_transform, self.inference)
            test_dataset = TilesDataset(test_data, test_transform, self.inference) 
            
            train_loader = DataLoader(dataset=train_dataset, batch_size = self.batch_size, shuffle=True, num_workers=self.num_workers)
            test_loader = DataLoader(dataset=test_dataset, batch_size = self.batch_size, shuffle=False, num_workers=self.num_workers)

            return train_loader, test_loader


class PairsTilesDataset(Dataset):
    def __init__(self, data, transform, inference:bool = False):
        super().__init__()
        self.data = data.values
        self.transform = transform
        self.inference = inference
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.inference:
            img1, img2 = self.data[index][-2:]
        else:
            img1, img2, label = self.data[index]
        img_path_1 = os.path.join(img1)
        image1 = img.imread(img_path_1)
        img_path_2 = os.path.join(img2)
        image2 = img.imread(img_path_2)
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        if self.inference:
            return image1, image2
        return image1, image2, label

@dataclass
class PairsTilesData:
    dataset: str
    batch_size: int = 8
    num_workers: int = 1
    device: str = field(default = 'cpu')
    inference: bool = False
    
    def read_data(self, data_path:str):
        data = pd.read_csv(data_path)
        train_data, test_data = train_test_split(data, stratify=data.are_neighbors, test_size=0.2, random_state=42)
        return train_data, test_data

    def set_dataloaders(self, train_data, test_data, k_fold=0, train_sampler=None, valid_sampler=None):
        train_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize([0,0,0],[1,1,1])])
        test_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize([0,0,0],[1,1,1])])

        if k_fold:
            train_dataset = PairsTilesDataset(train_data, train_transform)
            valid_dataset = PairsTilesDataset(train_data, train_transform)
            
            train_loader = DataLoader(dataset=train_dataset, batch_size = self.batch_size, shuffle=False, num_workers=self.num_workers, sampler=train_sampler)
            valid_loader = DataLoader(dataset=valid_dataset, batch_size = self.batch_size, shuffle=False, num_workers=self.num_workers, sampler=valid_sampler) 

            return train_loader, valid_loader

        else:
            train_dataset = PairsTilesDataset(train_data, train_transform, self.inference)
            test_dataset = PairsTilesDataset(test_data, test_transform, self.inference) 
            
            train_loader = DataLoader(dataset=train_dataset, batch_size = self.batch_size, shuffle=True, num_workers=self.num_workers)
            test_loader = DataLoader(dataset=test_dataset, batch_size = self.batch_size, shuffle=False, num_workers=self.num_workers)

            return train_loader, test_loader

        
def load_visium_data(dataset_name, data_type:str = 'random_data', min_cells:int = 0, min_counts:int = 0, smooth_type:str = '', debug_mode:bool = False):
    dataset_path = f'{dataset_name}'
    st_data = st.Read10X(dataset_path)
    if min_counts:
        st.pp.filter_genes(st_data, min_counts=min_counts)
    if min_cells:
        st.pp.filter_genes(st_data, min_cells=min_cells)
    X = st_data.X.toarray()
    if debug_mode:
        st_data.var.drop(st_data.var.index[100:], inplace=True)
        X = X[:,:100]
    if data_type == 'random_data':
        dataset = Data(dataset=dataset_name, num_workers=5)
    if data_type == 'spots_data':    
        dataset = SpotsData(num_workers=5)
    dataset.set_dataloaders(X, st_data, debug_mode, smooth_type)
    return dataset, st_data

def load_edge_detection_data(dataset_name):
    tiles_data = TilesData(dataset=f'/FPST/data/{dataset_name}')
    train_data, test_data = tiles_data.read_data(data_path=f'{dataset_name}/adata.csv')
    train_dl, test_dl = tiles_data.set_dataloaders(train_data, test_data)
    return train_dl, test_dl

def load_pairs_tiles_data(dataset_name):
    tiles_data = PairsTilesData(dataset=f'/FPST/data/{dataset_name}')
    train_data, test_data = tiles_data.read_data(data_path=f'{dataset_name}/pairs_neighborhood.csv')
    train_dl, test_dl = tiles_data.set_dataloaders(train_data, test_data)
    return train_dl, test_dl

if __name__ == '__main__':
    dataset_name = '/FPST/data/Visium_Mouse_Olfactory_Bulb'
    # load_edge_detection_data(dataset_name)
    # load_pairs_tiles_data(dataset_name)
    load_visium_data(dataset_name, 'random_data', min_cells=177, min_counts=10, smooth_type='mean_wrt_to_neighbors_tiles_means', debug_mode=True)

