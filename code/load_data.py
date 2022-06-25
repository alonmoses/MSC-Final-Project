import scanpy as sc
import stlearn as st
import pandas as pd

from dataclasses import dataclass, field
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

import torch
from torch.utils.data import DataLoader, Dataset


class ExpressionDataset(Dataset):
    """
    Generate expression dataset to use in the our models, where each sample should be a tuple of (gene, spot, expression)
    """

    def __init__(self, df, device):
        self.num_samples = len(df)
        self.genes = torch.tensor(df['gene'].values).to(device)
        self.spots = torch.tensor(df['spot'].values).to(device)
        self.labels = torch.tensor(df['expression'].sparse.to_dense().values)
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

    def set_dataloaders(self):

        # Load the expressions into dataframe
        expressions = self._get_expressions(dataset_name=self.dataset)
        
        # Split the data to train, val, test
        self._split_dataset(df=expressions)

        # Ceating Datasets
        ds_train = ExpressionDataset(df=self.df_train, device=self.device)
        ds_val = ExpressionDataset(df=self.df_val, device=self.device)
        ds_test = ExpressionDataset(df=self.df_test, device=self.device)
        dl_train_val = ExpressionDataset(df=self.df_train_val, device=self.device)

        # Creating DataLoaders
        self.dl_train = DataLoader(dataset=ds_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        self.dl_val = DataLoader(dataset=ds_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        self.dl_test = DataLoader(dataset=ds_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        self.dl_train_val = DataLoader(dataset=dl_train_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def _get_expressions(self, dataset_name: str ='V1_Human_Lymph_Node') -> pd.DataFrame:
        
        # Load dataset as stlearn AnnData object
        print(dataset_name)
        scanpy_anndata = sc.datasets.visium_sge(sample_id=dataset_name)
        self.stlearn_anndata = st.convert_scanpy(scanpy_anndata)

        # # Filter the data by min_counts and min_cells
        # _min_counts = 15000
        # _min_cells = 1000
        # st.pp.filter_genes(adata=stlearn_anndata, min_counts=_min_counts)
        # st.pp.filter_genes(adata=stlearn_anndata, min_cells=_min_cells)

        spots = self.stlearn_anndata.obs.index.values
        genes = self.stlearn_anndata.var.index.values
        expressions_matrix = pd.DataFrame.sparse.from_spmatrix(self.stlearn_anndata.X, columns=genes, index=spots)

        expressions = expressions_matrix.stack().reset_index()
        expressions.columns = ['spot', 'gene', 'expression']

        # Ordinal encoding the genes and spots for supported type
        oe = OrdinalEncoder()
        expressions[['spot', 'gene']] = oe.fit_transform(expressions[['spot', 'gene']].values)
        expressions[['spot', 'gene']] = expressions[['spot', 'gene']].astype(int)

        return expressions    

    def _split_dataset(self, df: pd.DataFrame, test_size: float = 0.1, val_size: float = 0.1):
        self.df_train_val, self.df_test = train_test_split(df, test_size=test_size)
        self.df_train, self.df_val = train_test_split(self.df_train_val, test_size=val_size)    

