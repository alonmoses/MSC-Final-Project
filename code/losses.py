import torch
from torch import nn

class RMSELossWithoutZeros(nn.Module):

    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps  # Add eps to avoid devision by 0

    def forward(self, yhat, y, **kwargs):
        # Create mask for all non zero items in the tensor
        non_zero_mask = torch.nonzero(y, as_tuple=True)
        y_non_zeros = y[non_zero_mask]  # Keep only non zero in y
        yhat_non_zeros = yhat[non_zero_mask]    # Keep only non zero in y_hat

        loss = torch.sqrt(self.mse(yhat_non_zeros, y_non_zeros) + self.eps)
        return loss