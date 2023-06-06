import numpy as np
import torch
from torch.utils.data import Dataset

class Stock_Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, x, y=None, normalized=False, initscaler=True):
        # Normalized
        stack = np.column_stack((x, y))

        if (normalized):
            normalstack = scaler.fit_transform(stack) if initscaler else scaler.transform(stack)
            x = normalstack[:, :-1]
            y = y if y is None else normalstack[:, -1]

        self.y = y if y is None else torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)