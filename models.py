from math import prod
from pdb import set_trace as TT

from omegaconf import DictConfig
import torch as th


# Define the model, mapping RGB images to 3D binary encodings
class ConvDense(th.nn.Module):
    def __init__(self, in_shape, out_shape, cfg: DictConfig):    
        super().__init__()
        self.out_shape = out_shape
        self.conv1 = th.nn.Conv2d(4, 6, 5)
        self.pool = th.nn.MaxPool2d(2, 2)
        self.conv2 = th.nn.Conv2d(6, 16, 5)
        self.conv3 = th.nn.Conv2d(16, 8, 5)
        fc_shape = prod(self.pool(self.conv3(self.pool(self.conv2(self.pool(self.conv1(th.zeros(1, *in_shape))))))).shape[1:])
        self.fc1 = th.nn.Linear(fc_shape, 120)
        self.fc2 = th.nn.Linear(120, 84)
        out_size_flat = prod(out_shape)
        self.fc3 = th.nn.Linear(84, out_size_flat)
    
    def forward(self, x):
        b = x.shape[0]
        x = self.pool(th.nn.functional.relu(self.conv1(x)))
        x = self.pool(th.nn.functional.relu(self.conv2(x)))
        x = self.pool(th.nn.functional.relu(self.conv3(x)))
        x = x.reshape(b, -1)
        x = th.nn.functional.relu(self.fc1(x))
        x = th.nn.functional.relu(self.fc2(x))
        x = th.nn.functional.sigmoid(self.fc3(x))
        x = x.view(-1, *self.out_shape)
        return x
        
class Dense(th.nn.Module):
    def __init__(self, in_shape, out_shape, cfg: DictConfig):    
        super().__init__()
        self.out_shape = out_shape
        n_hid_1 = 2048
        n_hid_2 = 2048
        self.fc1 = th.nn.Linear(prod(in_shape), n_hid_1)
        self.fc2 = th.nn.Linear(n_hid_1, n_hid_2)
        out_size_flat = prod(out_shape)
        self.fc3 = th.nn.Linear(n_hid_2, out_size_flat)

    def forward(self, x): 
        b = x.shape[0]
        x = x.view(b, -1)
        x = th.nn.functional.relu(self.fc1(x))
        x = th.nn.functional.relu(self.fc2(x))
        x = th.nn.functional.sigmoid(self.fc3(x))
        x = x.view(-1, *self.out_shape)
        return x


class DenseDeconv(th.nn.Module):
    def __init__(self, in_shape, out_shape, cfg: DictConfig):    
        super().__init__()
        self.fc1 = th.nn.Linear(prod(in_shape), 256)
        self.fc2 = th.nn.Linear(256, 84)
        self.fc3 = th.nn.Linear(84, 256)
        self.deconv1 = th.nn.ConvTranspose3d(4, 16, 5, padding=1)
        self.deconv2 = th.nn.ConvTranspose3d(16, 8, 5, padding=1)
        self.deconv3 = th.nn.ConvTranspose3d(8, 1, 5, padding=1)
        self.upsample = th.nn.Upsample(scale_factor=2, mode="nearest")
    
    def forward(self, x): 
        b = x.shape[0]
        x = x.view(b, -1)
        x = th.nn.functional.relu(self.fc1(x))
        x = th.nn.functional.relu(self.fc2(x))
        x = th.nn.functional.sigmoid(self.fc3(x))
        x = x.view(-1, 4, 4, 4, 4)
        x = th.nn.functional.relu(self.deconv1(x))
        # x = self.upsample(x)
        x = th.nn.functional.relu(self.deconv2(x))
        # x = self.upsample(x)
        x = th.nn.functional.relu(self.deconv3(x))
        return x[:, 0]  # HACK for binary outputs only