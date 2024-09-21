from unittest.mock import Mock
import numpy as np
from predtiler.dataset import get_tiling_dataset, get_tile_manager
# import sys; sys.path.append('../')
from predtiler.tile_stitcher import stitch_predictions
from predtiler.tile_manager import TilingMode
from functools import partial

def get_data_3D(n=5,Z=9, H=512,W=512,C=2):
    data = np.arange(n*Z*H*W*C).reshape(n,Z,H,W,C)
    return data

def get_data_2D(n=5,H=512,W=512,C=2):
    data = np.arange(n*H*W*C).reshape(n,H,W,C)
    return data

class DummDataset:
    def __init__(self, datatype ='2D', patch_size=64, z_patch_size=5) -> None:
        assert datatype in ['2D', '3D'], 'datatype must be either 2D or 3D'
        self.datatype = datatype
        self.z_patch_size = z_patch_size
        self.patch_size = patch_size
        if datatype == '2D':
            self.data = get_data_2D()
        elif datatype == '3D':
            self.data = get_data_3D()

    def patch_location(self, index):
        if self.datatype == '2D':
            n_idx = np.random.randint(0,len(self.data))
            h = np.random.randint(0, self.data.shape[1]-self.patch_size)
            w = np.random.randint(0, self.data.shape[2]-self.patch_size)
            return (n_idx, h, w)
        elif self.datatype == '3D':
            n_idx = np.random.randint(0,len(self.data))
            z = np.random.randint(0, self.data.shape[1]-self.z_patch_size)
            h = np.random.randint(0, self.data.shape[2]-self.patch_size)
            w = np.random.randint(0, self.data.shape[3]-self.patch_size)
            return (n_idx, z, h, w)
    
    def __len__(self):
        return len(self.data) * (self.data.shape[-2]//self.patch_size) * (self.data.shape[-3]//self.patch_size)

    def __getitem__(self, index):
        if self.datatype == '2D':
            n_idx, h, w = self.patch_location(index)
            return self.data[n_idx, h:h+self.patch_size, w:w+self.patch_size].transpose(2,0,1)
        elif self.datatype == '3D':
            n_idx, z, h, w = self.patch_location(index)
            return self.data[n_idx, z:z+self.z_patch_size, h:h+self.patch_size, w:w+self.patch_size].transpose(3,0,1,2)


def test_stich_prediction_2D():
    data_type = '2D'
    data_fn = get_data_2D
    patch_size = 256
    tile_size = 128
    data = data_fn()
    manager = get_tile_manager(data_shape=data.shape[:-1], tile_shape=(1,tile_size,tile_size), 
                               patch_shape=(1,patch_size,patch_size))
    
    dset_class = get_tiling_dataset(DummDataset, manager)
    dset = dset_class(data_type, patch_size)
    
    predictions = []
    for i in range(len(dset)):
        predictions.append(dset[i])
    
    predictions = np.stack(predictions)
    stitched_pred = stitch_predictions(predictions, dset.tile_manager)
    assert (stitched_pred== data).all()



def test_stich_prediction_3D():
    data_type = '3D'
    data_fn = get_data_3D
    patch_size = 256
    tile_size = 128
    data = data_fn()
    z_patch_size = 5
    z_tile_size = 3
    manager = get_tile_manager(data_shape=data.shape[:-1], tile_shape=(1,z_tile_size, tile_size,tile_size), 
                               patch_shape=(1,z_patch_size, patch_size,patch_size))
    
    dset_class = get_tiling_dataset(DummDataset, manager)
    dset = dset_class(data_type, patch_size)
    
    predictions = []
    for i in range(len(dset)):
        predictions.append(dset[i])
    
    predictions = np.stack(predictions)
    stitched_pred = stitch_predictions(predictions, dset.tile_manager)
    assert (stitched_pred== data).all()
