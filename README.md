A lean wrapper around your dataset class to enable tiled prediction. 

[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/ashesh-0/PredTiler/blob/main/LICENSE)
[![CI](https://github.com/ashesh-0/PredTiler/actions/workflows/ci.yml/badge.svg)](https://github.com/ashesh-0/PredTiler/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ashesh-0/PredTiler/graph/badge.svg?token=M655MOS7EL)](https://codecov.io/gh/ashesh-0/PredTiler)

## Objective
This package subclasses the dataset class you use to train your network.
With PredTiler, you can use your dataset class as is, and PredTiler will take care of the tiling logic for you. 
It will automatically generate patches in such a way that they can be tiled with the overlap of `(patch_size - tile_size)//2`. 
We also provide a function to stitch the tiles back together to get the final prediction. 

In case you are facing issues, feel free to raise an issue and I will be happy to help you out ! 
In future, I plan to add detailed instructions for:
1. multi-channel data
2. 3D data
3. Data being a list of numpy arrays, each poissibly having different shapes.

## Installation

```bash
pip install predtiler
```

## Usage
To work with PredTiler, the only requirement is that your dataset class must have a **patch_location(self, index)** method that returns the location of the patch at the given index. 
Your dataset class should only use the location information returned by this method to return the patch. 
PredTiler will override this method to return the location of the patches needed for tiled prediction. 

Note that your dataset class could be arbitrarily complex (augmentations, returning multiple patches, working with 3D data, etc.). The only requirement is that it should use the crop present at the location returned by **patch_location** method. Below is an example of a simple dataset class that can be used with PredTiler.

```python
class YourDataset:
    def __init__(self, data_path, patch_size=64) -> None:
        self.patch_size = patch_size
        self.data = load_data(data_path) # shape: (N, H, W, C)

    def patch_location(self, index:int)-> Tuple[int, int, int]:
        # it just ignores the index and returns a random location
        n_idx = np.random.randint(0,len(self.data))
        h = np.random.randint(0, self.data.shape[1]-self.patch_size)
        w = np.random.randint(0, self.data.shape[2]-self.patch_size)
        return (n_idx, h, w)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        n_idx, h, w = self.patch_location(index)
        # return the patch at the location (patch_size, patch_size)
        return self.data[n_idx, h:h+self.patch_size, w:w+self.patch_size]
```

## Getting overlapping patches needed for tiled prediction
To use PredTiler, we need to get a new class that wraps around your dataset class.
For this we also need a tile manager that will manage the tiles.

```python

from predtiler.dataset import get_tiling_dataset, get_tile_manager
patch_size = 256
tile_size = 128
data_shape = (10, 2048, 2048) # size of the data you are working with
manager = get_tile_manager(data_shape=data_shape, tile_shape=(1,tile_size,tile_size), 
                               patch_shape=(1,patch_size,patch_size))
    
dset_class = get_tiling_dataset(YourDataset, manager)
```

At this point, you can use the `dset_class` as you would use `YourDataset` class. 

```python
data_path = ... # path to your data
dset = dset_class(data_path, patch_size=patch_size)
```

## Stitching the predictions
The benefit of using PredTiler is that it will automatically generate the patches in such a way that they can be tiled with the overlap of `(patch_size - tile_size)//2`. This allows you to use your dataset class as is, without worrying about the tiling logic.

```python
model = ... # your model
predictions = []
for i in range(len(dset)):
    inp = dset[i]
    inp = torch.Tensor(inp)[None,None]
    pred = model(inp)
    predictions.append(pred[0].numpy())

predictions = np.stack(predictions) # shape: (number_of_patches, C, patch_size, patch_size)
stitched_pred = stitch_predictions(predictions, dset.tile_manager)
```


