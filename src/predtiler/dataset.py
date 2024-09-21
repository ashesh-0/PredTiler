
from predtiler.tile_manager import TileIndexManager, TilingMode

# class TilingDataset:
#     def __init_subclass__(cls, parent_class=None, tile_manager=None, **kwargs):
#         super().__init_subclass__(**kwargs)
#         assert tile_manager is not None, 'tile_manager must be provided'
#         cls.tile_manager = tile_manager
#         if parent_class is not None:
#             has_callable_method = callable(getattr(parent_class, 'patch_location', None))
#             assert has_callable_method, f'{parent_class.__name__} must have a callable method with following signature: def patch_location(self, index)'
#             cls.__bases__ = (parent_class,) + cls.__bases__

#     def __len__(self):
#         return self.tile_manager.total_grid_count()

#     def patch_location(self, index):
#         print('Calling patch_location')
#         patch_loc_list = self.tile_manager.get_patch_location_from_dataset_idx(index)
#         return patch_loc_list
    

# def get_tiling_dataset(dataset_class, tile_manager) -> type:
#     class CorrespondingTilingDataset(TilingDataset, parent_class=dataset_class, tile_manager=tile_manager):
#         pass
    
#     return CorrespondingTilingDataset

def get_tiling_dataset(dataset_class, tile_manager) -> type:
    has_callable_method = callable(getattr(dataset_class, 'patch_location', None))
    assert has_callable_method, f'{dataset_class.__name__} must have a callable method with following signature: def patch_location(self, index)'

    class TilingDataset(dataset_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tile_manager = tile_manager

        def __len__(self):
            return self.tile_manager.total_grid_count()

        def patch_location(self, index):
            patch_loc_list = self.tile_manager.get_patch_location_from_dataset_idx(index)
            return patch_loc_list
    
    return TilingDataset
    



def get_tile_manager(data_shape, tile_shape, patch_shape, tiling_mode=TilingMode.ShiftBoundary):
    return TileIndexManager(data_shape, tile_shape, patch_shape, tiling_mode)


