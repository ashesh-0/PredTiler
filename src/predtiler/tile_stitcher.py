from typing import List

import numpy as np

from predtiler.tile_manager import TilingMode


def stitch_predictions(predictions:np.ndarray, manager):
    """
    Args:
        predictions: N*C*H*W or N*C*D*H*W numpy array where N is the number of datasets, C is the number of channels, H is the height, W is the width, D is the depth.
        manager:
    """
    
    mng = manager
    shape = list(mng.data_shape)
    shape.append(predictions.shape[1])
    print(shape)

    output = np.zeros(shape, dtype=predictions.dtype)
    for dset_idx in range(predictions.shape[0]):
        # grid start, grid end
        gs = np.array(mng.get_location_from_dataset_idx(dset_idx), dtype=int)
        ge = gs + mng.grid_shape

        # patch start, patch end
        ps = gs - mng.patch_offset()
        pe = ps + mng.patch_shape

        # valid grid start, valid grid end
        vgs = np.array([max(0,x) for x in gs], dtype=int)
        vge = np.array([min(x,y) for x,y in zip(ge, mng.data_shape)], dtype=int)
        assert np.all(vgs ==gs)
        assert np.all(vge ==ge)
        
        if mng.tiling_mode == TilingMode.ShiftBoundary:
            for dim in range(len(vgs)):
                if ps[dim] == 0:
                    vgs[dim] = 0
                if pe[dim] == mng.data_shape[dim]:
                    vge[dim]= mng.data_shape[dim]

        # relative start, relative end. This will be used on pred_tiled
        rs = vgs - ps
        re = rs + ( vge - vgs)
        
        for ch_idx in range(predictions.shape[1]):
            if len(output.shape) == 4:
                # channel dimension is the last one.
                output[vgs[0]:vge[0],
                    vgs[1]:vge[1],
                    vgs[2]:vge[2],
                    ch_idx] = predictions[dset_idx][ch_idx,rs[1]:re[1], rs[2]:re[2]]
            elif len(output.shape) == 5:
                # channel dimension is the last one.
                assert vge[0] - vgs[0] == 1, 'Only one frame is supported'
                output[vgs[0],
                    vgs[1]:vge[1],
                    vgs[2]:vge[2],
                    vgs[3]:vge[3],
                    ch_idx] = predictions[dset_idx][ch_idx, rs[1]:re[1], rs[2]:re[2], rs[3]:re[3]]
            else:
                raise ValueError(f'Unsupported shape {output.shape}')
            
    return output
