import numpy as np
import os

import dataprovider3.emio as emio


# Minnie dataset
data_dir = 'minnie/ground_truth'
minnie_dirs = ['mip1/padded_x512_y512_z0', 'mip1/padded_x512_y512_z32']
minnie1_keys = ['minnie{:0>2d}'.format(i+1) for i in range(15)]
minnie2_keys = ['minnie{:0>2d}'.format(i+1) for i in range(15,27)]
minnie_keys = minnie1_keys + minnie2_keys
minnie_info = {
    'img': 'img.h5',
    'seg': 'seg.h5',
    'msk': 'msk.h5',
    'mye': 'mye.h5',
    'fld': 'fld.h5',
    'blv': 'blv.h5',
    'loc': True,
}


def load_data(base_dir, data_ids=None, **kwargs):
    if data_ids is None:
        return {}
    
    base_dir = os.path.expanduser(base_dir)
    base_dir = os.path.join(base_dir, data_dir)

    data = {}
    for data_id in data_ids:
        if data_id == 'minnie':
            # Superset
            superset = {}
            for dkey in minnie_keys:
                minnie_dir = minnie_dirs[int(dkey in minnie2_keys)]
                dpath = os.path.join(base_dir, minnie_dir)
                if os.path.exists(dpath):
                    superset[dkey] = load_dataset(dpath, tag=dkey, **kwargs)
            data['minnie_superset'] = superset
        else:
            # Individual dataset
            if data_id in minnie_keys:
                minnie_dir = minnie_dirs[int(data_id in minnie2_keys)]
                dpath = os.path.join(base_dir, minnie_dir)
                assert os.path.exists(dpath)
                data[data_id] = load_dataset(dpath, tag=data_id, **kwargs)
    return data


def load_dataset(dpath, tag, class_keys=[], **kwargs):
    assert len(class_keys) > 0
    dset = {}
    dpath = os.path.join(dpath, tag)

    # Image
    fpath = os.path.join(dpath, minnie_info['img'])
    print(fpath)
    dset['img'] = emio.imread(fpath).astype(np.float32)
    dset['img'] /= 255.0

    # Mask
    fpath = os.path.join(dpath, minnie_info['msk'])
    print(fpath)
    dset['msk'] = emio.imread(fpath).astype(np.uint8)

    # Segmentation
    if 'aff' in class_keys or 'long' in class_keys:
        fpath = os.path.join(dpath, minnie_info['seg'])
        print(fpath)
        dset['seg'] = emio.imread(fpath).astype(np.uint32)

    # Myelin (optional)
    if 'mye' in class_keys:
        fpath = os.path.join(dpath, info['mye'])
        if os.path.exists(fpath):
            print(fpath)
            dset['mye'] = emio.imread(fpath).astype(np.uint8)
        else:
            assert 'msk' in dset
            dset['mye'] = np.zeros_like(dset['msk'])

    # Blood vessel (optional)
    if 'blv' in class_keys:
        fpath = os.path.join(dpath, info['blv'])
        if os.path.exists(fpath):
            print(fpath)
            dset['blv'] = emio.imread(fpath).astype(np.uint8)
        else:
            assert 'msk' in dset
            dset['blv'] = np.zeros_like(dset['msk'])

    # Fold mask (optional)
    fpath = os.path.join(dpath, info['fld'])
    if os.path.exists(fpath):
        print(fpath)
        fld = emio.imread(fpath).astype(np.uint8)
        assert 'msk' in dset
        dset['msk'][fld > 0] = 0

    # Additoinal info
    dset['loc'] = minnie_info['loc']

    return dset
