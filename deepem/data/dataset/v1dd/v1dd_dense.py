import os
import numpy as np

import dataprovider3.emio as emio


# V1DD dense annotation
data_dir = 'v1dd/ground_truth/dense'
data_info = {
    'img': 'img.h5',
    'msk': 'msk.h5',
    'seg': 'seg.h5',
    'mye': 'mye.h5',
    'fld': 'fld.h5',
    'dir': 'mip1/padded_x512_y512_z32',
    'loc': True,
}
data_keys = ['v1dd_dense0{:0>2d}'.format(i+1) for i in range(5)]


def load_data(base_dir, data_ids=None, **kwargs):
    if data_ids is None:
        return {}
    
    base_dir = os.path.expanduser(base_dir)
    base_dir = os.path.join(base_dir, data_dir)

    data = {}
    for data_id in data_ids:
        if data_id == 'v1dd_dense_superset':
            # Superset
            superset = {}
            for dkey in data_keys:
                dpath = os.path.join(base_dir, data_info['dir'], dkey)
                if os.path.exists(dpath):
                    superset[dkey] = load_dataset(dpath, **kwargs)
            data['v1dd_dense_superset'] = superset
        else:
            # Individual dataset
            if data_id in data_keys:
                dpath = os.path.join(base_dir, data_info['dir'], data_id)
                assert os.path.exists(dpath)
                data[data_id] = load_dataset(dpath, **kwargs)
    return data


def load_dataset(dpath, class_keys=[], **kwargs):
    assert len(class_keys) > 0
    dset = {}

    # Image
    fpath = os.path.join(dpath, data_info['img'])
    print(fpath)
    dset['img'] = emio.imread(fpath).astype(np.float32)
    dset['img'] /= 255.0

    # Mask
    fpath = os.path.join(dpath, data_info['msk'])
    print(fpath)
    dset['msk'] = emio.imread(fpath).astype(np.uint8)

    # Segmentation
    fpath = os.path.join(dpath, data_info['seg'])
    print(fpath)
    dset['seg'] = emio.imread(fpath).astype(np.uint32)

    # Myelin (optional)
    if 'mye' in class_keys:
        fpath = os.path.join(dpath, data_info['mye'])
        if os.path.exists(fpath):
            print(fpath)
            dset['mye'] = emio.imread(fpath).astype(np.uint8)
        else:
            assert 'msk' in dset
            dset['mye'] = np.zeros_like(dset['msk'])

    # Fold mask (optional)
    fpath = os.path.join(dpath, data_info['fld'])
    if os.path.exists(fpath):
        print(fpath)
        fld = emio.imread(fpath).astype(np.uint8)
        assert 'msk' in dset
        dset['msk'][fld > 0] = 0

    # Additoinal info
    dset['loc'] = data_info['loc']

    return dset
