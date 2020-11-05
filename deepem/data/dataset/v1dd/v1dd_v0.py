from deepem.data.dataset.v1dd import basil
from deepem.data.dataset.v1dd import minnie
from deepem.data.dataset.v1dd import v1dd_dense
from deepem.data.dataset.v1dd import v1dd_sparse


def load_data(*args, **kwargs):
    data = {}
    data.update(basil.load_data(*args, **kwargs))
    data.update(minnie.load_data(*args, **kwargs))
    data.update(v1dd_dense.load_data(*args, **kwargs))
    data.update(v1dd_sparse.load_data(*args, **kwargs))
    return data
