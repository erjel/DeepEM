from deepem.data.dataset.aibs import basil
from deepem.data.dataset.aibs import minnie
from deepem.data.dataset.aibs import v1dd_dense
from deepem.data.dataset.aibs import human_dense


def load_data(*args, **kwargs):
    data = {}
    data.update(basil.load_data(*args, **kwargs))
    data.update(minnie.load_data(*args, **kwargs))
    data.update(v1dd_dense.load_data(*args, **kwargs))
    data.update(human_dense.load_data(*args, **kwargs))
    return data
