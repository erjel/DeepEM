from __future__ import print_function

import torch
from torch import nn
from torch.nn import functional as F

from deepem.utils import torch_utils


class Model(nn.Module):
    """
    Model wrapper for inference.
    """
    def __init__(self, model, opt):
        super(Model, self).__init__()
        self.model = model
        self.in_spec = dict(opt.in_spec)
        self.pretrain = opt.pretrain
        self.vec2aff = opt.vec2aff

    def forward(self, sample):
        inputs = [sample[k] for k in sorted(self.in_spec)]
        preds = self.model(*inputs)
        outputs = dict()
        for k, x in preds.items():
            if k == 'embedding' and self.vec2aff:
                outputs[k] = torch_utils.vec2aff(x)
            else:
                outputs[k] = F.sigmoid(x)
        return outputs


    def load(self, fpath):
        state_dict = torch.load(fpath)
        if self.pretrain:
            model_dict = self.model.state_dict()
            state_dict = {k:v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(state_dict)
            self.model.load_state_dict(model_dict)
        else:
            self.model.load_state_dict(state_dict)


class OnnxModel(Model):
    def __init__(self, model, opt):
        super(OnnxModel, self).__init__(model, opt)

    def forward(self, x):
        outputs = self.model(x)
        return [F.sigmoid(x) for x in outputs]
