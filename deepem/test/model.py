from __future__ import print_function

import numpy as np

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
        self.cropsz = np.maximum(opt.cropsz, 0)

        # Metric learning
        self.vec_to = opt.vec_to
        self.vec_aff = opt.vec_aff
        self.mean_loss = opt.mean_loss
        self.gamma = 2 * opt.delta_d

    def forward(self, sample):
        inputs = [sample[k] for k in sorted(self.in_spec)]
        preds = self.model(*inputs)
        outputs = dict()
        for k, x in preds.items():
            if k == 'embedding':
                if self.vec_to == 'aff':
                    outputs[k] = torch_utils.vec2aff(x, aff=self.vec_aff,
                                            mean_loss=self.mean_loss,
                                            gamma=self.gamma)
                elif self.vec_to == 'pca':
                    outputs[k] = torch_utils.vec2pca(x)
                else:
                    outputs[k] = x
            else:
                outputs[k] = F.sigmoid(x)

            # Crop outputs.
            if any(self.cropsz):
                ndim = outputs[k].dim()
                cropsz = [0] * ndim
                cropsz[-3:] = self.cropsz
                for cs in cropsz:
                    if cs > 0:
                        slices.append(slice(cs,-cs))
                    else:
                        slices.append(slice(None))
                outputs[k] = outputs[k][slices]

        return outputs


    def load(self, fpath):
        chkpt = torch.load(fpath)
        # Backward compatibility
        state_dict = chkpt['state_dict'] if 'state_dict' in chkpt else chkpt
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
