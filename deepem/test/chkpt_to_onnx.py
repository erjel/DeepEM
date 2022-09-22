#!/usr/bin/env python
# -*- coding: utf-8 -*-
import imp
import os
import torch
from torch import nn

from deepem.test.option import Options
from deepem.test.model import Model
from deepem.test.utils import load_chkpt


def batchnorm3d_to_instancenorm3d(model):
    conversion_count = 0
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name], num_converted = batchnorm3d_to_instancenorm3d(module)
            conversion_count += num_converted

        if isinstance(module, nn.BatchNorm3d):
            layer_old = module
            layer_new = nn.InstanceNorm3d(
                module.num_features, affine=module.affine, track_running_stats=False
            )
            layer_new.weight = module.weight
            layer_new.bias = module.bias
            model._modules[name] = layer_new
            conversion_count += 1
    print(f"replaced {conversion_count} BatchNorm3d layer to InstanceNorm3d layer.")
    return model, conversion_count


def prepare_model_for_onnx(model):
    model, _ = batchnorm3d_to_instancenorm3d(model)
    return model


def get_output_fn(opt):
    print("LOAD CHECKPOINT: {} iters.".format(opt.chkpt_num))
    return os.path.join(opt.model_dir, "model{}.onnx".format(opt.chkpt_num))


def load_model_no_cuda(opt):
    """Similar to deepem.test.utils.load_model, but uses cuda only if availabe
    """
    # Create a model.
    mod = imp.load_source("model", opt.model)
    model = Model(mod.create_model(opt), opt)

    # Load from a checkpoint, if any.
    if opt.chkpt_num > 0:
        model = load_chkpt(model, opt.model_dir, opt.chkpt_num)

    model = model.train()
    if torch.cuda.is_available():
        model.cuda()
    return model


if __name__ == "__main__":
    # Options
    # # first make it a cpu model, the model will be loaded to GPU in chunkflow later if there is a GPU device.
    # d['gpu_ids'] = []
    # d['model'] = "~/DeepEM/deepem/models/rsunet_deprecated.py"
    # d['width'] = [16,32,64,128]
    # d['group'] = 0
    # d['act'] = 'ReLU'
    # d['force_crop'] = None
    # d['temperature'] = None
    # d['blend'] = 'bump'
    # d['chkpt_num'] = 200000
    # d['model_dir'] = "~/convert"
    # d['no_eval'] = True
    # d['in_spec'] = {'input': (1,20,256,256)}
    # d['out_spec'] = {'affinity': (3,16,192,192)}
    # d['scan_spec'] = {'affinity': (3,16,192,192)}
    # d['cropsz'] = None
    # d['pretrain'] = True
    # d['precomputed'] = False
    # d['overlap'] = (0,0,0)
    # d['bump'] = None
    opt = Options().parse()
    opt.gpu_id = []
    opt.bump = None

    torch_model = load_model_no_cuda(opt)
    # replace batchnorm for onnx
    torch_model = prepare_model_for_onnx(torch_model)
    output_fn = get_output_fn(opt)
    input_data = {"input": torch.rand(*((1,) + opt.in_spec["input"]))}
    torch.onnx.export(
        torch_model,
        input_data,
        output_fn,
        verbose=False,
        export_params=True,
        opset_version=10,
        input_names=["input"],
        output_names=[*opt.out_spec.keys()],
    )
    print(f"Relative ONNX filepath: {output_fn}")
