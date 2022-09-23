import argparse
import json
import math
import os
import numpy as np

from deepem.utils.py_utils import vec3, vec3f


class Options(object):
    """
    Test options.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--exp_name', required=True)
        self.parser.add_argument('--chkpt_num', type=int, default=0)
        self.parser.add_argument('--gpu_id', type=str, default='0')

        # CPU inference
        self.parser.add_argument('--cpu', action='store_true')

        # cuDNN auto-tuning
        self.parser.add_argument('--no_autotune', action='store_false')

        # Overwriting
        self.parser.add_argument('--model', default=None)
        self.parser.add_argument('--pretrain', action='store_true')
        self.parser.add_argument('--no_eval', action='store_true')
        self.parser.add_argument('--inputsz', type=vec3, default=None)
        self.parser.add_argument('--outputsz', type=vec3, default=None)
        self.parser.add_argument('--force_crop', type=vec3, default=None)
        self.parser.add_argument('--fov', type=vec3, default=None)
        self.parser.add_argument('--depth', type=int, default=4)
        self.parser.add_argument('--width', type=int, default=None, nargs='+')
        self.parser.add_argument('--group', type=int, default=0)
        self.parser.add_argument('--act', default='ReLU')

        # Multiclass detection
        self.parser.add_argument('--aff',  action='store_true')
        self.parser.add_argument('--long', type=int, default=0)
        self.parser.add_argument('--aff_deprecated', type=int, default=None)
        self.parser.add_argument('--bdr',  action='store_true')
        self.parser.add_argument('--syn',  action='store_true')
        self.parser.add_argument('--psd',  action='store_true')
        self.parser.add_argument('--mit',  action='store_true')
        self.parser.add_argument('--mye',  action='store_true')
        self.parser.add_argument('--mye_thresh', type=float, default=0.5)
        self.parser.add_argument('--blv',  action='store_true')
        self.parser.add_argument('--blv_num_channels', type=int, default=2)
        self.parser.add_argument('--glia',  action='store_true')
        self.parser.add_argument('--sem',  action='store_true')

        # Test-time augmentation
        self.parser.add_argument('--test_aug', type=int, default=None, nargs='+')
        self.parser.add_argument('--test_aug16', action='store_true')
        self.parser.add_argument('--variance', action='store_true')

        # Temperature T for softer softmax
        self.parser.add_argument('--temperature', type=float, default=None)

        # Cloud-volume input
        self.parser.add_argument('--gs_input', default='')
        self.parser.add_argument('--gs_input_mask', default='')
        self.parser.add_argument('--gs_input_norm', type=float, default=None, nargs='+')
        self.parser.add_argument('--in_mip', type=int, default=0)
        self.parser.add_argument('--cache', action='store_true')
        self.parser.add_argument('--coord_mip', type=int, default=0)
        self.parser.add_argument('-b','--begin', type=vec3, default=None)
        self.parser.add_argument('-e','--end', type=vec3, default=None)
        self.parser.add_argument('-c','--center', type=vec3, default=None)
        self.parser.add_argument('-s','--size', type=vec3, default=None)        

        # Cloud-volume output
        self.parser.add_argument('--gs_output', default='')
        self.parser.add_argument('--tags', type=json.loads, default=None)
        self.parser.add_argument('--keywords', default=[], nargs='+')
        self.parser.add_argument('-p','--parallel', type=int, default=16)
        self.parser.add_argument('-d','--downsample', action='store_true')
        self.parser.add_argument('-r','--resolution', type=vec3, default=(4,4,40))
        self.parser.add_argument('-o','--offset', type=vec3, default=None)
        self.parser.add_argument('--chunk_size', type=vec3, default=(64,64,16))

        # Data
        self.parser.add_argument('--data_dir', default="")
        self.parser.add_argument('--data_names', nargs='+')
        self.parser.add_argument('--input_name', default="img.h5")

        # Forward scanning
        self.parser.add_argument('--out_prefix', default='')
        self.parser.add_argument('--out_tag', default='')
        self.parser.add_argument('--overlap', type=vec3f, default=(0.5,0.5,0.5))
        self.parser.add_argument('--stride', type=vec3, default=None)
        self.parser.add_argument('--mirror', type=vec3, default=None)
        self.parser.add_argument('--crop_border', type=vec3, default=None)
        self.parser.add_argument('--crop_center', type=vec3, default=None)
        self.parser.add_argument('--blend', default='bump')
        self.parser.add_argument('--bump', default='zung')  # 'zung'/'wu'/'wu_no_crust'

        # Asymmetric mask
        self.parser.add_argument('--mask_edges', type=vec3, default=[(0,0,1),(0,1,0),(1,0,0)], nargs='+')

        # Benchmark
        self.parser.add_argument('--dummy', action='store_true')
        self.parser.add_argument('--dummy_inputsz', type=int, default=[128,1024,1024], nargs='+')

        # Mixed-precision inference
        self.parser.add_argument('--mixed_precision', action='store_true')

        # Export to ONNX
        self.parser.add_argument('--onnx', action='store_true')

        self.initialized = True        

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        # Device
        opt.device = 'cpu' if opt.cpu else 'cuda'

        # Directories
        if opt.exp_name.split('/')[0] == 'experiments':
            opt.exp_dir = opt.exp_name
        else:
            opt.exp_dir = f"experiments/{opt.exp_name}"
        opt.model_dir = os.path.join(opt.exp_dir, 'models')
        opt.fwd_dir = os.path.join(opt.exp_dir, 'forward')

        # Model spec
        opt.fov = tuple(opt.fov)
        opt.inputsz = opt.fov if opt.inputsz is None else opt.inputsz
        opt.outputsz = opt.fov if opt.outputsz is None else opt.outputsz
        opt.in_spec = dict(input=(1,) + opt.inputsz)
        opt.out_spec = dict()

        # Crop output
        diff = np.array(opt.fov) - np.array(opt.outputsz)
        assert all(diff >= 0)
        if any(diff > 0):
            # opt.cropsz = opt.outputsz
            opt.cropsz = [o/float(f) for f,o in zip(opt.fov,opt.outputsz)]
        else:
            opt.cropsz = None

        if opt.aff:
            opt.out_spec['affinity'] = (3,) + opt.outputsz
        if opt.aff_deprecated:
            opt.out_spec['affinity'] = (opt.aff_deprecated,) + opt.outputsz
        if opt.bdr:
            opt.out_spec['boundary'] = (1,) + opt.outputsz
        if opt.syn:
            opt.out_spec['synapse'] = (1,) + opt.outputsz
        if opt.psd:
            opt.out_spec['synapse'] = (1,) + opt.outputsz
        if opt.mit:
            opt.out_spec['mitochondria'] = (1,) + opt.outputsz
        if opt.mye:
            opt.out_spec['myelin'] = (1,) + opt.outputsz
        if opt.blv:
            opt.out_spec['blood_vessel'] = (opt.blv_num_channels,) + opt.outputsz
        if opt.glia:
            opt.out_spec['glia'] = (1,) + opt.outputsz
        if opt.sem:
            opt.out_spec['soma'] = (1,) + opt.outputsz
            opt.out_spec['axon'] = (1,) + opt.outputsz
            opt.out_spec['dendrite'] = (1,) + opt.outputsz
            opt.out_spec['glia'] = (1,) + opt.outputsz
            opt.out_spec['bvessel'] = (1,) + opt.outputsz
        assert(len(opt.out_spec) > 0)

        # Scan spec
        opt.scan_spec = dict()
        if opt.aff:
            opt.scan_spec['affinity'] = (3,) + opt.outputsz
        if opt.aff_deprecated:
            opt.scan_spec['affinity'] = (3,) + opt.outputsz
        if opt.bdr:
            opt.scan_spec['boundary'] = (1,) + opt.outputsz
        if opt.syn:
            opt.scan_spec['synapse'] = (1,) + opt.outputsz
        if opt.psd:
            opt.scan_spec['synapse'] = (1,) + opt.outputsz
        if opt.mit:
            opt.scan_spec['mitochondria'] = (1,) + opt.outputsz
        if opt.mye:
            opt.scan_spec['myelin'] = (1,) + opt.outputsz
        if opt.blv:
            opt.scan_spec['blood_vessel'] = (opt.blv_num_channels,) + opt.outputsz
        if opt.glia:
            opt.scan_spec['glia'] = (1,) + opt.outputsz
        if opt.sem:
            opt.scan_spec['soma'] = (1,) + opt.outputsz
            opt.scan_spec['axon'] = (1,) + opt.outputsz
            opt.scan_spec['dendrite'] = (1,) + opt.outputsz
            opt.scan_spec['glia'] = (1,) + opt.outputsz
            opt.scan_spec['bvessel'] = (1,) + opt.outputsz

        # Test-time augmentation
        if opt.test_aug16:
            opt.test_aug = list(range(16))

        # Overlap & stride
        if opt.stride is None:
            # infer stride from overlap
            opt.overlap = self.get_overlap(opt.outputsz, opt.overlap)
            opt.stride = tuple(int(f-o) for f,o in zip(opt.outputsz, opt.overlap))
        else:            
            # infer overlap from stride
            opt.overlap = tuple(int(f-s) for f,s in zip(opt.outputsz, opt.stride))
        opt.scan_params = dict(stride=opt.stride, blend=opt.blend)

        # Output tagging
        if opt.tags is not None:
            for k in opt.tags:
                assert k in opt.scan_spec

        args = vars(opt)
        print('------------ Options -------------')
        for k, v in args.items():
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        self.opt = opt
        return self.opt

    def get_overlap(self, fov, overlap):
        assert len(fov) == 3
        assert len(overlap) == 3
        overlap_filter = lambda f,o: math.floor(f*o) if o > 0 and o < 1 else o
        return tuple(int(overlap_filter(f,o)) for f,o in zip(fov,overlap))
