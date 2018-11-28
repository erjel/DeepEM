from __future__ import print_function
import argparse
import os

from deepem.utils.py_utils import vec3, vec3f


class Options(object):
    """
    Test options.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--log_name', required=True)
        self.parser.add_argument('--chkpt_num', type=int, default=0)
        self.parser.add_argument('--gpu_id', type=str, default='0')

        # cuDNN auto-tuning
        self.parser.add_argument('--no_autotune', action='store_false')

        # Overwriting
        self.parser.add_argument('--model', default=None)
        self.parser.add_argument('--pretrain', action='store_true')
        self.parser.add_argument('--no_eval', action='store_true')
        self.parser.add_argument('--inputsz', type=vec3, default=None)
        self.parser.add_argument('--outputsz', type=vec3, default=None)
        self.parser.add_argument('--fov', type=vec3, default=None)
        self.parser.add_argument('--depth', type=int, default=4)
        self.parser.add_argument('--width', type=int, default=None, nargs='+')

        # Multiclass detection
        self.parser.add_argument('--aff',  type=int, default=0)
        self.parser.add_argument('--long', type=int, default=0)
        self.parser.add_argument('--bdr',  type=int, default=0)
        self.parser.add_argument('--psd',  type=int, default=0)
        self.parser.add_argument('--mit',  type=int, default=0)
        self.parser.add_argument('--mye',  type=int, default=0)
        self.parser.add_argument('--bld',  type=int, default=0)

        # Test-time augmentation
        self.parser.add_argument('--test_aug', type=int, default=None, nargs='+')

        # Cloud-volume input
        self.parser.add_argument('--gs_input', default='')
        self.parser.add_argument('--in_mip', type=int, default=0)
        self.parser.add_argument('--cache', action='store_true')
        self.parser.add_argument('-b','--begin', type=vec3, default=None)
        self.parser.add_argument('-e','--end', type=vec3, default=None)
        self.parser.add_argument('-c','--center', type=vec3, default=None)
        self.parser.add_argument('-s','--size', type=vec3, default=None)

        # Cloud-volume output
        self.parser.add_argument('--gs_output', default='')
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
        self.parser.add_argument('--mirror', type=vec3, default=None)
        self.parser.add_argument('--crop_border', type=vec3, default=None)
        self.parser.add_argument('--crop_center', type=vec3, default=None)
        self.parser.add_argument('--blend', default='bump')

        # Benchmark
        self.parser.add_argument('--dummy', action='store_true')
        self.parser.add_argument('--dummy_inputsz', type=int, default=[128,1024,1024], nargs='+')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        # Directories
        if opt.exp_name.split('/')[0] == 'experiments':
            opt.exp_dir = opt.exp_name
        else:
            opt.exp_dir = 'experiments/{}'.format(opt.exp_name)
        opt.model_dir = os.path.join(opt.exp_dir, 'models')
        opt.fwd_dir = os.path.join(opt.exp_dir, 'forward')

        # Model spec
        opt.fov = tuple(opt.fov)
        opt.inputsz = opt.fov if opt.inputsz is None else opt.inputsz
        opt.outputsz = opt.fov if opt.outputsz is None else opt.outputsz
        opt.cropsz = tuple((i-o)//2 for i, o in zip(opt.inputsz, opt.outputsz))
        opt.in_spec = dict(input=(1,) + opt.inputsz)
        opt.out_spec = dict()
        if opt.vec > 0:
            opt.out_spec['embedding'] = (opt.vec,) + opt.outputsz
        if opt.aff > 0:
            opt.out_spec['affinity'] = (opt.aff,) + opt.outputsz
        if opt.bdr:
            opt.out_spec['boundary'] = (1,) + opt.outputsz
        if opt.psd:
            opt.out_spec['synapse'] = (1,) + opt.outputsz
        if opt.mit:
            opt.out_spec['mitochondria'] = (1,) + opt.outputsz
        if opt.mye:
            opt.out_spec['myelin'] = (1,) + opt.outputsz
        if opt.bld:
            opt.out_spec['bloodvessel'] = (1,) + opt.outputsz
        if opt.som:
            opt.out_spec['soma'] = (1,) + opt.outputsz
        assert(len(opt.out_spec) > 0)

        # Scan spec
        opt.scan_spec = dict()
        if opt.vec > 0:
            dim = 3 if opt.vec_to else opt.vec
            opt.scan_spec['embedding'] = (dim,) + opt.outputsz
        if opt.aff > 0:
            opt.scan_spec['affinity'] = (3,) + opt.outputsz
        if opt.bdr:
            opt.scan_spec['boundary'] = (1,) + opt.outputsz
        if opt.psd:
            opt.scan_spec['synapse'] = (1,) + opt.outputsz
        if opt.mit:
            opt.scan_spec['mitochondria'] = (1,) + opt.outputsz
        if opt.mye:
            opt.scan_spec['myelin'] = (1,) + opt.outputsz
        if opt.bld:
            opt.scan_spec['bloodvessel'] = (1,) + opt.outputsz
        if opt.som:
            opt.scan_spec['soma'] = (1,) + opt.outputsz
        stride = self.get_stride(opt.outputsz, opt.overlap)
        opt.scan_params = dict(stride=stride, blend=opt.blend)

        args = vars(opt)
        print('------------ Options -------------')
        for k, v in args.items():
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        self.opt = opt
        return self.opt

    def get_stride(self, fov, overlap):
        assert(len(fov) == 3)
        assert(len(overlap) == 3)
        inverse = lambda f,o: float(1-o) if o>0 and o<1 else int(f-o)
        return tuple(inverse(f,o) for f,o in zip(fov,overlap))
