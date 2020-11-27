import argparse
import json
import multiprocessing
import subprocess
from taskqueue import LocalTaskQueue


cmd = "deepem/test/run.py"


def single_run(opt, gpu_id):
    args = opt.params.format(**b, iter=opt.iter, gpu_id=gpu_id)
    if opt.dry_run:
        print(["python", cmd, args])
    else:
        subprocess.run(["python", cmd] + args.split())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--spec',
        type=str,
        required=True,
        help='local path to the JSON batch spec')
    parser.add_argument(
        '--iter', 
        type=int, 
        default=0)
    parser.add_argument(
        '--gpu_ids', 
        type=str, 
        default=['0'], 
        nargs='+')
    parser.add_argument(
        '--params', 
        type=str,
        required=True,
        help='inference parameters')
    parser.add_argument(
        '--dry_run',
        action='store_true')
    opt = parser.parse_args()

    # JSON batch spec
    with open(opt.spec, "r") as f:
        batch = json.load(f)

    # Run inference
    gpu_ids = [opt.gpu_ids[i % len(opt.gpu_ids)] for i in range(len(batch))]
    tasks = (single_run(opt, gpu_id=i) for i in gpu_ids)
    with LocalTaskQueue(parallel=len(opt.gpu_ids)) as tq:
        tq.insert_all(tasks)

    # Run inference
    # pool = multiprocessing.Pool(len(opt.gpu_ids))
    # pool.map_async()

    # for i, b in enumerate(batch):
    #     print(f"Batch run {i+1}")
    #     args = opt.params.format(**b, iter=opt.iter)
    #     if opt.dry_run:
    #         print(["python", cmd, args])
    #     else:
    #         subprocess.run(["python", cmd] + args.split())
