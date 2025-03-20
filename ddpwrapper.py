import argparse, importlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import torch.distributed as dist

from utils.utils import save_arguments_to_log_file

def setup_dist(local_rank):
    if dist.is_initialized():
        return
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group('nccl', init_method='env://')


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str, help="module name", default="attack")
    parser.add_argument("--local_rank", type=int, nargs="?", help="for ddp", default=0)
    args, unknown = parser.parse_known_args()
    attack_api = importlib.import_module(args.module, package=None)

    attack_parser = attack_api.get_parser()
    args, unknown = attack_parser.parse_known_args()

    local_rank = os.environ['LOCAL_RANK']
    setup_dist(int(local_rank))
    torch.backends.cudnn.benchmark = True
    rank, gpu_num = get_dist_info()

    args.save_path = f"{args.save_path}/total_iter{args.total_iter}/step_size{args.step_size}/noise_clamp{args.noise_clamp}"
    os.makedirs(args.save_path, exist_ok=True)
    save_arguments_to_log_file(args, f"{args.save_path}/arguments_log.txt")
    
    attack_api.attack(args, gpu_num, rank)