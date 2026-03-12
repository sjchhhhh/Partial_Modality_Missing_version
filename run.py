import os
import gc
import time
import random
import logging
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pynvml
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool

from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.load_data import MMDataLoader
from config.config_regression import ConfigRegression

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run(args, dataloader):
    is_distributed = dist.is_initialized()
    local_rank = args.local_rank if is_distributed else 0
    is_main_process = not is_distributed or dist.get_rank() == 0

    if is_main_process and not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    args.model_save_path = os.path.join(
        args.model_save_dir,
        f'{args.modelName}-{args.datasetName}-{args.train_mode}.pth'
    )

    # device
    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        # fallback: single / DataParallel 模式
        if len(args.gpu_ids) == 0 and torch.cuda.is_available():
            # auto-select the most-free GPU among all available
            n_gpu = torch.cuda.device_count()
            dst_gpu_id, min_mem_used = 0, 1e16
            try:
                pynvml.nvmlInit()
                for g_id in range(n_gpu):
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(g_id)
                        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        if meminfo.used < min_mem_used:
                            min_mem_used = meminfo.used
                            dst_gpu_id = g_id
                    except Exception:
                        pass
                pynvml.nvmlShutdown()
            except Exception:
                dst_gpu_id = 0
            print(f'Auto-select GPU: {dst_gpu_id} (memory used: {min_mem_used})')
            logger.info(f'Auto-select GPU: {dst_gpu_id}')
            args.gpu_ids = [dst_gpu_id]
        using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
        device = torch.device('cuda:%d' % int(args.gpu_ids[0]) if using_cuda else 'cpu')

    args.device = device
    if is_main_process:
        if is_distributed:
            logger.info(f"DDP training on {dist.get_world_size()} GPUs, local_rank={local_rank}")
        else:
            logger.info("Let's use %d GPUs!" % max(len(args.gpu_ids), 1 if torch.cuda.is_available() else 0))

    # load models
    model = AMIO(args).to(device)

    def count_parameters(m):
        answer = 0
        for p in m.parameters():
            if p.requires_grad:
                answer += p.numel()
        return answer

    if is_main_process:
        logger.info(f'The model has {count_parameters(model)} trainable parameters')

    # wrap with DDP when distributed
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    atio = ATIO().getTrain(args)
    # do train
    atio.do_train(model, dataloader)

    # load pretrained model（重新构建非 DDP 模型做测试）
    if is_distributed:
        dist.barrier()
    assert os.path.exists(args.model_save_path)
    test_model = AMIO(args).to(device)
    state = torch.load(args.model_save_path, map_location=device)
    if isinstance(state, dict) and state and next(iter(state.keys())).startswith('module.'):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    test_model.load_state_dict(state, strict=True)
    results = atio.do_test(test_model, dataloader['test'], mode="TEST")

    del model, test_model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(5)

    return results


def run_normal(args):
    args.res_save_dir = os.path.join(args.res_save_dir, 'normals')
    init_args = args
    model_results = []
    seeds = args.seeds

    missing_rate = 0.0
    args = init_args
    # load config
    config = ConfigRegression(args)
    args = config.get_config()
    # load data
    is_distributed = dist.is_initialized()
    dataloader = MMDataLoader(args, distributed=is_distributed)
    # run results
    for i, seed in enumerate(seeds):
        if i == 0 and args.data_missing:
            missing_rate = str(args.missing_rate[0]) if args.diff_missing is None else '-'.join(
                [str(round(m, 1)) for m in args.diff_missing]
            )
        setup_seed(seed)
        args.seed = seed
        logger.info('Start running %s...' % (args.modelName))
        logger.info(args)
        # runnning
        args.cur_time = i + 1
        test_results = run(args, dataloader)
        # restore results（仅主进程汇总）
        if not dist.is_initialized() or dist.get_rank() == 0:
            model_results.append(test_results)
            logger.info(f"==> Test results of seed {seed}:\n{test_results}")

    # 只有主进程负责写结果文件
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    criterions = list(model_results[0].keys())
    # load other results
    save_path = os.path.join(
        args.res_save_dir,
        f'{args.datasetName}-{args.train_mode}-{missing_rate}.csv'
    )
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Model"] + criterions)
    # save results
    res = [args.modelName]
    for c in criterions:
        values = [r[c] for r in model_results]
        mean = round(np.mean(values) * 100, 2)
        std = round(np.std(values) * 100, 2)
        res.append((mean, std))
    df.loc[len(df)] = res
    df.to_csv(save_path, index=None)
    logger.info('Results are added to %s...' % (save_path))

    # detailed results
    import datetime
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(
        args.res_save_dir,
        f'{args.datasetName}-{args.train_mode}-{missing_rate}-detail.csv'
    )
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Time", "Model", "Params", "Seed"] + criterions)
    # seed
    for i, seed in enumerate(seeds):
        res = [cur_time, args.modelName, str(args), f'{seed}']
        for c in criterions:
            val = round(model_results[i][c] * 100, 2)
            res.append(val)
        df.loc[len(df)] = res
    # mean
    res = [cur_time, args.modelName, str(args), '<mean/std>']
    for c in criterions:
        values = [r[c] for r in model_results]
        mean = round(np.mean(values) * 100, 2)
        std = round(np.std(values) * 100, 2)
        res.append((mean, std))
    df.loc[len(df)] = res
    # max
    res = [cur_time, args.modelName, str(args), '<max/seed>']
    for c in criterions:
        values = [r[c] for r in model_results]
        max_val = round(np.max(values) * 100, 2)
        max_seed = seeds[np.argmax(values)]
        res.append((max_val, max_seed))
    df.loc[len(df)] = res
    # min
    res = [cur_time, args.modelName, str(args), '<min/seed>']
    for c in criterions:
        values = [r[c] for r in model_results]
        min_val = round(np.min(values) * 100, 2)
        min_seed = seeds[np.argmin(values)]
        res.append((min_val, min_seed))
    df.loc[len(df)] = res
    df.to_csv(save_path, index=None)
    logger.info('Detailed results are added to %s...' % (save_path))


def set_log(args):
    res_dir = os.path.join(args.res_save_dir, 'normals')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    suffix = '-mr' + '_'.join([str(mr) for mr in args.missing_rate]) if args.diff_missing is not None else f'-mr{args.missing_rate[0]}'
    exp_suffix = f'-{args.exp_name}' if args.exp_name else ''
    log_file_path = os.path.join(res_dir, f'{args.modelName}-{args.datasetName}{suffix}{exp_suffix}.log')
    # set logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    for ph in list(logger.handlers):
        logger.removeHandler(ph)
    # add FileHandler to log file
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    # add StreamHandler to terminal outputs
    formatter_stream = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter_stream)
    logger.addHandler(ch)
    return logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--need_task_scheduling', type=bool, default=False,
                        help='use the task scheduling module.')
    parser.add_argument('--is_tune', type=bool, default=False,
                        help='tune parameters ?')
    parser.add_argument('--train_mode', type=str, default="regression",
                        help='regression')
    parser.add_argument('--modelName', type=str, default='emt-dlfr',
                        help='support emt-dlfr/mult/tfr_net')
    parser.add_argument('--datasetName', type=str, default='mosi',
                        help='support mosi/mosei/sims')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num workers of loading data')
    parser.add_argument('--model_save_dir', type=str, default='results/models',
                        help='path to save results.')
    parser.add_argument('--res_save_dir', type=str, default='results/results',
                        help='path to save results.')
    parser.add_argument('--gpu_ids', type=int, nargs='*', default=[],
                        help='GPUs to use in non-distributed mode. If none, auto-select the most-free GPU.')
    parser.add_argument('--missing', type=float, default=0.4)
    # more
    parser.add_argument('--seed', type=int, default=2222, help='start seed')
    parser.add_argument('--num_seeds', type=int, default=None, help='number of total seeds')
    parser.add_argument('--exp_name', type=str, default='', help='experiment name')
    parser.add_argument('--diff_missing', type=float, nargs='+', default=None, help='different missing rates for text, audio, and video')
    parser.add_argument('--KeyEval', type=str, default='Loss', help='the evaluation metric used to select the best model')
    # for sims
    parser.add_argument('--use_normalized_data', action='store_true', help='use normalized audio & video data (for now, only for sims)')
    # for distributed training
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training (set by torchrun)')

    return parser.parse_args()


def init_distributed():
    """Initialize distributed training if launched by torchrun."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return local_rank, rank, world_size
    return 0, 0, 1


if __name__ == '__main__':
    args = parse_args()
    # initialize distributed training (if any)
    local_rank, rank, world_size = init_distributed()
    args.local_rank = local_rank
    is_main_process = rank == 0

    args.missing_rate = tuple([args.missing, args.missing, args.missing]) if args.diff_missing is None else args.diff_missing
    global logger
    if is_main_process:
        logger = set_log(args)
    else:
        # 非主进程仅保留最少量日志，避免刷屏
        logger = logging.getLogger()
        logger.setLevel(logging.WARNING)

    args.seeds = [111, 1111, 11111] if args.num_seeds is None else list(range(args.seed, args.seed + args.num_seeds))
    args.num_seeds = len(args.seeds)
    run_normal(args)

    # cleanup distributed
    if dist.is_initialized():
        dist.destroy_process_group()
