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

    if not os.path.exists(args.model_save_dir) and is_main_process:
        os.makedirs(args.model_save_dir)
    if getattr(args, 'unified_missing', False):
        suffix = f'-unified-{args.seed}'
    else:
        suffix = f'-mr{args.missing_rate[0]:.1f}-{args.seed}' if args.save_model else ''
    args.model_save_path = os.path.join(
        args.model_save_dir,
        f'{args.modelName}-{args.datasetName}-{args.train_mode}{suffix}.pth'
    )

    # device
    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        # 单进程回退逻辑：自动选择空闲 GPU 或 CPU
        if len(args.gpu_ids) == 0 and torch.cuda.is_available():
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
        if using_cuda:
            device = torch.device('cuda:%d' % int(args.gpu_ids[0]))
            print(f"[Device] Using GPU: cuda:{args.gpu_ids[0]} ({torch.cuda.get_device_name(args.gpu_ids[0]) if torch.cuda.is_available() else 'N/A'})")
            logger.info(f"Using GPU: cuda:{args.gpu_ids[0]}")
        else:
            device = torch.device('cpu')
            print("[Device] Using CPU (no GPU or gpu_ids not set)")
            logger.info("Using CPU!")

    args.device = device
    if is_main_process:
        if is_distributed:
            logger.info(f"DDP training on {dist.get_world_size()} GPUs, local_rank={local_rank}")
        else:
            logger.info(f"Single-process training on device: {device}")

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

    # using DDP for multi-GPU
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    atio = ATIO().getTrain(args)
    # do train
    atio.do_train(model, dataloader)

    # load pretrained model（统一从 rank0 保存的 checkpoint 重新构建单卡模型做测试）
    if is_distributed:
        dist.barrier()
    assert os.path.exists(args.model_save_path)
    state = torch.load(args.model_save_path, map_location=device)
    if isinstance(state, dict) and state and next(iter(state.keys())).startswith('module.'):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    test_model = AMIO(args).to(device)
    test_model.load_state_dict(state, strict=True)
    results = atio.do_test(test_model, dataloader['test'], mode="TEST")

    del model, test_model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(5)

    return results


def run_eval_only(args, dataloader, model_path):
    """仅加载已训模型并在给定 dataloader 的 test 上评估，用于统一模型在各缺失率下的测试。"""
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    device = torch.device('cuda:%d' % int(args.gpu_ids[0])) if using_cuda else torch.device('cpu')
    args.device = device
    model = AMIO(args).to(device)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and state and next(iter(state.keys())).startswith('module.'):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    atio = ATIO().getTrain(args)
    results = atio.do_test(model, dataloader['test'], mode="TEST")
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return results


def run_unified(args):
    """统一缺失率训练：单次训练中每个样本随机缺失率，一次得到可应对多种缺失率的模型；训完后在该模型上对各缺失率做测试。"""
    init_args = args
    seeds = getattr(args, 'seeds', [getattr(args, 'seed', 2222)])
    _mrp = getattr(args, 'missing_rates_pool', None)
    missing_rates_pool = list(_mrp) if _mrp is not None else list(np.arange(0, 1.05, 0.1).round(2))
    _mr = getattr(args, 'missing_rates', None)
    eval_rates = list(_mr) if _mr is not None else missing_rates_pool
    res_save_dir = os.path.join(args.res_save_dir, 'normals')
    model_save_dir = args.model_save_dir
    if not os.path.exists(res_save_dir):
        os.makedirs(res_save_dir)

    # 训练阶段：每个 seed 训一个 unified 模型（训练集随机缺失率，valid/test 用固定 0.5）
    for i, seed in enumerate(seeds):
        init_args.unified_missing = True
        init_args.random_missing_rate = True
        init_args.missing_rates_pool = missing_rates_pool
        init_args.missing_rate = (0.5, 0.5, 0.5)
        init_args.seed = seed
        init_args.save_model = True
        config = ConfigRegression(init_args)
        args = config.get_config()
        args.seed = seed
        args.seeds = seeds
        args.unified_missing = True
        args.random_missing_rate = True
        args.missing_rates_pool = missing_rates_pool
        args.missing_rate = (0.5, 0.5, 0.5)
        args.save_model = True
        args.cur_time = i + 1
        setup_seed(seed)
        logger.info('Unified missing: training one model with random missing rate, seed=%s' % seed)
        dataloader = MMDataLoader(args)
        run(args, dataloader)

    # 评估阶段：对每个 seed 的 unified 模型，在每个缺失率下做 test
    criterions = None
    all_results = {}
    for seed in seeds:
        unified_path = os.path.join(model_save_dir, f'{args.modelName}-{args.datasetName}-{args.train_mode}-unified-{seed}.pth')
        if not os.path.exists(unified_path):
            logger.warning('Unified model not found: %s' % unified_path)
            continue
        for mr in eval_rates:
            init_args.missing_rate = tuple([float(mr), float(mr), float(mr)])
            config = ConfigRegression(init_args)
            args = config.get_config()
            args.seed = seed
            args.seeds = seeds
            args.missing_rate = tuple([float(mr), float(mr), float(mr)])
            dl = MMDataLoader(args)
            res = run_eval_only(args, dl, unified_path)
            if criterions is None:
                criterions = list(res.keys())
            all_results[(seed, mr)] = res
        gc.collect()

    if not criterions or not all_results:
        return
    save_path = os.path.join(res_save_dir, f'{args.datasetName}-{args.train_mode}-unified.csv')
    df = pd.DataFrame(columns=["Model", "Missing_Rate"] + criterions)
    for mr in eval_rates:
        values_per_criterion = {c: [] for c in criterions}
        for seed in seeds:
            if (seed, mr) not in all_results:
                continue
            for c in criterions:
                values_per_criterion[c].append(all_results[(seed, mr)][c])
        line = [f'{args.modelName}-unified', float(mr)]
        for c in criterions:
            v = values_per_criterion[c]
            if v:
                line.append((round(np.mean(v) * 100, 2), round(np.std(v) * 100, 2)))
            else:
                line.append((None, None))
        df.loc[len(df)] = line
    df.to_csv(save_path, index=None)
    logger.info('Unified results (per missing rate) saved to %s' % save_path)


def run_normal(args):
    res_save_dir = os.path.join(args.res_save_dir, 'normals')
    init_args = args
    model_results = []
    seeds = args.seeds

    missing_rate = 0.0
    args = init_args
    # load config
    config = ConfigRegression(args)
    args = config.get_config()
    # load data（根据是否分布式选择是否使用 DistributedSampler）
    is_distributed = dist.is_initialized()
    dataloader = MMDataLoader(args, distributed=is_distributed)
    # run results
    for i, seed in enumerate(seeds):
        if i == 0 and args.data_missing:
            missing_rate = str(round(args.missing_rate[0], 1))
        setup_seed(seed)
        args.seed = seed
        logger.info('Start running %s... with missing_rate=%s' % (args.modelName, missing_rate))
        logger.info(args)
        # runnning
        args.cur_time = i+1
        test_results = run(args, dataloader)
        # 仅主进程汇总结果
        if not dist.is_initialized() or dist.get_rank() == 0:
            model_results.append(test_results)
            logger.info(f"==> Test results of seed {seed}:\n{test_results}")
    # 只有主进程负责写结果
    if dist.is_initialized() and dist.get_rank() != 0:
        return None, None

    criterions = list(model_results[0].keys())
    # load other results
    save_path = os.path.join(res_save_dir, \
                        f'{args.datasetName}-{args.train_mode}-{missing_rate}.csv')
    if not os.path.exists(res_save_dir):
        os.makedirs(res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Model"] + criterions)
    # save results
    res = [args.modelName]
    for c in criterions:
        values = [r[c] for r in model_results]
        mean = round(np.mean(values)*100, 2)
        std = round(np.std(values)*100, 2)
        res.append((mean, std))
    df.loc[len(df)] = res
    df.to_csv(save_path, index=None)
    logger.info('Results are added to %s...' %(save_path))
    # store results
    returned_res = res[1:]

    # detailed results
    import datetime
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(res_save_dir, \
                        f'{args.datasetName}-{args.train_mode}-{missing_rate}-detail.csv')
    if not os.path.exists(res_save_dir):
        os.makedirs(res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Time", "Model", "Params", "Seed"] + criterions)
    # seed
    for i, seed in enumerate(seeds):
        res = [cur_time, args.modelName, str(args), f'{seed}']
        for c in criterions:
            val = round(model_results[i][c]*100, 2)
            res.append(val)
        df.loc[len(df)] = res
    # mean
    res = [cur_time, args.modelName, str(args), '<mean/std>']
    for c in criterions:
        values = [r[c] for r in model_results]
        mean = round(np.mean(values)*100, 2)
        std = round(np.std(values)*100, 2)
        res.append((mean, std))
    df.loc[len(df)] = res
    # max
    res = [cur_time, args.modelName, str(args), '<max/seed>']
    for c in criterions:
        values = [r[c] for r in model_results]
        max_val = round(np.max(values)*100, 2)
        max_seed = seeds[np.argmax(values)]
        res.append((max_val, max_seed))
    df.loc[len(df)] = res
    # min
    res = [cur_time, args.modelName, str(args), '<min/seed>']
    for c in criterions:
        values = [r[c] for r in model_results]
        min_val = round(np.min(values)*100, 2)
        min_seed = seeds[np.argmin(values)]
        res.append((min_val, min_seed))
    df.loc[len(df)] = res
    df.to_csv(save_path, index=None)
    abs_path = os.path.abspath(save_path)
    logger.info('Detailed results are added to %s' % abs_path)
    print(f"[结果] 已写入 detail: {abs_path}")

    return returned_res, criterions


def set_log(args):
    res_dir = os.path.join(args.res_save_dir, 'normals')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    log_file_path = os.path.join(res_dir, f'run-once-{args.modelName}-{args.datasetName}.log')
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
                        help='indicates the gpus will be used in single-process mode. In DDP mode, use torchrun to launch.')
    parser.add_argument('--missing_rates', type=float, nargs='+', default=None)

    # new
    parser.add_argument('--seed', type=int, default=2222, help='start seed (used when --num_seeds is set)')
    parser.add_argument('--num_seeds', type=int, default=None,
                        help='number of seeds to run; seeds = range(seed, seed+num_seeds). e.g. --num_seeds 5 runs 5 seeds')
    # NOTE: 之前用于自定义任意多种子列表的 --seeds 已废弃，保留为注释以便参考
    # parser.add_argument('--seeds', type=int, nargs='+', default=None,
    #                     help='explicit seed list, e.g. --seeds 42 123 456 789. overrides --seed/--num_seeds')
    parser.add_argument('--exp_name', type=str, default='', help='experiment name')
    parser.add_argument('--diff_missing', type=float, nargs='+', default=None, help='different missing rates for text, audio, and video')
    parser.add_argument('--KeyEval', type=str, default='Loss', help='the evaluation metric used to select best model')
    parser.add_argument('--save_model', action='store_true', help='save the best model in each run (i.e., each seed)')
    parser.add_argument('--unified_missing', action='store_true',
                        help='统一缺失率训练：单次训练中每个样本随机缺失率，一次得到可应对多种缺失率的模型；训完后在该模型上对各缺失率做测试')
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
    # initialize distributed training
    local_rank, rank, world_size = init_distributed()
    args.local_rank = local_rank
    is_main_process = rank == 0

    global logger
    if is_main_process:
        logger = set_log(args)
    else:
        # 非主进程降低日志等级，避免刷屏
        logger = logging.getLogger()
        logger.setLevel(logging.WARNING)

    if len(args.gpu_ids) > 0 and torch.cuda.is_available() and not dist.is_initialized():
        print(f"[启动] 单进程模式，将使用 GPU: cuda:{args.gpu_ids[0]}")
    else:
        if dist.is_initialized():
            if is_main_process:
                print(f"[启动] 分布式模式，world_size={world_size}, local_rank={local_rank}")
        else:
            if len(args.gpu_ids) == 0:
                print("[启动] 未传入 gpu_ids，将使用 CPU")
            elif not torch.cuda.is_available():
                print("[启动] 当前环境看不到 GPU，将使用 CPU")
    # ===== 种子设置：回退到与 master 相同的简单多种子逻辑 =====
    # 固定使用起始种子 + 数量（默认 3 个），不再支持自定义任意种子列表。
    args.seeds = [111, 1111, 11111] if args.num_seeds is None else list(
        range(args.seed, args.seed + args.num_seeds)
    )
    args.num_seeds = len(args.seeds)
    print(f"[种子] 将运行 {args.num_seeds} 个随机种子: {args.seeds}")

    if args.missing_rates is None:
        if args.datasetName in ['mosi', 'mosei']:
            args.missing_rates = np.arange(0, 1.0 + 0.1, 0.1).round(1)
        else:
            args.missing_rates = np.arange(0, 0.5 + 0.1, 0.1).round(1)
    if getattr(args, 'unified_missing', False):
        run_unified(args)
    else:
        aggregated_results, metrics = [], []
        for mr in args.missing_rates:
            args.missing_rate = tuple([mr, mr, mr])
            res, criterions = run_normal(args)
            aggregated_results.append(res)
            metrics = criterions

        # save aggregated results only when multiple missing rates (avoid overwriting with single rate)
        if len(args.missing_rates) > 1:
            save_path = os.path.join(
                args.res_save_dir,
                'normals',
                f'{args.datasetName}-{args.train_mode}-aggregated.csv'
            )
            if not os.path.exists(args.res_save_dir):
                os.makedirs(args.res_save_dir)
            seeds_str = ','.join(map(str, args.seeds))
            if os.path.exists(save_path):
                df = pd.read_csv(save_path)
                if 'Seeds' not in df.columns:
                    df.insert(2, 'Seeds', '')
            else:
                df = pd.DataFrame(columns=["Model", "Missing_Rate", "Seeds"] + metrics)
            for mr, res in zip(args.missing_rates, aggregated_results):
                line = [args.modelName, mr, seeds_str] + res
                df.loc[len(df)] = line
            # auc
            agg_results = np.array(aggregated_results)[:, :, 0]
            auc_res = np.sum(agg_results[:-1] + agg_results[1:], axis=0) / 2 * 0.1
            df.loc[len(df)] = [args.modelName, 'AUC', seeds_str] + auc_res.round(1).tolist()
            df.to_csv(save_path, index=None)
            logger.info('Aggregated results are added to %s...' % (save_path))

    # cleanup distributed
    if dist.is_initialized():
        dist.destroy_process_group()