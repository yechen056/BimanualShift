import os
import pickle
import gc
import torch
from omegaconf import DictConfig

from rlbench import ObservationConfig
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
from yarr.runners.offline_train_runner import OfflineTrainRunner
from yarr.utils.stat_accumulator import SimpleAccumulator

import torch.distributed as dist
from bift.methods import agent_factory
from bift.methods import replay_utils
from bift import peract_config
from functools import partial
from tqdm import tqdm


def run_seed(
    rank,
    cfg: DictConfig,
    obs_config: ObservationConfig,
    seed,
    world_size,
) -> None:

    peract_config.config_logging()

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    tasks = cfg.rlbench.tasks
    cams = cfg.rlbench.cameras

    task_folder = cfg.replay.task_folder if len(tasks) > 1 else tasks[0]
    replay_path = os.path.join(cfg.replay.path, task_folder)
    agent = agent_factory.create_agent(cfg)

    if not agent:
        print("Unable to create agent")
        return

    if cfg.method.name.startswith("BIMANUALSHIFT_PERACT"):
        print(replay_path)
        if os.path.exists(replay_path) and os.listdir(replay_path):
            print("Replay files found. Loading...")
            replay_buffer = replay_utils.create_replay(cfg, replay_path)
            replay_files = [
                os.path.join(replay_path, f)
                for f in os.listdir(replay_path)
                if f.endswith(".replay")
            ]
            for replay_file in tqdm(replay_files, desc="Processing files"):
                with open(replay_file, "rb") as f:
                    try:
                        replay_data = pickle.load(f)
                        replay_buffer.load_add(replay_data)
                    except pickle.UnpicklingError as e:
                        print(f"Error unpickling file {replay_file}: {e}")
        else:
            print("No replay files found. Creating replay...")
            replay_buffer = replay_utils.create_replay(cfg, replay_path)
            replay_utils.fill_multi_task_replay(
                cfg,
                obs_config,
                rank,
                replay_buffer,
                tasks,
            )

    else:
        raise ValueError("Method %s does not exists." % cfg.method.name)

    wrapped_replay = PyTorchReplayBuffer(
        replay_buffer, num_workers=cfg.framework.num_workers
    )
    stat_accum = SimpleAccumulator(eval_video_fps=30)

    cwd = os.getcwd()
    weightsdir = os.path.join(cwd, "seed%d" % seed, "weights")
    logdir = os.path.join(cwd, "seed%d" % seed)

    train_runner = OfflineTrainRunner(
        agent=agent,
        wrapped_replay_buffer=wrapped_replay,
        train_device=rank,
        stat_accumulator=stat_accum,
        iterations=cfg.framework.training_iterations,
        logdir=logdir,
        logging_level=cfg.framework.logging_level,
        log_freq=cfg.framework.log_freq,
        weightsdir=weightsdir,
        num_weights_to_keep=cfg.framework.num_weights_to_keep,
        save_freq=cfg.framework.save_freq,
        tensorboard_logging=cfg.framework.tensorboard_logging,
        csv_logging=cfg.framework.csv_logging,
        load_existing_weights=cfg.framework.load_existing_weights,
        rank=rank,
        world_size=world_size,
        cfg=cfg,
    )

    train_runner._on_thread_start = partial(
        peract_config.config_logging, cfg.framework.logging_level
    )

    train_runner.start()

    del train_runner
    del agent
    gc.collect()
    torch.cuda.empty_cache()
