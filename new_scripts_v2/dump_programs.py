"""Dump programs for all tasks"""

import argparse
import pickle
import gc
import glob

import time
import os

from tqdm import tqdm
import tvm
from tvm import auto_scheduler
from tvm.auto_scheduler.cost_model import XGBModel
from common import load_and_register_tasks, get_to_measure_filename

def dump_program(task, size, max_retry_iter=10):
    filename = get_to_measure_filename(task)
    print(filename)
    task.target = tvm.target.Target("cuda")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)
    tasks = [task]
    task_weights = [1]
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        verbose = 0,
        num_measure_trials=4000,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(filename)],
    )

    tuner.tune(tune_option)
    del measure_ctx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-idx", type=int)
    parser.add_argument("--end-idx", type=int)
    parser.add_argument("--step-idx", type=int,default=8)
    parser.add_argument("--size", type=int, default=4000)
    args = parser.parse_args()

    tasks = load_and_register_tasks()

    start_idx = args.start_idx or 0
    end_idx = args.end_idx or len(tasks)
    step_idx = args.step_idx or 1

    # Dump programs for all tasks
    for task in tqdm(tasks[start_idx::step_idx]):
        try:
            dump_program(task, size=args.size)
        except Exception as e:
            print(e)

            pass
        gc.collect()

