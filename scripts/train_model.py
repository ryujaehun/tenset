"""Train a cost model with a dataset."""
import argparse
import logging
import pickle
import random
import torch
import numpy as np
import wandb
import tvm
from tvm.auto_scheduler.utils import to_str_round
from tvm.auto_scheduler.cost_model import RandomModelInternal

from common import load_and_register_tasks, str2bool

from tvm.auto_scheduler.dataset import Dataset, LearningTask
from tvm.auto_scheduler.cost_model.xgb_model import XGBModelInternal
from tvm.auto_scheduler.cost_model.mlp_model import MLPModelInternal
from tvm.auto_scheduler.cost_model.lgbm_model import LGBModelInternal
from tvm.auto_scheduler.cost_model.tabnet_model import TabNetModelInternal
from tvm.auto_scheduler.cost_model.metric import (
    metric_rmse,
    metric_r_squared,
    metric_pairwise_comp_accuracy,
    metric_top_k_recall,
    metric_peak_score,
    metric_mape,
    random_mix,
)
from tvm import auto_scheduler
from tvm.auto_scheduler.dataset import LearningTask
from common import get_task_info_filename, get_measure_record_filename
import os 
def evaluate_model(model, test_set,ratio=1):
    # make prediction
    prediction = model.predict(test_set)

    # compute weighted average of metrics over all tasks
    tasks = list(test_set.tasks())
    weights = [len(test_set.throughputs[t]) for t in tasks]
    print("Test set sizes:", weights)

    rmse_list = []
    r_sqaured_list = []
    pair_acc_list = []
    mape_list = []
    peak_score1_list = []
    peak_score5_list = []


    for task in tasks:
        
        preds = prediction[task]
        labels = test_set.throughputs[task]
        length = int(len(preds)*0.9)
        idx = np.arange(len(preds))[:length]
        preds = preds[idx]
        labels = labels[idx]
        rmse_list.append(np.square(metric_rmse(preds, labels)))
        r_sqaured_list.append(metric_r_squared(preds, labels))
        pair_acc_list.append(metric_pairwise_comp_accuracy(preds, labels))
        mape_list.append(metric_mape(preds, labels))
        peak_score1_list.append(metric_peak_score(preds, labels, 1))
        peak_score5_list.append(metric_peak_score(preds, labels, 5))

    rmse = np.sqrt(np.average(rmse_list, weights=weights))
    r_sqaured = np.average(r_sqaured_list, weights=weights)
    pair_acc = np.average(pair_acc_list, weights=weights)
    mape = np.average(mape_list, weights=weights)
    peak_score1 = np.average(peak_score1_list, weights=weights)
    peak_score5 = np.average(peak_score5_list, weights=weights)

    eval_res = {
        "RMSE": rmse,
        "R^2": r_sqaured,
        "pairwise comparision accuracy": pair_acc,
        "mape": mape,
        "average peak score@1": peak_score1,
        "average peak score@5": peak_score5,
    }
    return eval_res


def make_model(name, use_gpu=False,args=None,wandb=None):
    """Make model according to a name"""
    if name == "xgb":
        return XGBModelInternal(use_gpu=use_gpu)
    elif name == "mlp" or name == "transformer" or name.lower()=='lstm':
        return MLPModelInternal(args=args,wandb=wandb)#,model_type='transformer')
    elif name == 'lgbm':
        return LGBModelInternal(use_gpu=use_gpu)
    elif name == 'tab':
        return TabNetModelInternal(loss_type=args.loss,wandb=wandb,use_gpu=use_gpu)
    elif name == "random":
        return RandomModelInternal()
    else:
        raise ValueError("Invalid model: " + name)
 

def train_zero_shot(dataset, train_ratio, model_names, split_scheme, use_gpu,args,wandb):
    # Split dataset
    
    if split_scheme == "within_task":
        train_set, test_set = dataset.random_split_within_task(train_ratio)
    elif split_scheme == "by_task":
        train_set, test_set = dataset.random_split_by_task(train_ratio)
    elif split_scheme == "by_target":
        train_set, test_set = dataset.random_split_by_target(train_ratio)
    else:
        raise ValueError("Invalid split scheme: " + split_scheme)

    print("Train set: %d. Task 0 = %s" % (len(train_set), train_set.tasks()[0]))
    if len(test_set) == 0:
        test_set = train_set
    print("Test set:  %d. Task 0 = %s" % (len(test_set), test_set.tasks()[0]))

    # Make models
    names = model_names.split("@")
    models = []
    for name in names:
        models.append(make_model(name, use_gpu,args,wandb))
    eval_results = []
    for name, model in zip(names, models):
        # Train the model
        filename = f'{args.save}' + ".pkl"
        model.fit_base(train_set, valid_set=test_set)
        print("Save model to %s" % filename)
        model.save(filename)

        # Evaluate the model
        eval_res = evaluate_model(model, test_set)
        print(name, to_str_round(eval_res))
        eval_results.append(eval_res)

    # Print evaluation results
    for i in range(len(models)):
        print("-" * 60)
        print("Model: %s" % names[i])
        for key, val in eval_results[i].items():
            print("%s: %.4f" % (key, val))
    return models[0]


def eval_cost_model_on_weighted_tasks(model, eval_task_dict, eval_dataset, top_ks):
    """Evaluate a cost model on weighted tasks"""
    preds_dict = model.predict(eval_dataset)

    best_latency = 0
    latencies = [0] * len(top_ks)
    for task, weight in eval_task_dict.items():
        if task not in eval_dataset.throughputs:
            print(f"Warning: cannot find {task.workload_key} in the eval_dataset. Skipped.")
            continue

        preds = preds_dict[task]
        labels, min_latency = eval_dataset.throughputs[task], eval_dataset.min_latency[task]

        real_values = labels[np.argsort(-preds)]
        real_latency = min_latency / np.maximum(real_values, 1e-5)

        for i, top_k in enumerate(top_ks):
            latencies[i] += np.min(real_latency[:top_k]) * weight
        best_latency += min_latency * weight

    return latencies, best_latency


def eval_cost_model_on_network(model, network_key, target, top_ks,args):
    # Read tasks of the network
    target = tvm.target.Target(target)
    task_info_filename = get_task_info_filename(network_key, target)
    tasks, task_weights = pickle.load(open(task_info_filename, "rb"))
    network_task_key2 = (network_key, str(target))

    # Featurizes a dataset 
    dataset_file = f".dataset_cache/{network_task_key2}.network.feature_cache"
    if not os.path.exists(dataset_file):
        # get file names of these tasks
        filenames = []
        for task in tasks:
            filename = get_measure_record_filename(task, target)
            filenames.append(filename)

        # make a dataset
        auto_scheduler.dataset.make_dataset_from_log_file(
            filenames, dataset_file, min_sample_size=0)
    dataset = pickle.load(open(dataset_file, "rb"))
    if args.maml:
        eval_res = evaluate_model(model, dataset,0.8)
    else:
        eval_res = evaluate_model(model, dataset,1)
    print(to_str_round(eval_res))
    print("===============================================")

    # Make learning tasks and attach weights
    target = dataset.tasks()[0].target
    learning_tasks = [LearningTask(t.workload_key, target) for t in tasks]
    task_dict = {task: weight for task, weight in zip(learning_tasks, task_weights)}

    return eval_cost_model_on_weighted_tasks(model, task_dict, dataset, top_ks)





def eval_cost_model_on_log_file(model, log_file, network_key, target, top_ks):
    target = tvm.target.Target(target)
    task_info_filename = get_task_info_filename(network_key, target)
    tasks, task_weights = pickle.load(open(task_info_filename, "rb"))

    dataset_file = "tmp_dataset_file.pkl"
    auto_scheduler.dataset.make_dataset_from_log_file(
        [log_file], dataset_file, min_sample_size=0)
    dataset = pickle.load(open(dataset_file, "rb"))

    target = dataset.tasks()[0].target
    learning_tasks = [LearningTask(t.workload_key, target) for t in tasks]
    task_dict = {task: weight for task, weight in zip(learning_tasks, task_weights)}

    return eval_cost_model_on_weighted_tasks(model, task_dict, dataset, top_ks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save",  type=str, default='')
    parser.add_argument("--loss",  type=str, default='rmse')
    parser.add_argument("--maml", default=False, action="store_true")
    parser.add_argument("--eval", default=False, action="store_true")
    parser.add_argument("--wd", type=float, default=7e-4)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--meta_outer_lr", type=float, default=7e-4)
    parser.add_argument("--meta_inner_lr", type=float, default=7e-4)
    parser.add_argument("--dataset", nargs="+", type=str, default=["/root/scripts/dataset-k80.pkl"])
    parser.add_argument("--models", type=str, default="mlp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument(
        "--split-scheme",
        type=str,
        choices=["by_task", "within_task", "by_target"],
        default="by_task",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--use-gpu", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Whether to use GPU for xgb.")
    args = parser.parse_args()
    print("Arguments: %s" % str(args))
    if args.maml:
        wandb.init(name=f'{args.lr}_{args.wd}_{args.meta_inner_lr}_{args.meta_outer_lr}',project=f"MetaTune_{args.loss}_{args.mode}", tags=[f"{args.meta_inner_lr}",f"{args.meta_outer_lr}"])
    elif args.models in ['xgb','lgbm','tab','random']:
        wandb.init(name=f'{args.meta_inner_lr}_{args.meta_outer_lr}',project=f"Baseline_{args.models}_{args.loss}", tags=[f"{args.meta_inner_lr}",f"{args.meta_outer_lr}"])
    else:
        wandb.init(name=f'{args.models}_{args.loss}',project=f"{args.models}_{args.loss}", tags=[f"{args.models}",f"{args.loss}"])
    wandb.config.update(args)
    args.save = f'{args.models}_{args.loss}_{args.lr}_{args.wd}'
    if args.maml:
        args.save += f'_maml_{args.meta_inner_lr}_{args.meta_outer_lr}'
    # Setup random seed and logging
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    logging.basicConfig()
    logging.getLogger("auto_scheduler").setLevel(logging.DEBUG)

    print("Load all tasks...")
    load_and_register_tasks()

    print("Load dataset...")
    dataset = pickle.load(open(args.dataset[0], "rb"))
    for i in range(1, len(args.dataset)):
        tmp_dataset = pickle.load(open(args.dataset[i], "rb"))
        dataset.update_from_dataset(tmp_dataset)

    model = train_zero_shot(dataset, args.train_ratio, args.models, args.split_scheme, args.use_gpu,args,wandb)
    
    network_keys = [
        ("resnet_50", [(1, 3, 224,224)]),
        ("mobilenet_v2", [(1, 3, 224,224)]),
        ("resnext_50", [(1, 3, 224,224)]),
        ("bert_base", [(1, 128)]),
        ("bert_tiny", [(1, 128)]),
    ]
    mm=args.dataset[0].split('-')[-1].split('.')[0]
    target = 'cuda -model=k80'
    top_ks = [1, 5]
    top_1_total = []
    top_5_total = []
    args.eval = True
    for network_key in network_keys:
        latencies, best_latency = eval_cost_model_on_network(model, network_key, target, top_ks,args)
        for top_k, latency in zip(top_ks, latencies):
            print(f"Network: {network_key}\tTop-{top_k} score: {best_latency / latency}")
            
        
        top_1_total.append(best_latency/latencies[0])
        print(f"top 1 score: {best_latency/latencies[0]}")
        top_5_total.append(best_latency / latencies[1])
        print(f"top 5 score: {best_latency / latencies[1]}")
        wandb.log({
                    f"Eval {network_key} Top-1 score": (best_latency / latencies[0]),
                    f"Eval {network_key} Top-5 score": (best_latency / latencies[1]),
                    }, )
     
    wandb.log({
                    f"final Top-1 score": sum(top_1_total) / len(top_1_total),
                    f"final Top-5 score": sum(top_5_total) / len(top_5_total),
                    }, )
    print(f"average top 1 score is {sum(top_1_total) / len(top_1_total)}")
    print(f"average top 5 score is {sum(top_5_total) / len(top_5_total)}")

 


