from collections import OrderedDict
import copy
from itertools import chain
import multiprocessing
import os
import pickle
import random
import time
import io
import json
import numpy as np
import torch
import torch.nn.functional as F
import logging
logger = logging.getLogger("auto_scheduler")
import higher
from tvm.auto_scheduler.dataset import Dataset, LearningTask
from tvm.auto_scheduler.feature import (
    get_per_store_features_from_measure_pairs, get_per_store_features_from_states)
from tvm.auto_scheduler.measure_record import RecordReader
from .xgb_model import get_workload_embedding
from .cost_model import PythonBasedModel
from torch.utils.data import DataLoader
class SegmentDataLoader():
    def __init__(
            self,
            dataset,
            batch_size,
            device,
            use_workload_embedding=True,
            use_target_embedding=False,
            target_id_dict={},
            fea_norm_vec=None,
            shuffle=False,
            num_workers=12,
            pin_memory =True,
            prefetch_factor=4,
    ):
        # super(SegmentDataLoader,self).__init__(dataset = dataset,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,prefetch_factor=prefetch_factor)
        self.device = device
        self.shuffle = shuffle
        self.number = len(dataset)
        self.batch_size = batch_size

        self.segment_sizes = torch.empty((self.number,), dtype=torch.int32)
        self.labels = torch.empty((self.number,), dtype=torch.float32)

        # Flatten features
        flatten_features = []
        ct = 0
        for task in dataset.features:
            throughputs = dataset.throughputs[task]
            self.labels[ct: ct + len(throughputs)] = torch.tensor(throughputs)
            task_embedding = None
            if use_workload_embedding or use_target_embedding:
                task_embedding = np.zeros(
                    (10 if use_workload_embedding else 0),
                    dtype=np.float32,
                )

                if use_workload_embedding:
                    tmp_task_embedding = get_workload_embedding(task.workload_key)
                    task_embedding[:9] = tmp_task_embedding

                if use_target_embedding:
                    target_id = target_id_dict.get(
                        str(task.target), np.random.randint(0, len(target_id_dict))
                    )
                    if 9+target_id<len(task_embedding):
                        task_embedding[9+target_id] = 1.0


            for row in dataset.features[task]:
                self.segment_sizes[ct] = len(row)

                if task_embedding is not None:
                    tmp = np.tile(task_embedding, (len(row), 1))
                    flatten_features.extend(np.concatenate([row, tmp], axis=1))
                else:
                    flatten_features.extend(row)
                ct += 1
        # max_seg_len = self.segment_sizes.max()
        self.features = torch.tensor(np.array(flatten_features, dtype=np.float32))
        if fea_norm_vec is not None:
            self.normalize(fea_norm_vec)

        self.feature_offsets = (
                    torch.cumsum(self.segment_sizes, 0, dtype=torch.int32) - self.segment_sizes).cpu().numpy()
        self.iter_order = self.pointer = None

    def normalize(self, norm_vector=None):
        if norm_vector is None:
            norm_vector = torch.ones((self.features.shape[1],))
            for i in range(self.features.shape[1]):
                max_val = self.features[:, i].max().item()
                if max_val > 0:
                    norm_vector[i] = max_val
        self.features /= norm_vector

        return norm_vector

    def __iter__(self):
        if self.shuffle:
            self.iter_order = torch.randperm(self.number)
        else:
            self.iter_order = torch.arange(self.number)
        self.pointer = 0

        return self

    def sample_batch(self, batch_size):
        batch_indices = np.random.choice(self.number, batch_size)
        return self._fetch_indices(batch_indices)

    def __next__(self):
        if self.pointer >= self.number:
            raise StopIteration

        batch_indices = self.iter_order[self.pointer: self.pointer + self.batch_size]
        self.pointer += self.batch_size
        return self._fetch_indices(batch_indices)

    def _fetch_indices(self, indices):
        segment_sizes = self.segment_sizes[indices]

        feature_offsets = self.feature_offsets[indices]
        feature_indices = np.empty((segment_sizes.sum(),), dtype=np.int32)
        ct = 0
        for offset, seg_size in zip(feature_offsets, segment_sizes.numpy()):
            feature_indices[ct: ct + seg_size] = np.arange(offset, offset + seg_size, 1)
            ct += seg_size

        features = self.features[feature_indices]
        labels = self.labels[indices]
        return (x.to(self.device) for x in (segment_sizes, features, labels))

    def __len__(self):
        return self.number


class SegmentSumMLPModule(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, use_norm=False, add_sigmoid=False):
        super().__init__()

        self.segment_encoder = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )

        self.add_sigmoid = add_sigmoid

        if use_norm:
            self.norm = torch.nn.BatchNorm1d(hidden_dim)
        else:
            self.norm = torch.nn.Identity()

        self.l0 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.l1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Linear(hidden_dim, out_dim)

    def freeze_for_fine_tuning(self):
        for x in self.segment_encoder.parameters():
            x.requires_grad_(False)

    def forward(self, segment_sizes, features, params=None):
        n_seg = segment_sizes.shape[0]
        device = features.device

        segment_sizes = segment_sizes.long()

        features = self.segment_encoder(
            features
        )
        segment_indices = torch.repeat_interleave(
            torch.arange(n_seg, device=device), segment_sizes
        )

        n_dim = features.shape[1]
        segment_sum = torch.scatter_add(
            torch.zeros((n_seg, n_dim), dtype=features.dtype, device=device),
            0,
            segment_indices.view(-1, 1).expand(-1, n_dim),
            features,
        )
        output = self.norm(segment_sum)
        output = self.l0(output) + output
        output = self.l1(output) + output
        output = self.decoder(
            output
        ).squeeze()

        if self.add_sigmoid:
            output = torch.sigmoid(output)

        return output

class LSTMModuel(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.segment_encoder = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )

        self.norm = torch.nn.Identity()

        self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim)
        self.l0 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.l1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Linear(hidden_dim, out_dim)


    def forward(self, segment_sizes, features, params=None):
        features = self.segment_encoder(
            features
        )

        seqs = []
        ct = 0
        for seg_size in segment_sizes:
            seqs.append(features[ct: ct + seg_size])
            ct += seg_size
        output = torch.nn.utils.rnn.pad_sequence(seqs)

        output, (h, c)  = self.lstm(output)
        output = self.norm(h[0])
        output = self.l0(output) + output
        output = self.l1(output) + output

        output = self.decoder(
            output
        ).squeeze()

        return output


class OneShotModule(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, out_dim, add_sigmoid=False):
        super().__init__()

        self.add_sigmoid = add_sigmoid

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
        )

        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,dropout=0.2)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, 2)


        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, segment_sizes, features,params=None):
        n_seg = segment_sizes.shape[0]
        device = features.device
        features = self.encoder(features)
        seqs = []
        ct = 0
        for seg_size in segment_sizes:
            seqs.append(features[ct: ct + seg_size])
            ct += seg_size
        output = torch.nn.utils.rnn.pad_sequence(seqs)
        output = self.transformer_encoder(output)
        output = self.decoder(output).sum(0).squeeze()
        if self.add_sigmoid:
            output = torch.sigmoid(output)
        return output

class MHAModule(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, out_dim, add_sigmoid=False):
        super().__init__()

        self.add_sigmoid = add_sigmoid

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.l0 = torch.nn.MultiheadAttention(hidden_dim, num_heads)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, segment_sizes, features,params=None):
        n_seg = segment_sizes.shape[0]
        device = features.device
        features = self.encoder(features)
        # ([4258, 1024])
        seqs = []
        ct = 0
        for seg_size in segment_sizes:
            seqs.append(features[ct: ct + seg_size])
            ct += seg_size
        output = torch.nn.utils.rnn.pad_sequence(seqs)
        output = self.l0(output, output, output)[0] + output
        output = self.decoder(output).sum(0).squeeze()
        if self.add_sigmoid:
            output = torch.sigmoid(output)

        return output


def make_net(params):
    if params["type"] == "SegmentSumMLP":
        return SegmentSumMLPModule(
            params["in_dim"], params["hidden_dim"], params["out_dim"],
            add_sigmoid=params['add_sigmoid']
        )
    elif params["type"] == "MultiHeadAttention":
        return MHAModule(
            params['in_dim'], params['hidden_dim'], params['num_heads'], params['out_dim'],
            add_sigmoid=params['add_sigmoid']
        )
    elif params["type"] == "OneShot":
        return OneShotModule(
            params['in_dim'], params['hidden_dim'], params['num_heads'], params['out_dim'],
            add_sigmoid=params['add_sigmoid']
        )
    elif params["type"] == "LSTM":
        return LSTMModuel(
            params["in_dim"], params["hidden_dim"], params["out_dim"],
        )
    else:
        raise ValueError("Invalid type: " + params["type"])


def moving_average(average, update):
    if average is None:
        return update
    else:
        return average * 0.95 + update * 0.05


class MLPModelInternal:
    def __init__(self, device=None, few_shot_learning="base_only", use_workload_embedding=True, use_target_embedding=True,
                 loss_type='lambdaRankLoss',model_type='mlp',args=None,wandb=None):
        if device is None:
            if torch.cuda.device_count():
                device = 'cuda:0'
            else:
                device = 'cpu'
        if args!=None:
            self.args = args
        else:
            self.args = None
        if wandb !=None:
            self.wandb = wandb
        else:
            self.wandb = None
        # Common parameters
        if self.args.models =='mlp':
            self.batch_size = 4096
            self.infer_batch_size = 4096
            self.net_params = {
            "type": "SegmentSumMLP",
            "in_dim": 164 + (10 if use_workload_embedding else 0),  
            "hidden_dim": 256,
            "out_dim": 1,
        }

        elif self.args.models =='transformer':
            self.batch_size = 1024
            self.infer_batch_size = 1024
            self.net_params = {
            "type": "MultiHeadAttention",
            "in_dim": 164+ (10 if use_workload_embedding else 0),  
            "num_heads": 8,
            "hidden_dim": 1024,
            "out_dim": 1,
            }
        elif self.args.models =='oneshot':
            self.batch_size = 1024
            self.infer_batch_size = 1024
            self.net_params = {
            "type": "OneShot",
            "in_dim": 164+ (10 if use_workload_embedding else 0),  
            "num_heads": 8,
            "hidden_dim": 1024,
            "out_dim": 1,
            }
        elif self.args.models.lower() =='lstm':
            self.batch_size = 1024
            self.infer_batch_size = 1024
            self.net_params = {
            "type": "LSTM",
            "in_dim": 164+ (10 if use_workload_embedding else 0),  
            "hidden_dim": 1024,
            "out_dim": 1,
            }
        else:
            raise('Invaid Model')
        self.target_id_dict = {}
        loss_type = self.loss_type = self.args.loss
        
        self.n_epoch = 70
        self.lr = 5e-4

        if loss_type == 'rmse' or loss_type == 'rmse':
            self.loss_func = torch.nn.MSELoss()
            self.net_params['add_sigmoid'] = True
        elif loss_type == 'rankNetLoss':
            self.loss_func = RankNetLoss()
            self.net_params['add_sigmoid'] = False
        elif loss_type == 'lambdaRankLoss':
            self.loss_func = LambdaRankLoss()
            self.net_params['add_sigmoid'] = False
            self.n_epoch = 50
        elif loss_type == 'listNetLoss':
            self.loss_func = ListNetLoss()
            self.lr = 9e-4
            self.n_epoch = 50
            self.net_params['add_sigmoid'] = False
        else:
            raise ValueError("Invalid loss type: " + loss_type)

        self.grad_clip = 0.5
        if self.args.maml:
            few_shot_learning='MAML'
        self.few_shot_learning = few_shot_learning
        self.fea_norm_vec = None
        self.use_workload_embedding = use_workload_embedding
        self.use_target_embedding = use_target_embedding

        # Hyperparameters for self.fit_base
        
        self.wd = 1e-5
        self.device = device
        self.print_per_epoches = 5

        # Hyperparameters for MAML
        self.meta_outer_lr = self.args.meta_outer_lr
        self.meta_inner_lr = self.args.meta_inner_lr
        self.meta_test_num_steps = 5
        self.few_shot_number = 32
        self.meta_batch_size_tasks = 8
        self.meta_batch_size_per_task = 256

        # Hyperparameters for fine-tuning
        self.fine_tune_lr = 1e-6
        self.fine_tune_batch_size = 512
        self.fine_tune_num_steps = 10
        self.fine_tune_wd = 0
        
        # models
        self.base_model = None
        self.local_model = {}

    def fit_base(self, train_set, valid_set=None, valid_train_set=None):
        if self.few_shot_learning == "local_only":
            self.base_model = None
        elif self.few_shot_learning == "MAML":
            self.fine_tune_lr = self.meta_inner_lr
            self.fine_tune_num_steps = self.meta_test_num_steps * 2
            # if self.args.mode == 0:
            #     self.base_model = self._fit_a_MAML_model(train_set, valid_set, valid_train_set)
            # else:
            self.base_model = self._metatune_a_model(train_set, valid_set, valid_train_set)
        else:
            self.base_model = self._fit_a_model(train_set, valid_set, valid_train_set)

    def fit_local(self, train_set, valid_set=None):
        if self.few_shot_learning == "base_only":
            return
        elif self.few_shot_learning == "local_only_mix_task":
            local_model = self._fit_a_model(train_set, valid_set)
            for task in train_set.tasks():
                self.local_model[task] = local_model
        elif self.few_shot_learning == "local_only_per_task":
            for task in train_set.tasks():
                task_train_set = train_set.extract_subset([task])
                local_model = self._fit_a_model(task_train_set, valid_set)
                self.local_model[task] = local_model
        elif self.few_shot_learning == "plus_mix_task":
            self.net_params["hidden_dim"] = 128
            self.loss_type = 'rmse'
            self.loss_func = torch.nn.MSELoss()
            self.net_params['add_sigmoid'] = True
            base_preds = self._predict_a_dataset(self.base_model, train_set)
            diff_train_set = Dataset()
            for task in train_set.tasks():
                diff_train_set.load_task_data(
                    task,
                    train_set.features[task],
                    train_set.throughputs[task] - base_preds[task]
                )

            if valid_set:
                base_preds = self._predict_a_dataset(self.base_model, valid_set)
                diff_valid_set = Dataset()
                for task in valid_set.tasks():
                    diff_valid_set.load_task_data(
                        task,
                        valid_set.features[task],
                        valid_set.throughputs[task] - base_preds[task]
                    )
            else:
                diff_valid_set = None

            diff_model = self._fit_a_model(diff_train_set, diff_valid_set)

            for task in train_set.tasks():
                self.local_model[task] = diff_model
        elif self.few_shot_learning == "plus_per_task":
            base_preds = self._predict_a_dataset(self.base_model, train_set)
            for task in train_set.tasks():
                diff_train_set = Dataset()
                diff_train_set.load_task_data(
                    task,
                    train_set.features[task],
                    train_set.throughputs[task] - base_preds[task]
                )
                diff_model = self._fit_a_model(diff_train_set, valid_set)
                self.local_model[task] = diff_model
        elif self.few_shot_learning == "fine_tune_mix_task":
            self.base_model = self._fine_tune_a_model(self.base_model, train_set, valid_set)
        else:
            raise ValueError("Invalid few-shot learning method: " + self.few_shot_learning)

    def predict(self, dataset):
        
        if self.few_shot_learning in ["base_only", "fine_tune_mix_task", "fine_tune_per_task", "MAML"]:
                
            return self._predict_a_dataset(self.base_model, dataset)
        elif self.few_shot_learning in ["local_only_mix_task", "local_only_per_task"]:
            ret = {}
            for task in dataset.tasks():
                local_preds = self._predict_a_task(self.local_model[task], task, dataset.features[task])
                ret[task] = local_preds
            return ret
        elif self.few_shot_learning in ["plus_mix_task", "plus_per_task"]:
            base_preds = self._predict_a_dataset(self.base_model, dataset)
            ret = {}
            for task in dataset.tasks():
                if task not in self.local_model and self.few_shot_learning == "plus_mix_task":
                    self.local_model[task] = list(self.local_model.values())[0]
                
                local_preds = self._predict_a_task(self.local_model[task], task, dataset.features[task])
                ret[task] = base_preds[task] + local_preds
            return ret
        else:
            raise ValueError("Invalid few show learing: " + self.few_shot_learning)

    def _fit_a_model(self, train_set, valid_set=None, valid_train_set=None, n_epoch=None):
        print("=" * 60 + "\nFit a net. Train size: %d" % len(train_set))

        for task in train_set.tasks():
            self.register_new_task(task)

        
        train_loader = SegmentDataLoader(
            train_set, self.batch_size, self.device, self.use_workload_embedding, self.use_target_embedding,
            self.target_id_dict, shuffle=True  )

        # Normalize features
        if self.fea_norm_vec is None:
            self.fea_norm_vec = train_loader.normalize()
        else:
            train_loader.normalize(self.fea_norm_vec)

        

        if valid_set:
            for task in valid_set.tasks():
                self.register_new_task(task)
            valid_loader = SegmentDataLoader(valid_set, self.infer_batch_size, self.device, self.use_workload_embedding,
                                             self.use_target_embedding, self.target_id_dict,fea_norm_vec=self.fea_norm_vec)

        n_epoch = n_epoch or self.n_epoch
        early_stop = n_epoch // 6

        net = make_net(self.net_params).to(self.device)
        # if self.wandb!=None:
        #     self.wandb.watch(net)
        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, weight_decay=self.wd
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_epoch // 3, gamma=1)

        train_loss = None
        best_epoch = None
        best_train_loss = 1e10
        for epoch in range(n_epoch):
            tic = time.time()

            # train
            net.train()
            for batch, (segment_sizes, features, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                loss = self.loss_func(net(segment_sizes, features), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
                optimizer.step()

                train_loss = moving_average(train_loss, loss.item())
            lr_scheduler.step()

            train_time = time.time() - tic

            if epoch % self.print_per_epoches == 0 or epoch == n_epoch - 1:

                if valid_set and valid_loader:
                    valid_loss = self._validate(net, valid_loader)

                else:
                    valid_loss = 0.0

                if self.loss_type == "rmse":
                    loss_msg = "Train RMSE: %.4f\tValid RMSE: %.4f" % (np.sqrt(train_loss), np.sqrt(valid_loss))
                elif self.loss_type in ["rankNetLoss", "lambdaRankLoss", "listNetLoss"]:
                    loss_msg = "Train Loss: %.4f\tValid Loss: %.4f" % (train_loss, valid_loss)

                print("Epoch: %d\tBatch: %d\t%s\tTrain Speed: %.0f" % (
                    epoch, batch, loss_msg, len(train_loader) / train_time,))
                if self.wandb!=None:
                    if self.loss_type == "rmse":
                        self.wandb.log({
                        "Train RMSE": np.sqrt(train_loss),
                        "Valid RMSE": np.sqrt(valid_loss),
                        "Epoch": epoch,
                        "batch": batch,
                        "Speed": len(train_loader) / train_time})
                    else:
                        self.wandb.log({
                        "Train Loss": train_loss,
                        "Valid Loss": valid_loss,
                        "Epoch": epoch,
                        "batch": batch,
                        "Speed": len(train_loader) / train_time})
            # Early stop
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_epoch = epoch
            elif epoch - best_epoch >= early_stop:
                print("Early stop. Best epoch: %d" % best_epoch)
                break
            if self.args!=None:
                self.save(f"{self.args.save}.pkl")
            else:
                self.save('tmp_mlp.pkl')

        return net

    def register_new_task(self, task):
        target = str(task.target)

        if target not in self.target_id_dict:
            self.target_id_dict[target] = len(self.target_id_dict)
    def _metatune_a_model(self, train_set, valid_set, valid_train_set=None):
        net = make_net(self.net_params).to(self.device)
        self._fine_tune_for_metatune(net,train_set,valid_set,valid_train_set,epoch=70)
        self._fit_METATUNE(net,train_set,valid_set,valid_train_set,epoch=70)
        return net


    def _fine_tune_for_metatune(self,  model,train_set, valid_set=None,valid_train_set=None, epoch=3):


        for task in train_set.tasks():
            self.register_new_task(task)


        train_loader = SegmentDataLoader(
            train_set, self.batch_size, self.device, self.use_workload_embedding, self.use_target_embedding,
            self.target_id_dict, shuffle=True
        )

        # Normalize features
        if self.fea_norm_vec is None:
            self.fea_norm_vec = train_loader.normalize()
        else:
            train_loader.normalize(self.fea_norm_vec)



        if valid_set:
            for task in valid_set.tasks():
                self.register_new_task(task)
            valid_loader = SegmentDataLoader(valid_set, self.infer_batch_size, self.device, self.use_workload_embedding,
                                                self.use_target_embedding, self.target_id_dict,fea_norm_vec=self.fea_norm_vec)

        n_epoch = epoch

        net = model.to(self.device)
        # if self.wandb!=None:
        #     self.wandb.watch(net)
        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, weight_decay=self.wd
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_epoch // 3, gamma=1)

        train_loss = None
        best_epoch = None
        best_train_loss = 1e10
        for epoch in range(n_epoch):
            tic = time.time()

            # train
            net.train()
            for batch, (segment_sizes, features, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                loss = self.loss_func(net(segment_sizes, features), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
                optimizer.step()

                train_loss = moving_average(train_loss, loss.item())
            lr_scheduler.step()

            train_time = time.time() - tic

            if epoch % self.print_per_epoches == 0 or epoch == n_epoch - 1:

                if valid_set and valid_loader:
                    valid_loss = self._validate(net, valid_loader)

                else:
                    valid_loss = 0.0

                if self.loss_type == "rmse":
                    loss_msg = "Train RMSE: %.4f\tValid RMSE: %.4f" % (np.sqrt(train_loss), np.sqrt(valid_loss))
                elif self.loss_type in ["rankNetLoss", "lambdaRankLoss", "listNetLoss"]:
                    loss_msg = "Train Loss: %.4f\tValid Loss: %.4f" % (train_loss, valid_loss)

                print("Epoch: %d\tBatch: %d\t%s\tTrain Speed: %.0f" % (
                    epoch, batch, loss_msg, len(train_loader) / train_time,))
                if self.wandb!=None:
                    if self.loss_type == "rmse":
                        self.wandb.log({
                        "Train RMSE": np.sqrt(train_loss),
                        "Valid RMSE": np.sqrt(valid_loss),
                        "Epoch": epoch,
                        "batch": batch,
                        "Speed": len(train_loader) / train_time})
                    else:
                        self.wandb.log({
                        "Train Loss": train_loss,
                        "Valid Loss": valid_loss,
                        "Epoch": epoch,
                        "batch": batch,
                        "Speed": len(train_loader) / train_time})
            # Early stop
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_epoch = epoch
    def _start(self,model,base_param):

        with torch.no_grad():
            for name, w in model.named_parameters():
                w.copy_(base_param[name])
    def _end(self,model,index,base_params):
        base_param = {
            name: w.clone().detach() for name, w in model.named_parameters() #if w.requires_grad and "decode" not in name    
        }
        base_params[index] = base_param
    def _sync_weight(self,model,model_weights,base_param):
        _base_param = {name: w.clone().detach() for name, w in base_param.items()}
        # task 방향으로 1/tasks번 update 해야뎀 ! 중요 
        scale = 1/len(model_weights.items())
        for k,model_weight in model_weights.items():
            for name,w in model_weight.items():
                base_param[name].data -= scale*self.meta_outer_lr * (base_param[name].data - w.data)
        for name, w in model.named_parameters():
            w.data += base_param[name] - _base_param[name]
    def _fit_METATUNE(self, model,train_set, valid_set=None, valid_train_set=None,epoch=10):
        # Large-Scale Meta-Learning with Continual Trajectory Shifting (ICML21,jaewoong et al.)
        # https://github.com/JWoong148/ContinualTrajectoryShifting
        

        batch_size_tasks = self.meta_batch_size_tasks
        batch_size_per_task = self.meta_batch_size_per_task
        few_shot_number = self.few_shot_number

        print_per_batches = 100
        
        # Compute normalization vector over the whole dataset
        if self.fea_norm_vec is None:
            all_train_loader = SegmentDataLoader(
                train_set, self.batch_size, self.device, self.use_workload_embedding,
            )
            self.fea_norm_vec = all_train_loader.normalize()
            del all_train_loader
      
        if valid_set:
            for task in valid_set.tasks():
                self.register_new_task(task)
            valid_loader = SegmentDataLoader(valid_set, self.infer_batch_size, self.device, self.use_workload_embedding,
                                             self.use_target_embedding, self.target_id_dict,fea_norm_vec=self.fea_norm_vec)

        # Build dataloaders
        total_dataset_length = 0
        train_loaders = {}
        for task in train_set.features:
            task_dataset = train_set.extract_subset([task])
            
            train_loaders[task] = SegmentDataLoader(
                task_dataset, None, self.device, self.use_workload_embedding,
                fea_norm_vec=self.fea_norm_vec, shuffle=True,
            )
            total_dataset_length +=len(train_loaders[task])

        net = model.to(self.device)
        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.meta_outer_lr, weight_decay=self.wd
        )   
        

        task_list = list(train_set.tasks())
   
        net.train()


        avg_loss = None
        total_epoch = int(total_dataset_length*epoch/(32*batch_size_tasks))
        print(f"Task Batch {total_epoch}")
        # epoch 100 * 32 개가 전체 데이터셋 크기 
        # 너무 길어서 5으로 나눔 .. 
        # round(len(total dataset)*epoch / 32)

        for batch in range(total_epoch):
            tasks = random.choices(task_list, k=batch_size_tasks)
            net.zero_grad()
            base_param = {
                name: w.clone().detach() for name, w in net.named_parameters() 
            }
            model_weights = {i:None for i in range(len(tasks))}
            # outer loss
            
            for task_idx,task in enumerate(tasks):
                # with higher.innerloop_ctx(net, inner_optimiser, copy_initial_weights=False) as (fmodel, diffopt):
                    #아.. 각 task별로 loader 가 존재 .. ? 
                train_loader = train_loaders[task]
            

                train_segment_sizes, train_features, train_labels = train_loader.sample_batch(
                    few_shot_number
                )
               
                self._start(net,base_param)
                net.train()
                # independent하게 .. train 해야 
                loss = self.loss_func(
                        net(train_segment_sizes, train_features), train_labels
                    )

                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self._end(net,task_idx,model_weights)
                avg_loss = moving_average(avg_loss, loss.item())
            
            self._sync_weight(net,model_weights,base_param)
            

            if batch % print_per_batches == 0 :
                # validate
                valid_loss = self._validate(net, valid_loader)
                print(
                    "Task Batch: TRAIN %d/%d\t RMSE: %.4f\tValid RMSE: %.4f"
                    % (
                        batch,
                        total_epoch,
                        np.sqrt(avg_loss),
                        np.sqrt(valid_loss),
                    )
                )
                if self.wandb != None:
                    self.wandb.log({
                    "batch": batch,
                    "Train RMSE": np.sqrt(avg_loss),
                    "Valid RMSE":  np.sqrt(valid_loss)})
                 
          

        return net

    def _fine_tune_a_model(self, model, train_set, valid_set=None, verbose=1):
        if self.fine_tune_num_steps == 0:
            return model
        if verbose >= 1:
            print("=" * 60 + "\nFine-tune a net. Train size: %d" % len(train_set))

        # model.freeze_for_fine_tuning()

        train_loader = SegmentDataLoader(
            train_set, self.fine_tune_batch_size or len(train_set),
            self.device, self.use_workload_embedding, fea_norm_vec=self.fea_norm_vec,
        )

        if valid_set:
            valid_loader = SegmentDataLoader(valid_set, self.infer_batch_size,
                                             self.device, self.use_workload_embedding, fea_norm_vec=self.fea_norm_vec)

        tic = time.time()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.fine_tune_lr, weight_decay=self.fine_tune_wd)
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.fine_tune_lr, weight_decay=self.wd)
        for step in range(self.fine_tune_num_steps):
            # train
            model.train()
            train_loss = None
            for batch, (segment_sizes, features, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                loss = self.loss_func(model(segment_sizes, features), labels)
                loss.backward()
                optimizer.step()

                train_loss = moving_average(train_loss, loss.item())

            if verbose >= 1:
                if valid_set:
                    valid_loss = self._validate(model, valid_loader)
                else:
                    valid_loss = 0

                if self.loss_type == "rmse":
                    loss_msg = "Train RMSE: %.4f\tValid RMSE: %.4f" % (np.sqrt(train_loss), np.sqrt(valid_loss))
                elif self.loss_type in ["rankNetLoss", "lambdaRankLoss", "listNetLoss"]:
                    loss_msg = "Train Loss: %.4f\tValid Loss: %.4f" % (train_loss, valid_loss)
                print("Fine-tune step: %d\t%s\tTime: %.1f" % (step, loss_msg, time.time() - tic,))
                if self.wandb!=None:
                    if self.loss_type == "rmse":
                        self.wandb.log({
                        "Train RMSE": np.sqrt(train_loss),
                        "Valid RMSE": np.sqrt(valid_loss),
                        "Epoch": step,
                        "Time": (step, loss_msg, time.time() - tic,)})
                    else:
                        self.wandb.log({
                        "Train Loss": train_loss,
                        "Valid Loss": valid_loss,
                        "Epoch": step,
                        "Time": (step, loss_msg, time.time() - tic,)})
        return model

    def _validate(self, model, valid_loader):
        model.eval()
        valid_losses = []

        for segment_sizes, features, labels in valid_loader:
            preds = model(segment_sizes, features)
            valid_losses.append(self.loss_func(preds, labels).item())

        return np.mean(valid_losses)

    def _predict_a_dataset(self, model, dataset):
        ret = {}
        from copy import deepcopy
        
        
        
        for task, features in dataset.features.items():
            if self.args.maml and self.args.eval:
                base_model = deepcopy(model)
                length = int(len(features)*0.7)
                idx = np.arange(len(features))[:length]
                idx2 = np.arange(len(features))[length:]
                tmp_set = Dataset.create_one_task(task, features[idx2], np.zeros((len(features[idx2]),)))
                self._fine_tune_a_model(model,tmp_set, None,verbose=0)
                ret[task] = self._predict_a_task(base_model, task, features[idx])
            else:
                ret[task] = self._predict_a_task(model, task, features)
            
        return ret

    def _predict_a_task(self, model, task, features):
        if model is None:
            return np.zeros(len(features), dtype=np.float32)

        tmp_set = Dataset.create_one_task(task, features, np.zeros((len(features),)))

        preds = []
        for segment_sizes, features, labels in SegmentDataLoader(
                tmp_set, self.infer_batch_size, self.device,
                self.use_workload_embedding, self.use_target_embedding, self.target_id_dict, fea_norm_vec=self.fea_norm_vec,
        ):
            preds.append(model(segment_sizes, features))
        return torch.cat(preds).detach().cpu().numpy()

    def _fit_a_MAML_model(self, train_set, valid_set=None, valid_train_set=None):
        print("=" * 60 + "\nFit a MAML net. Train size: %d" % len(train_set))
        
        batch_size_tasks = self.meta_batch_size_tasks
        batch_size_per_task = self.meta_batch_size_per_task
        few_shot_number = self.few_shot_number

        print_per_batches = 100
        n_batches = 3000
        early_stop = 200
        
        # Compute normalization vector over the whole dataset
        if self.fea_norm_vec is None:
            all_train_loader = SegmentDataLoader(
                train_set, self.batch_size, self.device, self.use_workload_embedding,
            )
            self.fea_norm_vec = all_train_loader.normalize()
            del all_train_loader

        
      
        if valid_set:
            for task in valid_set.tasks():
                self.register_new_task(task)
            valid_loader = SegmentDataLoader(valid_set, self.infer_batch_size, self.device, self.use_workload_embedding,
                                             self.use_target_embedding, self.target_id_dict,fea_norm_vec=self.fea_norm_vec)

     
        # Build dataloaders
        train_loaders = {}
        for task in train_set.features:
            task_dataset = train_set.extract_subset([task])
            train_loaders[task] = SegmentDataLoader(
                task_dataset, None, self.device, self.use_workload_embedding,
                fea_norm_vec=self.fea_norm_vec, shuffle=True,
            )
            

        # Make network
        net = make_net(self.net_params).to(self.device)
        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.meta_outer_lr, weight_decay=self.wd
        )

        # Training
        avg_outer_loss = None
        avg_inner_loss = None
        task_list = list(train_set.tasks())
        best_batch = None
        best_train_loss = 1e10
        for batch in range(n_batches):
            tasks = random.choices(task_list, k=batch_size_tasks)
            net.train()
            outer_loss = torch.tensor(0.0, device=self.device)
            # outer loss
            for task in tasks:
                net.zero_grad()
                train_loader = train_loaders[task]

                train_segment_sizes, train_features, train_labels = train_loader.sample_batch(
                    few_shot_number
                )
                test_segment_sizes, test_features, test_labels = train_loader.sample_batch(
                    batch_size_per_task
                )

                # inner loss
                params = OrderedDict(net.meta_named_parameters())
                inner_loss = self.loss_func(
                    net(train_segment_sizes, train_features, params=params), train_labels
                )
                params = gradient_update_parameters(
                    net,
                    inner_loss,
                    params=params,
                    step_size=self.meta_inner_lr,
                    first_order=False,
                )

                 
                avg_inner_loss = moving_average(avg_inner_loss, inner_loss.item())

                # acculate gradient for meta-update
                outer_loss += self.loss_func(
                    net(test_segment_sizes, test_features, params=params), test_labels
                )

            optimizer.zero_grad()
            outer_loss /= len(tasks)
            outer_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
            optimizer.step()

            avg_outer_loss = moving_average(avg_outer_loss, outer_loss.item())

            if batch % print_per_batches == 0 or batch == n_batches - 1:
                # validate
                valid_loss = self._validate(net, valid_loader)
                print(
                    "Task Batch: %d\tOuter RMSE: %.4f\tInner RMSE: %.4f\tValid RMSE: %.4f"
                    % (
                        batch,
                        np.sqrt(avg_outer_loss),
                        np.sqrt(avg_inner_loss),
                        np.sqrt(valid_loss),
                    )
                )
                if self.wandb != None:
                    self.wandb.log({
                    "batch": batch,
                    "Outer RMSE": np.sqrt(avg_outer_loss),
                    "Inner RMSE":  np.sqrt(avg_inner_loss),
                    "Valid RMSE":  np.sqrt(valid_loss)})
                 
            # Early stop
            if avg_outer_loss < best_train_loss:
                best_train_loss = avg_outer_loss
                best_batch = batch
            elif batch - best_batch >= early_stop:
                print("Early stop. Best batch: %d" % best_batch)
                break

        return net

    def load(self, filename):
        if self.device == 'cpu':
            self.base_model, self.local_model, self.few_shot_learning, self.fea_norm_vec = \
                CPU_Unpickler(open(filename, 'rb')).load()
        else:
            self.base_model, self.local_model, self.few_shot_learning, self.fea_norm_vec = \
                pickle.load(open(filename, 'rb'))
            self.base_model = self.base_model.cuda() if self.base_model else None 
            self.local_model = self.local_model.cuda() if self.local_model else None

    def save(self, filename):
        base_model = self.base_model.cpu() if self.base_model else None 
        local_model = self.local_model.cpu() if self.local_model else None
        pickle.dump((base_model, local_model, self.few_shot_learning, self.fea_norm_vec),
                    open(filename, 'wb'))
        self.base_model = self.base_model.to(self.device) if self.base_model else None 
        self.local_model = self.local_model.to(self.device) if self.local_model else None

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


class MLPModel(PythonBasedModel):
    """The wrapper of MLPModelInternal. So we can use it in end-to-end search."""

    def __init__(self, few_shot_learning="base_only", disable_update=False):
        super().__init__()

        self.disable_update = disable_update
        self.model = MLPModelInternal(few_shot_learning=few_shot_learning)
        self.dataset = Dataset()

    def update(self, inputs, results):
        if self.disable_update or len(inputs) <= 0:
            return
        tic = time.time()
        self.dataset.update_from_measure_pairs(inputs, results)
        self.model.fit_base(self.dataset)
        logger.info("MLPModel Training time: %.2f s", time.time() - tic)

    def predict(self, task, states):
        features = get_per_store_features_from_states(states, task)
        if self.model is not None:
            learning_task = LearningTask(task.workload_key, str(task.target))
            eval_dataset = Dataset.create_one_task(learning_task, features, None)
            ret = self.model.predict(eval_dataset)[learning_task]
        else:
            ret = np.random.uniform(0, 1, (len(states),))

        # Predict 0 for invalid states that failed to be lowered.
        for idx, feature in enumerate(features):
            if feature.min() == feature.max() == 0:
                ret[idx] = float('-inf')

        return ret

    def update_from_file(self, file_name, n_lines=None):
        inputs, results = RecordReader(file_name).read_lines(n_lines)
        logger.info("MLPModel: Loaded %s measurement records from %s", len(inputs), file_name)
        self.update(inputs, results)

    def save(self, file_name: str):
        self.model.save(file_name)

    def load(self, file_name: str):
        if self.model is None:
            self.model = MLPModelInternal()
        self.model.load(file_name)
        self.num_warmup_sample = -1


def vec_to_pairwise_prob(vec):
    s_ij = vec - vec.unsqueeze(1)
    p_ij = 1 / (torch.exp(s_ij) + 1)
    return torch.triu(p_ij, diagonal=1)


class RankNetLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, labels):
        preds_prob = vec_to_pairwise_prob(preds)
        labels_prob = torch.triu((labels.unsqueeze(1) > labels).float(), diagonal=1)
        return torch.nn.functional.binary_cross_entropy(preds_prob, labels_prob)


class LambdaRankLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def lamdbaRank_scheme(self, G, D, *args):
        return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(
            G[:, :, None] - G[:, None, :])

    def forward(self, preds, labels, k=None, eps=1e-10, mu=10., sigma=1., device=None):
        if device is None:
            if torch.cuda.device_count():
                device = 'cuda:0'
            else:
                device = 'cpu'
        preds = preds[None, :]
        labels = labels[None, :]
        y_pred = preds.clone()
        y_true = labels.clone()

        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)
        ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
        ndcg_at_k_mask[:k, :k] = 1

        true_sorted_by_preds.clamp_(min=0.)
        y_true_sorted.clamp_(min=0.)

        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1. + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        weights = self.lamdbaRank_scheme(G, D, mu, true_sorted_by_preds)

        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
        scores_diffs[torch.isnan(scores_diffs)] = 0.
        weighted_probas = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
        losses = torch.log2(weighted_probas)
        masked_losses = losses[padded_pairs_mask & ndcg_at_k_mask]
        loss = -torch.sum(masked_losses)
        return loss


class ListNetLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, labels, eps=1e-10):
        preds = preds[None, :]
        labels = labels[None, :]
        y_pred = preds.clone()
        y_true = labels.clone()

        preds_smax = F.softmax(y_pred, dim=1)
        true_smax = F.softmax(y_true, dim=1)

        preds_smax = preds_smax + eps
        preds_log = torch.log(preds_smax)

        return torch.mean(-torch.sum(true_smax * preds_log, dim=1))
