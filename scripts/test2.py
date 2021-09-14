#!/usr/bin/env python
import subprocess
import os
from datetime import datetime
import torch
import threading
today = datetime.today().strftime('%Y-%m-%d')
sem = threading.Semaphore(torch.cuda.device_count())
idx = 0
os.makedirs(f"log/{today}", exist_ok=True)

maml = [False,True]

loss = ['rmse','rankNetLoss','lambdaRankLoss','listNetLoss']
model = ['mlp','transformer','lstm']
lr = [1e-2,1e-3,1e-4,1e-5,1e-6]
wd = [1e-2,1e-3,1e-4,1e-5,1e-6]

MAML = True
class Worker(threading.Thread):
    def __init__(self, loss, model,lr,wd):
        super().__init__()
        self.loss = loss
        self.model = model
        self.lr = lr
        self.wd = wd

    def run(self):
        global MAML
        global idx
        sem.acquire()
        idx += 1
 
        if MAML:
            text = f"docker run --ipc=host -it --gpus '\"device={idx%8}\"' --cpus 8 --rm  -v /home/jaehun/tenset:/root/tvm -v /home/jaehun/tenset:/root test:latest python3 /root/scripts/train_model.py \
             --maml --use-gpu --loss {self.loss} --models {self.model} --meta_outer_lr {self.lr} --meta_inner_lr {self.wd} >& log/{today}/MAML_{self.model}_{self.loss}_{self.lr}_{self.wd}.log"
        else:
            text = f"docker run --ipc=host -it --gpus '\"device={idx%8}\"' --cpus 8 --rm  -v /home/jaehun/tenset:/root/tvm -v /home/jaehun/tenset:/root test:latest python3 /root/scripts/train_model.py \
             --use-gpu --loss {self.loss} --models {self.model} --lr {self.lr} --wd {self.wd} >& log/{today}/{self.model}_{self.loss}_{self.lr}_{self.wd}.log"
        proc = subprocess.Popen(text, shell=True, executable='/bin/bash')
        _ = proc.communicate()
        sem.release()


threads = []

for _loss in loss:
    for _model in model:
        for _lr in lr:
            for _wd in wd:
                thread = Worker(_loss,_model,_lr,_wd)
                thread.start()              # sub thread의 run 메서드를 호출
                threads.append(thread)

for thread in threads:
    thread.join()