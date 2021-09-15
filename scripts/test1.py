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
lr = [5e-5]
wd = [5e-5]

MAML = False
class Worker(threading.Thread):
    def __init__(self, loss, model):
        super().__init__()
        self.loss = loss
        self.model = model
  

    def run(self):
        global MAML
        global idx
        sem.acquire()
        idx += 1
 
        
        text = f"docker run --ipc=host -it --gpus '\"device={idx%8}\"' --cpus 8 --rm  -v /home/jaehun/tenset:/root/tvm -v /home/jaehun/tenset:/root test:latest python3 /root/scripts/train_model.py \
         --wandb   --use-gpu --loss {self.loss} --models {self.model}  >& log/{today}/{self.model}_{self.loss}.log"
        proc = subprocess.Popen(text, shell=True, executable='/bin/bash')
        _ = proc.communicate()
        sem.release()


threads = []
thread = Worker('rmse','xgb')
thread.start()              # sub thread의 run 메서드를 호출
threads.append(thread)
for _loss in loss:
    for _model in model:
        thread = Worker(_loss,_model)
        thread.start()              # sub thread의 run 메서드를 호출
        threads.append(thread)

for thread in threads:
    thread.join()