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
import time
maml = [False,True]

loss = ['rmse','rankNetLoss','lambdaRankLoss']
model = ['mlp','transformer','oneshot']
lr = [5e-5]
wd = [5e-5]

class Worker(threading.Thread):
    def __init__(self, loss, model):
        super().__init__()
        self.loss = loss
        self.model = model
        self.mode = 1
        self.count = torch.cuda.device_count()
  
    def run(self):
        global idx
        sem.acquire()
        idx += 1
 
        
        text = f"docker run --ipc=host -it --gpus '\"device={idx%self.count}\"' --cpus 8 --rm  -v /home/jaehun/tenset_3090:/root/tvm -v /home/jaehun/tenset_3090:/root test:latest python3 /root/scripts/train_model.py \
        --mode {self.mode} --maml  --wandb   --use-gpu --loss {self.loss} --models {self.model}  >& log/{today}/{self.model}_{self.loss}_{self.mode}.log"
        proc = subprocess.Popen(text, shell=True, executable='/bin/bash')
        _ = proc.communicate()
        time.sleep(3)
        sem.release()


threads = []

for _loss in loss:
    for _model in model:
        thread = Worker(_loss,_model)
        thread.start()              # sub thread의 run 메서드를 호출
        threads.append(thread)

for thread in threads:
    thread.join()