#!/usr/bin/env python
import subprocess
import os
from datetime import datetime
import torch
import threading
today = datetime.today().strftime('%Y-%m-%d')
count = torch.cuda.device_count()
sem = threading.Semaphore(count)
idx = 0
os.makedirs(f"log/{today}", exist_ok=True)
import time
maml = [False,True]

loss = ['rmse','lambdaRankLoss']
model = ['mlp','oneshot']
lr = [5e-5]
wd = [5e-5]

MAML = [False,True]
class Worker(threading.Thread):
    def __init__(self, loss, model,maml):
        super().__init__()
        self.loss = loss
        self.model = model
        self.maml = maml
    def run(self):
        global idx
        sem.acquire()
        idx += 1
        if self.maml:
            text = f"docker run --ipc=host -it --gpus '\"device={idx%count}\"' --cpus 8 --rm  -v /home/jaehun/tenset/2080:/build -v /home/jaehun/tenset:/root/tvm -v /home/jaehun/tenset:/root test:latest python3 /root/scripts/train_model_1.py \
         --wandb --maml  --use-gpu --loss {self.loss} --models {self.model}  >& log/{today}/NEW_EXP_6_MAML_{self.model}_{self.loss}.log"
        else:
            text = f"docker run --ipc=host -it --gpus '\"device={idx%count}\"' --cpus 8 --rm  -v /home/jaehun/tenset/2080:/build -v /home/jaehun/tenset:/root/tvm -v /home/jaehun/tenset:/root test:latest python3 /root/scripts/train_model_1.py \
         --wandb   --use-gpu --loss {self.loss} --models {self.model}  >& log/{today}/NEW_EXP_6_{self.model}_{self.loss}.log"
        proc = subprocess.Popen(text, shell=True, executable='/bin/bash')
        _ = proc.communicate()
        time.sleep(3)
        sem.release()


threads = []
thread = Worker('rmse','xgb',False)
thread.start()              # sub thread의 run 메서드를 호출
threads.append(thread)
thread = Worker('rmse','random',False)
thread.start()              # sub thread의 run 메서드를 호출
threads.append(thread)
for _maml in maml:
    for _loss in loss:
        for _model in model:
            thread = Worker(_loss,_model,_maml)
            thread.start()              # sub thread의 run 메서드를 호출
            threads.append(thread)

for thread in threads:
    thread.join()