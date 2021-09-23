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

loss = ['lambdaRankLoss']
model = ['oneshot']
device = [' --dataset e5 ',' --dataset e5 --dataset plat ',' --dataset e5 --dataset plat --dataset epyc '\
    ,' --dataset e5 --dataset plat --dataset epyc --dataset arm',' --dataset e5 --dataset plat --dataset epyc --dataset arm --dataset k80 ',' --dataset e5 --dataset plat --dataset epyc --dataset arm --dataset k80 ',' --dataset e5 --dataset plat --dataset epyc --dataset arm --dataset k80 --dataset t4 ']

MAML = [False,True]
class Worker(threading.Thread):
    def __init__(self, loss, model,maml,device):
        super().__init__()
        self.loss = loss
        self.model = model
        self.maml = maml 
        self.dataset = device
    def run(self):
        global idx
        sem.acquire()
        idx += 1
        name = self.dataset.replace(' --dataset','').replace(' ','_')
        if self.maml:
            text = f"docker run --ipc=host -it --gpus '\"device={idx%count}\"' --cpus 8 --rm  -v /home/jaehun/tenset/2080:/build -v /home/jaehun/tenset:/root/tvm -v /home/jaehun/tenset:/root test:latest python3 /root/scripts/train_model_1.py \
          {self.dataset} --maml --wandb   --use-gpu --loss {self.loss} --models {self.model}  >& log/{today}/2_MAML_SMALL_{name}_{self.model}_{self.loss}.log"
        else:
            text = f"docker run --ipc=host -it --gpus '\"device={idx%count}\"' --cpus 8 --rm  -v /home/jaehun/tenset/2080:/build -v /home/jaehun/tenset:/root/tvm -v /home/jaehun/tenset:/root test:latest python3 /root/scripts/train_model_1.py \
            {self.dataset} --wandb   --use-gpu --loss {self.loss} --models {self.model}  >& log/{today}/2_SMALL_{name}_{self.model}_{self.loss}.log"
        proc = subprocess.Popen(text, shell=True, executable='/bin/bash')
        _ = proc.communicate()
        time.sleep(3)
        sem.release()


threads = []
for _loss in loss:
    for _model in model:
        for _dataset in device:
            for _maml in MAML:
                thread = Worker(_loss,_model,_maml,_dataset)
                thread.start()              # sub thread의 run 메서드를 호출
                threads.append(thread)

for thread in threads:
    thread.join()