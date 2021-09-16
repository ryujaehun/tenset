#!/usr/bin/env python
import subprocess
import os
from datetime import datetime
import torch
import threading
today = datetime.today().strftime('%Y-%m-%d')
sem = threading.Semaphore(torch.cuda.device_count()-2)
idx = 0
os.makedirs(f"log/{today}", exist_ok=True)
import time
device_type = ['gpu']
loss = ['rmse']
model = ['mlp','transformer','oneshot']
maml = [1,2]

class Worker(threading.Thread):
    def __init__(self,_type,_model,_mode):
        super().__init__()
        self.type = _type
        self.model = _model
        self.mode = _mode
        self.loss = 'rmse'

    def run(self):
        global idx
        sem.acquire()
        idx += 1
      
        if self.mode != -1:
            text = f"docker run --ipc=host -it --gpus '\"device={idx%8-2}\"' --cpus 8 --rm  -v /home/jaehun/tenset:/root/tvm -v /home/jaehun/tenset:/root test:latest python3 /root/scripts/train_model_for_{self.type}.py \
            --wandb --mode {self.mode} --maml --use-gpu --loss {self.loss} --models {self.model}  >& log/{today}/CROSS_MAML_{self.type}_{self.model}_{self.mode}.log"
        else:
            text = f"docker run --ipc=host -it --gpus '\"device={idx%8-2}\"' --cpus 8 --rm  -v /home/jaehun/tenset:/root/tvm -v /home/jaehun/tenset:/root test:latest python3 /root/scripts/train_model_for_{self.type}.py \
             --wandb --use-gpu --loss {self.loss} --models {self.model}  >& log/{today}/CROSS_{self.type}_{self.model}.log"
        proc = subprocess.Popen(text, shell=True, executable='/bin/bash')
        _ = proc.communicate()
        time.sleep(3)
        sem.release()


threads = []

for _type in device_type:
    for _model in model:
        for _mode in maml:
            thread = Worker(_type,_model,_mode)
            thread.start()              # sub thread의 run 메서드를 호출
            threads.append(thread)

for thread in threads:
    thread.join()