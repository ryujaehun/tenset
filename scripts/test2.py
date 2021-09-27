#!/usr/bin/env python
import subprocess
import os
from datetime import datetime
#import torch
import threading
today = datetime.today().strftime('%Y-%m-%d')
count = 4#torch.cuda.device_count()
sem = threading.Semaphore(count)
idx = 0
os.makedirs(f"log/{today}", exist_ok=True)
import time

loss = ['rmse']
model = ['mlp']
device = [
' --dataset epyc ',
' --dataset arm ',
' --dataset e5 ',
' --dataset t4 ',
]

MAML = [True]
class Worker(threading.Thread):
    def __init__(self, loss, model,maml,device):
        super().__init__()
        self.loss = loss
        self.model = model
        self.device = device
        self.maml = maml 
        
    def run(self):
        
        global idx
        sem.acquire()
        idx += 1
        name = self.device.replace(' --dataset','').replace(' ','_')
        if self.maml:
            text = f"docker run --ipc=host -it --gpus '\"device={(idx%count)+10}\"' --cpus 8 --rm  -v /home/ubuntu/tenset:/root/tvm -v /home/ubuntu/tenset:/root test:latest python3 /root/scripts/train_model_0.py \
         {self.device} --maml --wandb   --use-gpu --loss {self.loss} --models {self.model}  >& log/{today}/fatal_resolve_MAML_PRETRAIN_{self.model}_{self.loss}_{name}.log"
        else:
            text = f"docker run --ipc=host -it --gpus '\"device={(idx%count)+10}\"' --cpus 8 --rm  -v /home/ubuntu/tenset:/root/tvm -v /home/ubuntu/tenset:/root test:latest python3 /root/scripts/train_model_0.py \
            {self.device} --wandb   --use-gpu --loss {self.loss} --models {self.model}  >& log/{today}/2_PRETRAIN_{self.model}_{self.loss}_{name}.log"
        proc = subprocess.Popen(text, shell=True, executable='/bin/bash')
        _ = proc.communicate()
        time.sleep(3)
        sem.release()

# 1. many-shot learning 이 필요한 이유
# 에러가 발생한 모델의 inference 재수행 ! 
# wandb에서 확인 후 수작업으로 수행 요망. 
# 4개의 gpu를 사용하며 10번 이후부터 사용
# condition
# 명명 규칙을 기존과 그대로 따라야함.
# model weight를 load해야함

threads = []

for _loss in loss:
    for _model in model:
        for _maml in MAML:
            for _device in device:
                thread = Worker(_loss,_model,_maml,_device)
                thread.start()              # sub thread의 run 메서드를 호출
                threads.append(thread)

for thread in threads:
    thread.join()
