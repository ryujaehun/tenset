#!/usr/bin/env python
import subprocess
import os
from datetime import datetime
#import torch
import threading
today = datetime.today().strftime('%Y-%m-%d')
count = 9#torch.cuda.device_count()
sem = threading.Semaphore(count)
idx = 0
os.makedirs(f"log/{today}", exist_ok=True)
import time

loss = ['rmse']
model = ['mlp']
device = [
' --dataset e5 ',
' --dataset e5 --dataset plat ',
' --dataset epyc ',
' --dataset epyc --dataset arm ',
' --dataset arm ',
' --dataset e5 --dataset k80 ',
' --dataset e5 --dataset epyc --dataset plat ',
' --dataset arm --dataset k80 ']

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
            text = f"docker run --ipc=host -it --gpus '\"device={idx%count}\"' --cpus 8 --rm  -v /home/ubuntu/tenset:/root/tvm -v /home/ubuntu/tenset:/root test:latest python3 /root/scripts/train_model_0.py \
         {self.device} --maml --wandb   --use-gpu --loss {self.loss} --models {self.model}  >& log/{today}/2_MAML_PRETRAIN_{self.model}_{self.loss}_{name}.log"
        else:
            text = f"docker run --ipc=host -it --gpus '\"device={idx%count}\"' --cpus 8 --rm  -v /home/ubuntu/tenset:/root/tvm -v /home/ubuntu/tenset:/root test:latest python3 /root/scripts/train_model_0.py \
            {self.device} --wandb   --use-gpu --loss {self.loss} --models {self.model}  >& log/{today}/2_PRETRAIN_{self.model}_{self.loss}_{name}.log"
        proc = subprocess.Popen(text, shell=True, executable='/bin/bash')
        _ = proc.communicate()
        time.sleep(3)
        sem.release()
# 1. many-shot learning 이 필요한 이유
#  - 기존 large-pre-trained model로는 inference time 에 unseen data에 대해서는 적절한 adaptation이 힘듬
#  - 반박 : 데이터를 샘플하여 만들면 시간만 많을 경우 찾을 수 있음. 
#  - search space의 크기가 너무 큼 ..? 혹은 cost model의 초기 성능이 search algorithm에 큰 영향을 미침 
#  - 반박 : search algorithm이 cost model의 초기 값에 영향을 많이 받는다는 것은 robust하게 설계하지 못하였다는 것
#  - 여기서 graph optimization과 cost optimization의 search space를 함침 
#  - 그럴시에 search space 가 비약적으로 커짐 10 ** 100 order를 넘음 \
#  - 탐색에 필요한 시간과 optimization plane 이 complex 하므로 정확한 cost model이 필요함.

# 실험 목적 
# 모든 데이터로 학습된 pretrain model 의 성능 평가 
# 일부 테스크 제외가 되었을 시에 모델의 성능 평가 
# 

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
