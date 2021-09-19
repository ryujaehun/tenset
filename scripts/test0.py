#!/usr/bin/env python
import subprocess
import os
from datetime import datetime
import torch
import threading
today = datetime.today().strftime('%Y-%m-%d')
count = 3#torch.cuda.device_count()
sem = threading.Semaphore(count)
idx = 0
os.makedirs(f"log/{today}", exist_ok=True)
import time

loss = ['rmse','lambdaRankLoss']
model = ['mlp','oneshot']


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
            text = f"docker run --ipc=host -it --gpus '\"device={idx%count}\"' --cpus 8 --rm  -v /home/jaehun/tenset/2080:/build -v /home/jaehun/tenset:/root/tvm -v /home/jaehun/tenset:/root test:latest python3 /root/scripts/train_model_0.py \
         --maml --wandb   --use-gpu --loss {self.loss} --models {self.model}  >& log/{today}/MAML_PRETRAIN_{self.model}_{self.loss}.log"
        else:
            text = f"docker run --ipc=host -it --gpus '\"device={idx%count}\"' --cpus 8 --rm  -v /home/jaehun/tenset/2080:/build -v /home/jaehun/tenset:/root/tvm -v /home/jaehun/tenset:/root test:latest python3 /root/scripts/train_model_0.py \
            --wandb   --use-gpu --loss {self.loss} --models {self.model}  >& log/{today}/PRETRAIN_{self.model}_{self.loss}.log"
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
# for baseline ... 
# thread = Worker('rmse','xgb')
# thread.start()              # sub thread의 run 메서드를 호출
# threads.append(thread)
# thread = Worker('rmse','random')
# thread.start()              # sub thread의 run 메서드를 호출
# threads.append(thread)
for _loss in loss:
    for _model in model:
        thread = Worker(_loss,_model)
        thread.start()              # sub thread의 run 메서드를 호출
        threads.append(thread)

for thread in threads:
    thread.join()