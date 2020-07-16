# -*- coding: utf-8 -*-
# @Time    : 2020/7/16 10:08
# @Author  : zhaohuayang
# @email   : zhaohuayang@myhexin.com
import os
import tensorflow as tf
import threading

TPU_NAMES = ['z1', 'z2', 'c1', ]

GS_ROOT = "gs://squad_cx/xlnet_data"
INIT_CKPT_DIR = "xlnet_cased_L-24_H-1024_A-16"
SQUAD_DIR = f"{GS_ROOT}/squad_data"
GS_INIT_CKPT_DIR = f"{GS_ROOT}/{INIT_CKPT_DIR}"
GS_PROC_DATA_DIR = f"{GS_ROOT}/proc_data/squad"


class myThread(threading.Thread):
    def __init__(self, tpu_id):
        threading.Thread.__init__(self)
        self.tpu_id = tpu_id

    def run(self):
        per_tpu_run(self.tpu_id)


def per_tpu_run(tpu_id):
    run_a_model(tpu_id)


def run_a_model(tpu_id):
    GS_MODEL_DIR = f"{GS_ROOT}/experiment/squad_{tpu_id}"

    xargs = f"""
            python3 run_squad.py \
                  --use_tpu=True \
                  --tpu={TPU_NAMES[tpu_id - 1]} \
                  --num_hosts=1 \
                  --num_core_per_host=8 \
                  --model_config_path={INIT_CKPT_DIR}/xlnet_config.json \
                  --spiece_model_file={INIT_CKPT_DIR}/spiece.model \
                  --output_dir={GS_PROC_DATA_DIR} \
                  --init_checkpoint={GS_INIT_CKPT_DIR}/xlnet_model.ckpt \
                  --model_dir={GS_MODEL_DIR} \
                  --train_file={SQUAD_DIR}/train-v2.0.json \
                  --predict_file={SQUAD_DIR}/dev-v2.0.json \
                  --uncased=False \
                  --max_seq_length=512 \
                  --do_train=True \
                  --train_batch_size=48 \
                  --do_predict=True \
                  --predict_batch_size=32 \
                  --learning_rate=3e-5 \
                  --adam_epsilon=1e-6 \
                  --iterations=1000 \
                  --save_steps=1000 \
                  --train_steps=8000 \
                  --warmup_steps=1000 
    """
    os.system(xargs)


if __name__ == '__main__':
    # 创建新线程
    thread1 = myThread(1)
    # thread2 = myThread(2)
    # 开启新线程
    thread1.start()
    # thread2.start()

    thread1.join()
    # thread2.join()
    print("退出主线程")
