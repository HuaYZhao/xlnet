# -*- coding: utf-8 -*-
# @Time    : 2020/7/16 10:08
# @Author  : zhaohuayang
# @email   : zhaohuayang@myhexin.com
import os
import tensorflow as tf
import threading
import shutil

TPU_NAMES = ['z1', 'z2', 'c1', ]

GS_ROOT = "gs://squad_cx/xlnet_data"
INIT_CKPT_DIR = f"xlnet_cased_L-24_H-1024_A-16"
SQUAD_DIR = f"squad_data"
GS_INIT_CKPT_DIR = f"{GS_ROOT}/{INIT_CKPT_DIR}"
GS_PROC_DATA_DIR = f"{GS_ROOT}/proc_data/squad"


class myThread(threading.Thread):
    def __init__(self, tpu_id, bs, lr):
        threading.Thread.__init__(self)
        self.tpu_id = tpu_id
        self.batch_size = bs
        self.lr = lr

    def run(self):
        per_tpu_run(self.tpu_id, self.batch_size, self.lr)


def per_tpu_run(tpu_id, bs, lr):
    for i in range(1):
        run_a_model(tpu_id, bs, lr, i)


def run_a_model(tpu_id, batch_size, lr, run_time):
    GS_MODEL_DIR = f"{GS_ROOT}/experiment/squad_large_{batch_size}_{lr}_{run_time}"

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
                  --predict_dir={GS_MODEL_DIR} \
                  --train_file={SQUAD_DIR}/train-v2.0.json \
                  --predict_file={SQUAD_DIR}/dev-v2.0.json \
                  --uncased=False \
                  --max_seq_length=512 \
                  --do_train=False \
                  --train_batch_size={batch_size} \
                  --do_predict=True \
                  --predict_batch_size=32 \
                  --learning_rate={lr} \
                  --adam_epsilon=1e-6 \
                  --iterations=1000 \
                  --save_steps=1000 \
                  --train_steps=8000 \
                  --warmup_steps=1000 
    """
    os.system(xargs)

    os.mkdir(f'squad_large_{batch_size}_{lr}_{run_time}')

    xargs = f"gsutil -m cp -r {GS_MODEL_DIR} gs://squad_cx/xlnet_args_train_models/"
    os.system(xargs)

    xargs = f"gsutil cp {GS_MODEL_DIR}/predictions.json squad_large_{batch_size}_{lr}_{run_time}/squad_preds.json"
    os.system(xargs)

    xargs = f"gsutil cp {GS_MODEL_DIR}/eval_all_nbest.pkl squad_large_{batch_size}_{lr}_{run_time}/eval_all_nbest.pkl"
    os.system(xargs)

    xargs = f"gsutil cp {GS_MODEL_DIR}/null_odds.json squad_large_{batch_size}_{lr}_{run_time}/squad_null_odds.json"
    os.system(xargs)

    xargs = f"gsutil -m cp -r squad_large_{batch_size}_{lr}_{run_time} gs://squad_cx/xlnet_args_train_results/"
    os.system(xargs)

    shutil.rmtree(f'squad_large_{batch_size}_{lr}_{run_time}')


if __name__ == '__main__':
    # 创建新线程
    thread1 = myThread(1, 48, 3e-5)
    # thread2 = myThread(2)
    # 开启新线程
    thread1.start()
    # thread2.start()

    thread1.join()
    # thread2.join()
    print("退出主线程")
