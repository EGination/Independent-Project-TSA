import os
import shutil
from datasets import load_dataset, concatenate_datasets

from api import DeepSeekAPI
from util import load_from_checkpoint, add_tsa_columns

DATA_DIR = "./data"
TRAIN_DIR = os.path.join(DATA_DIR, "train.csv")
CKPT_DIR = "./out/tsa_checkpoint"
CURR_BATCH = 1
BATCH_SIZE = 100

LABEL_MAP = {
    'id': 'id',
    'review': '评论',
    'star': '评级',
    'Location#Transportation': '交通方便',
    'Location#Downtown': '位于商圈附近',
    'Location#Easy_to_find': '是否容易寻找',
    'Service#Queue': '排队时间',
    'Service#Hospitality': '服务人员态度',
    'Service#Parking': '停车方便',
    'Service#Timely': '点菜/上菜速度',
    'Price#Level': '价格',
    'Price#Cost_effective': '性价比',
    'Price#Discount': '折扣力度',
    'Ambience#Decoration': '装修',
    'Ambience#Noise': '嘈杂情况',
    'Ambience#Space': '就餐空间',
    'Ambience#Sanitary': '卫生情况',
    'Food#Portion': '分量',
    'Food#Taste': '口味',
    'Food#Appearance': '外观',
    'Food#Recommend': '食物推荐程度'
}

def main():
    print("Hello from workspace!")
    
    dataset = load_dataset("csv", data_files=TRAIN_DIR, split="train")
    dataset = dataset.rename_columns(LABEL_MAP)
    dataset = dataset.remove_columns(["id", "评级"])

    s, t = CURR_BATCH * BATCH_SIZE, (CURR_BATCH + 1) * BATCH_SIZE 
    batch = dataset.select(range(s, t))
    batch = batch.map(add_tsa_columns, load_from_cache_file=False)

    checkpoint = load_from_checkpoint(CKPT_DIR, ref_dataset=batch)
    checkpoint = concatenate_datasets([checkpoint, batch])
    # print(checkpoint[0], checkpoint[10])
    checkpoint.save_to_disk(CKPT_DIR + '_tmp')
    shutil.rmtree(CKPT_DIR, ignore_errors=True)
    os.rename(CKPT_DIR + '_tmp', CKPT_DIR)

    print(f"Batch {CURR_BATCH} completed. The current length of checkpoint is {len(checkpoint)}.")

if __name__ == "__main__":
    main()
