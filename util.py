import os
import json
from datasets import Dataset, load_from_disk

from api import DeepSeekAPI
os.environ['DEEPSEEK_API_KEY'] = 'abc123xyz'


api = DeepSeekAPI()

def load_from_checkpoint(ckpt_path: str, ref_dataset=None):
	"""
	Load the dataset from checkpoint if exists, 
	otherwise create an empty dataset with corresponding columns from ref_dataset.
	"""
	if os.path.exists(ckpt_path):
		dataset = load_from_disk(ckpt_path)
		print(f"Current length of the checkpoint is {len(dataset)}.")
	else:
		print("Initializing checkpoint...")
		dataset = Dataset.from_dict({col: [] for col in ref_dataset.column_names})

	return dataset

def add_tsa_columns(example: dict):
    comment = example["评论"]

    result = api.generate_tsa(comment)
    target_list = result.get("目标列表", [])

    targets = [e["目标"] for e in target_list]
    labels = [e["情感"] for e in target_list]
    reasons = [e["理由"] for e in target_list]

    example["目标"] = targets
    example["标签"] = labels
    example["理由"] = reasons
    example["目标数量"] = len(targets)
    
    return example

def batch_tsa():
	pass
