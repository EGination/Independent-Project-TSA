import os
import random
from datasets import Dataset, concatenate_datasets
from util import load_from_checkpoint

CKPT_DIR = "./out/tsa_checkpoint"
TEMPLATE_ROOT = './augment_templates'

SYSTEM_PROMPT = """
你是一名精通目标情感分析（TSA）的数据科学家。在工作中，你必须严格遵守以下输出格式规范：

【全局输出格式规范】
你必须且只能输出一个标准的 JSON 对象，严禁包裹任何 Markdown 语法标记（如 ```json）。该对象必须严格包含中文键名，绝对不得出现任何英文键名。

【行为边界】
你只需输出符合上述规范的 JSON 数据，不得输出任何前导词、解释性文字或后续总结。确保 JSON 格式绝对合法。
"""

absa_cols = ['交通方便', '位于商圈附近', '是否容易寻找', '排队时间', '服务人员态度', '停车方便', '点菜/上菜速度', '价格', '性价比', '折扣力度', '装修', '嘈杂情况', '就餐空间', '卫生情况', '分量', '口味', '外观', '食物推荐程度']

def filter_invalid(dataset: Dataset):
    n_original = dataset.num_rows
    filtered = dataset.filter(lambda x: x['目标数量'] != -1, keep_in_memory=True, load_from_cache_file=False)

    n_invalid = n_original - filtered.num_rows
    print(f"Removed {n_invalid} invalid rows. {filtered.num_rows} rows remaining.")

    return filtered

def load_augment_templates(template_dir: str):
    templates = {"T1": [], "T2": [], "T3": [], "T4": []}
    for t_i in range(1, 5):
        for s_i in range(1, 5):
            curr_path = os.path.join(template_dir, f"t{t_i}_s{s_i}.txt")
            with open(curr_path, 'r', encoding='utf-8') as f:
                prompt = f.read()
                templates[f"T{t_i}"].append(prompt)

    return templates

def contrast_transform(example, idx):
    """
    Transform for T4
    """
    rng = random.Random(325 + idx)
    not_mentioned = [
        col for col, val in example.items()
        if val == -2
    ]

    if not not_mentioned:
        return example
    
    k = rng.randint(1, min(3, len(not_mentioned)))
    new_targets = rng.sample(not_mentioned, k)
    new_labels = ["未提及"] * len(new_targets)
    new_reasons = [f"评论未提及{col}。" for col in new_targets]

    orig_zip = list(zip(example["目标"], example["标签"], example["理由"]))

    insert_cnt = rng.randint(0, min(2, len(orig_zip)))
    chosen_inserts = rng.sample(orig_zip, insert_cnt)

    for target_val, label_val, reason_val in chosen_inserts:
        insert_idx = rng.randint(0, len(new_targets))
        new_targets.insert(insert_idx, target_val)
        new_labels.insert(insert_idx, str(label_val))
        new_reasons.insert(insert_idx, reason_val)

    example["目标"] = new_targets
    example["标签"] = new_labels
    example["理由"] = new_reasons

    return example

def task_map(dataset: Dataset, task:str):
    """
    T1 - Task1: Target Extraction Only
    T2 - Task2: Sentiment Analysis Only
    T3 - Task3: Vanilla
    T4 - Task4: Contrast Sentiment Analysis
    """
    task_to_rm_cols = {
        "T1": absa_cols + ['标签', '理由'],
        "T2": absa_cols, 
        "T3": absa_cols,
        "T4": absa_cols
    }
    if task == "T4":
        return dataset.map(contrast_transform, with_indices=True, keep_in_memory=True)
    remove_columns = task_to_rm_cols.get(task, absa_cols)
    return dataset.map(lambda _: {"任务": task}, remove_columns=remove_columns, keep_in_memory=True)

def inject(example, idx, templates):
    """
    Task 1: Comment -> Target;
    Task 2: Comment + Target -> Target + Label + Reason;
    Task 3: Comment -> Target + Label + Reason;
    Task 4: Comment + Target -> Target + Label + Reason;
    """
    task = example["任务"]
    rng = random.Random(325 + idx)
    chosen_template = rng.choice(templates[task])
    user_map = {
        "T1": chosen_template + example["评论"] + "输出JSON：",
        "T2": chosen_template + f"【待分析评论】\n评论：{example["评论"]}\n目标：{example["目标"]}" + "输出JSON：",
        "T3": chosen_template + example["评论"] + "输出JSON：",
        "T4": chosen_template + f"【待分析评论】\n评论：{example["评论"]}\n目标：{example["目标"]}" + "输出JSON：",
    }
    reply_map = {
        "T1": f"{{\"目标\": {example["目标"]}}}",
        "T2": f"{{\"目标\": {example['目标']}, \"标签\": {example['标签']}, \"理由\": {example["理由"]}}}",
        "T3": f"{{\"目标\": {example['目标']}, \"标签\": {example['标签']}, \"理由\": {example["理由"]}}}",
        "T4": f"{{\"目标\": {example['目标']}, \"标签\": {example['标签']}, \"理由\": {example["理由"]}}}",
    }

    conversation = [
        {
            "role": "system", 
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user", 
            "content": user_map[task]
        },
        {
            "role": "assistant", 
            "content": reply_map[task]
        }
    ]
    example["conversations"] = conversation

    return example

def task_augmentation(raw: Dataset, seed: int = 1018):
    """
    Perform split-based augmentation.
    """
    print("Task Augmentation.")

    dataset = raw.shuffle(seed=seed, keep_in_memory=True)

    # Task 3: Vanilla - 3200
    t3_pure = dataset.select(range(0, 3200))
    # Task 1: Target Extract - 900
    t1_pure = dataset.select(range(3200, 4100))
    # Task 2: Sentiment Analysis - 900
    t2_pure = dataset.select(range(4100, 5000))

    t3_pure = task_map(t3_pure, "T3")
    t1_pure = task_map(t1_pure, "T1")
    t2_pure = task_map(t2_pure, "T2")

    # Select 
    t3_reuse = t3_pure.select(range(0, 600))
    t1_from_t3 = t3_reuse.map(lambda _: {"任务": "T1"}, remove_columns=['标签', '理由'], keep_in_memory=True)
    t2_from_t3 = t3_reuse.map(lambda _: {"任务": "T2"}, keep_in_memory=True)

    t1_full = concatenate_datasets([t1_pure, t1_from_t3])
    t2_full = concatenate_datasets([t2_pure, t2_from_t3])

    t2_shuffled = t2_full.shuffle(seed=seed, keep_in_memory=True)
    # Task 4: Contractive Sentiment Analysis
    t4 = t2_shuffled.select(range(0, 800))
    t4_full = task_map(t4, "T4")

    res = concatenate_datasets([t1_full, t2_full, t3_pure, t4_full])
    res = res.shuffle(seed=seed, keep_in_memory=True)

    return res

def instruction_augmentation(raw: Dataset):
    templates = load_augment_templates(TEMPLATE_ROOT)
    print("Instruction Augmentation.")
    res = raw.map(
        lambda row, idx: inject(row, idx, templates),
        with_indices=True,
        keep_in_memory=True
    )
    return res


if __name__ == "__main__":
    print("augment.py")
    # Uncomment to enable dataset filtering.
    dataset = load_from_checkpoint(CKPT_DIR)
    filtered_dataset = filter_invalid(dataset)

    train = filtered_dataset.select(range(5000), keep_in_memory=True)
    # train = train.shuffle(seed=1018, keep_in_memory=True)

    # Old Augment Strategy
    # train = train.select(range(1700))

    # test = filtered_dataset.select(range(5000, 5100), keep_in_memory=True)
    # augment = filtered_dataset.select(range(5100, 5900), keep_in_memory=True)

    # augment = concatenate_datasets([train, augment])
    # augment = augment.shuffle(seed=1018, keep_in_memory=True)
    # augment = augment.map(lambda _: {'增强': 'S'}, keep_in_memory=True)

    task_aug = task_augmentation(train)
    inst_aug = instruction_augmentation(task_aug)
    inst_aug.save_to_disk('./out/augmented')
