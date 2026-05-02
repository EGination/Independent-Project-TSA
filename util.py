import json

from api import DeepSeekAPI

TEST_JSON = json.loads("""
{
  "目标列表": [
    {
      "目标": "目标名称",
      "情感": 1,
      "理由": "一句通顺完整的因果解释，说明为何该实体被赋予该情感倾向，需综合原文内容和推理过程"
    }
  ]
}
""")

api = DeepSeekAPI()

def add_tsa_columns(example):
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
