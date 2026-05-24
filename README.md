# Single-Pass TSA on Chinese Restaurant Reviews via Instruction-Tuned LLMs

This repository contains the official implementation, data transformation pipelines, and evaluation scripts for the paper: **"Single-Pass Targeted Sentiment Analysis on Chinese Restaurant Reviews via Instruction-Tuned LLMs"**.

We introduce a unified, single-pass inference framework utilizing instruction-tuned Large Language Models (LLMs) to simultaneously extract dynamic targets, sentiments, and supporting reasons from Chinese restaurant reviews in a structured format. To combat domain data scarcity and combinatorial overfitting, we develop a multi-dimensional data augmentation taxonomy covering contrastive, instruction, and task-level strategies.

---

## Key Features

- **Single-Pass Joint Extraction**: Extracts `(Target, Sentiment, Reason)` triplets simultaneously in a single forward pass, eliminating cascading errors common in traditional multi-stage pipelines.
- **Multi-Dimensional Data Augmentation**: Includes structured task-level regularizations, negative contrastive sampling, and prompt diversification to harden model decision boundaries.
- **Quantized LoRA Fine-Tuning**: Orchestrated with [Unsloth](https://github.com/unslothai/unsloth) to enable efficient 4-bit quantized training (`PEFT/LoRA`) on consumer-grade GPUs.
- **Deterministic Structural Parsing**: A custom defensive JSON-parsing layer that gracefully intercepts truncated or partially malformed LLM outputs.

---

## Repository Structure

Placeholder below, TBD
```text
├── data/
│   ├── raw_asap/               # Original ASAP dataset files
│   ├── transformed_6k.json    # Transformed dataset via DeepSeek API (6,200 samples)
│   └── templates/             # Task templates (T1: Target, T2: Sentiment, T4: Contrastive)
├── src/
│   ├── augment.py             # Data augmentation taxonomy pipeline
│   ├── train.py               # 4-bit LoRA training loop script (via Unsloth)
│   ├── evaluate.py            # Strict verification and Joint F1 matching logic
│   └── parser.py              # Defensive regular expression JSON parser
├── configs/
│   └── qwen_lora_config.json  # Hyperparameters for Qwen PEFT fine-tuning
├── README.md
└── requirements.txt

## Acknowledgement & Citation

This project is built upon the text corpora from the [ASAP dataset](https://github.com/Meituan-Dianping/asap), a benchmark for aspect-level sentiment analysis in Chinese restaurant reviews. We sincerely thank the original authors for their open-source contribution to the community. 

If you find this repository or our transformed dataset helpful, please consider citing the original ASAP paper as follows:

```
@inproceedings{bu-etal-2021-asap,
    title = "{ASAP}: A {C}hinese Review Dataset Towards Aspect Category Sentiment Analysis and Rating Prediction",
    author = "Bu, Jiahao  and
      Ren, Lei  and
      Zheng, Shuang  and
      Yang, Yang  and
      Wang, Jingang  and
      Zhang, Fuzheng  and
      Wu, Wei",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.167",
    pages = "2069--2079"
}
```
