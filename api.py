import os
import json
import yaml
from openai import OpenAI

with open('./configs.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

DIR_SYS_PROMPT = config['dir']['prompt']['tsa-gen']

class DeepSeekAPI:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com")

    def get_prompt(self, prompt_dir):
        with open(prompt_dir, 'r', encoding='utf-8') as f:
            prompt = f.read()
        return prompt
    
    def generate_tsa(self, comment: str) -> str:
        system_prompt = self.get_prompt(DIR_SYS_PROMPT)

        response = self.client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": comment},
            ],
            stream=False,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
            response_format={
                'type': 'json_object'
            },
        )
        return json.loads(response.choices[0].message.content)
