import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

base_path = './internlm2lianghua'
# download repo to the base_path directory using git
os.system('apt install git')
os.system('apt install git-lfs')
os.system(f'git clone https://code.openxlab.org.cn/JeffDing/internlm2lianghua.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

os.system(f'lmdeploy serve gradio {base_path}')
