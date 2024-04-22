import torch
import os
from lmdeploy.serve.gradio.turbomind_coupled import run_local
from lmdeploy.messages import TurbomindEngineConfig

base_path = './internlm2lianghua'
# download repo to the base_path directory using git
os.system(f'git clone https://code.openxlab.org.cn/JeffDing/internlm2lianghua.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

backend_config = TurbomindEngineConfig(max_batch_size=8)
model_path = base_path
run_local(model_path, backend_config=backend_config, server_name="huggingface-space")
