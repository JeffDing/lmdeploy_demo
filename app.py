import os
import gradio as gr
from lmdeploy import pipeline, TurbomindEngineConfig,GenerationConfig

backend_config = TurbomindEngineConfig(cache_max_entry_count=1,model_format="awq")

# download internlm2 to the base_path directory using git tool
base_path = './internlmlianghua'
os.system(f'git clone https://code.openxlab.org.cn/JeffDing/internlmlianghua.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

model_name = 'internlm2lianghua'
pipe = pipeline(base_path,model_name, backend_config=backend_config)
gen_config = GenerationConfig(top_p=0.8,top_k=40,temperature=0.8,max_new_tokens=1024)

def chat(message,history):
  response = pipe(message, gen_config = gen_config)
  return response.text

demo = gr.ChatInterface(
  fn = chat,
  title="InternLM2-Chat Demo",
  description="""InternLM is mainly developed by Shanghai AI Laboratory. """,
)
demo.queue().launch()
