import sys
import os
import torch
from transformers import AutoTokenizer, AutoModel

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from replace_llm_attention import patch_attention_layers

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b-32k", trust_remote_code=True)

model = AutoModel.from_pretrained("THUDM/chatglm2-6b-32k", trust_remote_code=True,force_download=True).half().cuda()
original_weights = model.state_dict()

model.config.use_cache = False

kwargs = {
    'attn_method': 'hyper', 
    'lsh_num_projs': 7,  
    'block_size': 32, 
    'sample_size': 64, 
    'min_seq_len': 4096,       
}
patch_attention_layers(model, model_name='chatglm2-6b-32k',patch_config='last',num_patch_layers=4, **kwargs)
model.load_state_dict(original_weights, strict=False)

model = model.eval()
response, history = model.chat(tokenizer, "Hello python", history=[])
print(response)
