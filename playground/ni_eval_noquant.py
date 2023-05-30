import copy
import torch
from src import AutoGPTQForCausalLM
from transformers import AutoTokenizer

base_model = 'facebook/opt-1.3b'
target_model = '.cache/models/opt-1.3b/word_semantics'

quantize_config = None

base_model  = AutoGPTQForCausalLM.from_pretrained(base_model, quantize_config)
target_model = AutoGPTQForCausalLM.from_pretrained(target_model, quantize_config)
original_target_model = copy.deepcopy(target_model)

for name1, param1 in base_model.named_parameters():
    target_model.state_dict()[name1] -= param1

base_model.requires_grad_(False)
with torch.no_grad():
    for name1, param1 in base_model.named_parameters():
        target_param = original_target_model.state_dict()[name1]
        delta_param = target_model.state_dict()[name1]
        assert torch.allclose(target_param, param1+delta_param, atol=1e-7)