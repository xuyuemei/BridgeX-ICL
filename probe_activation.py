import argparse
from types import MethodType
import torch
from vllm import LLM, SamplingParams
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="/data/xkx/mllms/LLM-Research/Meta-Llama-3-8B")
#parser.add_argument("-m", "--model", type=str, default="/data/xkx/mllms/LLM-Research/Mistral-7B-Instruct-v0-3")
parser.add_argument("--src_lang", type=str, default="id")
parser.add_argument("--trg_lang", type=str, default="he")
args = parser.parse_args()

lang_dic = {
    "en": "English", "de": "Deutsch", "sw": "kiswahili", 'zh': '中文', 'ja': '日本語',
    'tl': 'Tagalog', 'he': 'עִברִית', 'ar': 'عربي', 'id': 'Indonesia',
    'pt': 'Português', 'fr': 'Français', 'it': 'Italiano', 'es': 'Español'
}
src_lang = lang_dic[args.src_lang]
trg_lang = lang_dic[args.trg_lang]

with open(f'./LLaMA3/gold_bli/{args.src_lang}-{args.trg_lang}.txt', 'r', encoding='utf-8') as f:
    pairs = [line.strip().split() for line in f if line.strip()]

# Construct prompts for word translation
samples = []
for src_word, trg_word in pairs:
    samples.append(f'{src_lang}:"{src_word}"→{trg_lang}:')
print(f"samples: {len(samples)}")

# model = LLM(model=args.model, enforce_eager=True,
#             max_model_len=2048, trust_remote_code=True,
#             enable_chunked_prefill=True, max_num_batched_tokens=8192)
model = LLM(model=args.model, dtype="float16", max_model_len=4096)

num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
intermediate_size = model.llm_engine.model_config.hf_config.intermediate_size
print(f"num_layers = {num_layers}, intermediate_size = {intermediate_size}")


activation_freq = torch.zeros(num_layers, intermediate_size, len(samples)).to("cpu")
layer_token_counts = torch.zeros(num_layers, len(samples)).to("cpu")

def factory(idx, sample_idx):
    def custom_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        i = gate_up.size(-1)
        gate = torch.nn.functional.silu(gate_up[:, :i // 2])
        activation_sum = gate.sum(dim=0)
        activation_freq[idx, :, sample_idx] += activation_sum.cpu()
        layer_token_counts[idx, sample_idx] = gate.shape[0]
        x = gate * gate_up[:, i // 2:]
        x, _ = self.down_proj(x)
        return x
    return custom_forward

sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=30,  stop=["\n"])

for sample_idx, prompt in enumerate(samples):
    print(f"[{sample_idx+1}/{len(samples)}] Processing: {prompt}")
    for i in range(num_layers):
        obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp
        obj.forward = MethodType(factory(i, sample_idx), obj)
    _ = model.generate(prompts=[prompt], sampling_params=sampling_params)

#Normalize activation by token counts
activation_freq_normalized = torch.zeros_like(activation_freq)
for layer in range(num_layers):
    activation_freq_normalized[layer] = activation_freq[layer] / (layer_token_counts[layer].view(1, -1) + 1e-10)


save_path = f"try_bli/activation_value_{args.src_lang}_{args.trg_lang}_llama.pt"
torch.save(activation_freq_normalized, save_path)
