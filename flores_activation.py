import argparse
from types import MethodType
import torch
from vllm import LLM, SamplingParams
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model", type=str, default="/data/xkx/mllms/LLM-Research/Meta-Llama-3-8B")
# parser.add_argument("-m", "--model", type=str, default="/data/xkx/mllms/LLM-Research/Mistral-7B-Instruct-v0-3")
parser.add_argument("-l", "--lang", type=str, default="ar")
args = parser.parse_args()

is_llama = bool(args.model.lower().find('llama') >= 0)
is_mistral=bool(args.model.lower().find('mistral')>= 0)
model = LLM(model=args.model,enforce_eager=True,
            max_model_len=2048,trust_remote_code=True,enable_chunked_prefill=True,max_num_batched_tokens=8192)
max_length = model.llm_engine.model_config.max_model_len
num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
intermediate_size = model.llm_engine.model_config.hf_config.intermediate_size if (is_llama or is_mistral) else model.llm_engine.model_config.hf_config.ffn_hidden_size

over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')
sum1 = torch.zeros(num_layers, intermediate_size).to('cuda')
def factory(idx):
    def llama_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)  # b, l, 2i
        i = gate_up.size(-1)
        gate_up[ :, : i // 2] = torch.nn.SiLU()(gate_up[ :, : i // 2])
        activation = gate_up[:, : i // 2].float()

        sum1[idx, :] += activation.sum(dim=(0))
        over_zero[idx, :] += (activation > 0).sum(dim=(0))
        x = gate_up[ :, : i // 2] * gate_up[ :, i // 2 :]
        x, _ = self.down_proj(x)
        return x
    return llama_forward

for i in range(num_layers):
    if is_llama or is_mistral:
        obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp

    obj.forward = MethodType(factory(i), obj)

lang = args.lang
suffix = "llama" if is_llama else "mis"
ids = torch.load(f'data/id.{lang}.flores.{suffix}')
l = ids.size(0)
l = min(l, 6249984) // max_length * max_length
input_ids = ids[:l].reshape(-1, max_length)

_ = model.generate(prompt_token_ids=input_ids.tolist(), sampling_params=SamplingParams(max_tokens=1))
output = dict(n=l, over_zero=over_zero.to('cpu'),sum1=sum1.to('cpu'))

torch.save(output, f'activation/activation.{lang}.flores.{suffix}')
