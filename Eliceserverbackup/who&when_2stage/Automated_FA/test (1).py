from Lib.local_model import _run_local_generation
from inference_local import _set_global_determinism
from transformers import AutoModelForCausalLM, AutoTokenizer

import hashlib

# 시드 고정
_set_global_determinism(42)

# Qwen 모델 로드
model_id = "Qwen/Qwen2.5-14B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto").to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_id)
client_or_model_obj = (model, tokenizer)

# 테스트 메시지
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "테스트용 문장을 하나 만들어줘."},
]

def _hash(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()

out1 = _run_local_generation(client_or_model_obj, messages, model_family="qwen")
out2 = _run_local_generation(client_or_model_obj, messages, model_family="qwen")

print(_hash(out1), _hash(out2), out1 == out2)
