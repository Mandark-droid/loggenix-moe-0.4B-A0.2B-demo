---
title: Loggenix MoE 0.4B-A0.2B Demo
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
app_port: 7860
---

# Loggenix MoE 0.4B-A0.2B Demo

Interactive demo for the **Loggenix MoE 0.4B-A0.2B** Mixture of Experts model.

## Model

**Model ID**: `kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.1`

### Architecture
- **Type**: Mixture of Experts (MoE)
- **Total Parameters**: 0.4B (400M)
- **Active Parameters**: 0.2B (200M)
- **Context Length**: 8192 tokens
- **Precision**: FP16

### Formats Available
- HuggingFace Transformers
- GGUF (via Ollama)

## Features

### Inference Testing
- Interactive chat interface
- Multiple inference configurations (Speed/Balanced/Full Capacity)
- Task-based system prompts
- Tool calling support

### Response Flagging
- Flag problematic responses for quality monitoring
- Automatic persistence to HuggingFace Hub dataset
- Tracks model version and checkpoint for analysis

### Evaluation
- Browse evaluation datasets
- Compare expected vs actual model outputs
- Benchmark visualizations (MMLU, HellaSwag, PIQA, ARC, WinoGrande)

## Datasets

### Flagged Responses
`kshitijthakkar/loggenix-0.4B-flagged-responses` (private)

### Evaluation Datasets
- `kshitijthakkar/loggenix-synthetic-ai-tasks-eval-with-outputs`
- `kshitijthakkar/loggenix-synthetic-ai-tasks-eval_v5-with-outputs`
- `kshitijthakkar/loggenix-synthetic-ai-tasks-eval_v6-with-outputs`

## Usage

### HuggingFace Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
).eval()

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Hello!"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Ollama (GGUF)
```bash
ollama pull hf.co/kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.1-GGUF:Q8_0
ollama run hf.co/kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.1-GGUF:Q8_0
```

## Development

### Local Setup
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python enhanced_app.py
```

### Docker
```bash
docker build -t loggenix-demo .
docker run -p 7860:7860 loggenix-demo
```

## Author

Kshitij Thakkar

## License

Apache 2.0
