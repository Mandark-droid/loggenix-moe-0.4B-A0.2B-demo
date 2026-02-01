import torch
import time
import gc
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Dict, Any, Optional

# Performance optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Global model and tokenizer variables
model = None
tokenizer = None
_current_model_id = None  # Track which model is currently loaded

# Default model configuration - Update this when switching checkpoints
# Note: All models/checkpoints are private, ensure HF_TOKEN is set
DEFAULT_MODEL_ID = "kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.1"

# Inference configurations
INFERENCE_CONFIGS = {
    "Optimized for Speed": {
        "max_new_tokens_base": 512,
        "max_new_tokens_cap": 512,
        "min_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "use_cache": False,
        "description": "Fast responses with limited output length"
    },
    "Middle-ground": {
        "max_new_tokens_base": 4096,
        "max_new_tokens_cap": 4096,
        "min_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "use_cache": False,
        "description": "Balanced performance and output quality"
    },
    "Full Capacity": {
        "max_new_tokens_base": 8192,
        "max_new_tokens_cap": 8192,
        "min_tokens": 1,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "use_cache": False,
        "description": "Maximum output length with dynamic allocation"
    }
}


def get_inference_configs():
    """Get available inference configurations"""
    return INFERENCE_CONFIGS



def load_model(model_name: str = None):
    """Load model and tokenizer with optimizations. Supports dynamic model switching."""
    global model, tokenizer, _current_model_id

    # Use provided model_name or fall back to default
    model_id = model_name if model_name else DEFAULT_MODEL_ID

    # Check if we need to reload (different model requested)
    if model is not None and tokenizer is not None and _current_model_id == model_id:
        return model, tokenizer

    # If switching models, cleanup old model first
    if model is not None and _current_model_id != model_id:
        print(f"Switching model from {_current_model_id} to {model_id}")
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = None
        tokenizer = None

    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    ## load 8 bit quants
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    # # Or 4-bit for even more memory savings
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    # )

    print(f"Loading model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.float16,  # Use half precision for speed
        attn_implementation="flash_attention_2" if hasattr(torch.nn, 'scaled_dot_product_attention') else None,
        use_cache=True,
        #quantization_config=quantization_config,
    ).eval()

    # Track which model is loaded
    _current_model_id = model_id

    # Enable gradient checkpointing if available
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    # Set pad_token_id
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Set padding side to left for better batching
    tokenizer.padding_side = "left"

    memory = model.get_memory_footprint() / 1e6
    print(f"Memory footprint: {memory:,.1f} MB")

    return model, tokenizer


# ===== TOOL DEFINITIONS =====

def calculate_numbers(operation: str, num1: float, num2: float) -> Dict[str, Any]:
    """
    Sample tool to perform basic mathematical operations on two numbers.

    Args:
        operation: The operation to perform ('add', 'subtract', 'multiply', 'divide')
        num1: First number
        num2: Second number

    Returns:
        Dictionary with result and operation details
    """
    try:
        num1, num2 = float(num1), float(num2)

        if operation.lower() == 'add':
            result = num1 + num2
        elif operation.lower() == 'subtract':
            result = num1 - num2
        elif operation.lower() == 'multiply':
            result = num1 * num2
        elif operation.lower() == 'divide':
            if num2 == 0:
                return {"error": "Division by zero is not allowed"}
            result = num1 / num2
        else:
            return {"error": f"Unknown operation: {operation}"}

        return {
            "result": result,
            "operation": operation,
            "operands": [num1, num2],
            "formatted": f"{num1} {operation} {num2} = {result}"
        }
    except ValueError as e:
        return {"error": f"Invalid number format: {str(e)}"}
    except Exception as e:
        return {"error": f"Calculation error: {str(e)}"}


# Tool registry
AVAILABLE_TOOLS = {
    "calculate_numbers": {
        "function": calculate_numbers,
        "description": "Perform basic mathematical operations (add, subtract, multiply, divide) on two numbers",
        "parameters": {
            "operation": "The mathematical operation to perform",
            "num1": "First number",
            "num2": "Second number"
        }
    }
}


def execute_tool_call(tool_name: str, **kwargs) -> Dict[str, Any]:
    """Execute a tool call with given parameters"""
    print(f"Executing tool: {tool_name} with parameters: {kwargs}")
    if tool_name not in AVAILABLE_TOOLS:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        tool_function = AVAILABLE_TOOLS[tool_name]["function"]
        result = tool_function(**kwargs)
        return {
            "tool_name": tool_name,
            "parameters": kwargs,
            "result": result
        }
    except Exception as e:
        print(f"Tool execution failed: {str(e)}")
        return {
            "tool_name": tool_name,
            "parameters": kwargs,
            "error": f"Tool execution error: {str(e)}"
        }


def parse_tool_calls(text: str) -> list:
    """
    Parse tool calls from model output.
    Supports both formats:
    - [TOOL_CALL:tool_name(param1=value1, param2=value2)]
    - <tool_call>{"name": "tool_name", "parameters": {"param1": "value1", "param2": "value2"}}</tool_call>
    """
    tool_calls = []

    # Pattern for both formats
    pattern = r'(\[TOOL_CALL:(\w+)\((.*?)\)\]|<tool_call>\s*{"name":\s*"(\w+)",\s*"parameters":\s*{([^}]*)}\s*}\s*</tool_call>)'
    matches = re.findall(pattern, text)
    print("Raw matches:", matches)

    for match in matches:
        full_match, old_tool_name, old_params, json_tool_name, json_params = match

        # Determine which format was matched
        if old_tool_name:  # Old format: [TOOL_CALL:tool_name(params)]
            tool_name = old_tool_name
            params_str = old_params
            original_call = f"[TOOL_CALL:{tool_name}({params_str})]"

            try:
                params = {}
                if params_str.strip():
                    param_pairs = params_str.split(',')
                    for pair in param_pairs:
                        if '=' in pair:
                            key, value = pair.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"\'')  # Remove quotes
                            params[key] = value

                tool_calls.append({
                    "tool_name": tool_name,
                    "parameters": params,
                    "original_call": original_call
                })

            except Exception as e:
                print(f"Error parsing old format tool call '{tool_name}({params_str})': {e}")
                continue

        elif json_tool_name:  # JSON format: <tool_call>...</tool_call>
            tool_name = json_tool_name
            params_str = json_params
            original_call = full_match

            try:
                params = {}
                if params_str.strip():
                    # Parse JSON-like parameters
                    # Handle the format: "operation": "add", "num1": "125", "num2": "675"
                    param_pairs = params_str.split(',')
                    for pair in param_pairs:
                        if ':' in pair:
                            key, value = pair.split(':', 1)
                            key = key.strip().strip('"\'')  # Remove quotes and whitespace
                            value = value.strip().strip('"\'')  # Remove quotes and whitespace
                            params[key] = value

                tool_calls.append({
                    "tool_name": tool_name,
                    "parameters": params,
                    "original_call": original_call
                })

            except Exception as e:
                print(f"Error parsing JSON format tool call '{tool_name}': {e}")
                continue

    return tool_calls

def process_tool_calls(text: str) -> str:
    """Process tool calls in the generated text and replace with results"""
    tool_calls = parse_tool_calls(text)

    if not tool_calls:
        return text

    processed_text = text

    for tool_call in tool_calls:
        tool_name = tool_call["tool_name"]
        parameters = tool_call["parameters"]
        original_call = tool_call["original_call"]

        try:
            # Validate parameters before execution
            if not isinstance(parameters, dict):
                raise ValueError(f"Invalid parameters for tool {tool_name}: {parameters}")

            # Execute tool
            result = execute_tool_call(tool_name, **parameters)

            # Create replacement text
            if "error" in result:
                replacement = f"[TOOL_ERROR: {result['error']}]"
            else:
                if "result" in result["result"]:
                    replacement = f"[TOOL_RESULT: {result['result']['formatted']}]"
                else:
                    replacement = f"[TOOL_RESULT: {result['result']}]"

            # Replace tool call with result
            processed_text = processed_text.replace(original_call, replacement)

        except Exception as e:
            print(f"Error processing tool call '{tool_name}': {e}")
            replacement = f"[TOOL_ERROR: Failed to process tool call: {str(e)}]"
            processed_text = processed_text.replace(original_call, replacement)

    return processed_text

def monitor_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

def generate_response(system_prompt: str, user_input: str, config_name: str = "Middle-ground", model_name: str = None) -> str:
    """
    Run inference with the given task (system prompt) and user input using the specified config.

    Args:
        system_prompt: System instruction for the model
        user_input: User's input message
        config_name: Configuration preset to use
        model_name: Optional HuggingFace model ID (for dynamic checkpoint switching)
    """
    load_model(model_name)

    config = INFERENCE_CONFIGS[config_name]

    input_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    prompt_text = tokenizer.apply_chat_template(
        input_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    input_length = len(tokenizer.encode(prompt_text))
    context_length = min(input_length, 3584)  # Leave room for generation

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=context_length,
        padding=False
    ).to(model.device)

    actual_input_length = inputs['input_ids'].shape[1]
    max_new_tokens = min(config["max_new_tokens_cap"], 4096 - actual_input_length - 10)
    max_new_tokens = max(config["min_tokens"], max_new_tokens)

    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            do_sample=config["do_sample"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            use_cache=config["use_cache"],
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # Memory optimizations
            output_attentions=False,
            output_hidden_states=False,
            return_dict_in_generate=False,
        )
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.2f} seconds")

        memory = model.get_memory_footprint() / 1e6
        monitor_memory()
        print(f"Memory footprint: {memory:,.1f} MB")

    # Clean up
    gc.collect()

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if prompt_text in full_text:
        response_start = full_text.find(prompt_text) + len(prompt_text)
        generated_response = full_text[response_start:].strip()
    else:
        # More robust fallback: try to extract response after the last user message
        generated_response = full_text.strip()
        try:
            # Look for common assistant/response indicators
            response_indicators = ["Assistant:", "<|assistant|>", "[/INST]", "Response:"]
            for indicator in response_indicators:
                if indicator in full_text:
                    parts = full_text.split(indicator)
                    if len(parts) > 1:
                        generated_response = parts[-1].strip()
                        break

            # If no indicator found, try to remove the input part
            user_message = user_input
            if user_message in full_text:
                parts = full_text.split(user_message)
                if len(parts) > 1:
                    generated_response = parts[-1].strip()
        except Exception:
            generated_response = full_text.strip()

    # Process any tool calls in the generated response
    generated_response = process_tool_calls(generated_response)
    return generated_response
