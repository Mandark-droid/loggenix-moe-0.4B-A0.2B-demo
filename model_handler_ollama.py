import requests
import json
import re
import time
from typing import Dict, Any, Optional, List

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama URL

# Default model (fallback if no model_name provided)
DEFAULT_MODEL_NAME = "hf.co/kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.1:Q8_0"

# Currently active model (updated dynamically)
_current_model_name = DEFAULT_MODEL_NAME

# Inference configurations
INFERENCE_CONFIGS = {
    "Optimized for Speed": {
        "num_predict": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "description": "Fast responses with limited output length"
    },
    "Middle-ground": {
        "num_predict": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "description": "Balanced performance and output quality"
    },
    "Full Capacity": {
        "num_predict": 4096,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "description": "Maximum output length with dynamic allocation"
    }
}


def get_inference_configs():
    """Get available inference configurations"""
    return INFERENCE_CONFIGS


def check_ollama_connection():
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def list_ollama_models():
    """List available models in Ollama"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        return []
    except requests.RequestException:
        return []


def load_model(model_name: str = None):
    """Check Ollama connection and model availability"""
    global _current_model_name

    if model_name:
        _current_model_name = model_name

    if not check_ollama_connection():
        raise ConnectionError(
            "Cannot connect to Ollama. Please make sure Ollama is running.\n"
            "Start Ollama with: ollama serve"
        )

    available_models = list_ollama_models()
    if _current_model_name not in available_models:
        print(f"Model '{_current_model_name}' not found in Ollama. Attempting to pull...")
        # Try to pull the model
        try:
            pull_response = requests.post(
                f"{OLLAMA_BASE_URL}/api/pull",
                json={"name": _current_model_name},
                timeout=600  # 10 min timeout for pulling
            )
            if pull_response.status_code == 200:
                print(f"Successfully pulled model: {_current_model_name}")
            else:
                print(f"Failed to pull model. Available models: {available_models}")
                return False
        except Exception as e:
            print(f"Error pulling model: {str(e)}")
            return False

    print(f"Using Ollama model: {_current_model_name}")
    return True


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
                    param_pairs = params_str.split(',')
                    for pair in param_pairs:
                        if ':' in pair:
                            key, value = pair.split(':', 1)
                            key = key.strip().strip('"\'')
                            value = value.strip().strip('"\'')
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


def call_ollama_api(messages: List[Dict], config: Dict, model_name: str = None, stream: bool = False) -> str:
    """
    Make a request to Ollama API

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        config: Configuration dictionary with inference parameters
        model_name: Model name to use (uses current model if not specified)
        stream: Whether to stream the response

    Returns:
        Generated response text
    """
    # Use provided model_name or fall back to current
    active_model = model_name or _current_model_name

    # Convert messages to prompt format expected by your model
    prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            prompt += f"System: {msg['content']}\n\n"
        elif msg["role"] == "user":
            prompt += f"User: {msg['content']}\n\n"
        elif msg["role"] == "assistant":
            prompt += f"Assistant: {msg['content']}\n\n"

    prompt += "Assistant: "

    payload = {
        "model": active_model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "num_predict": config.get("num_predict", 2048),
            "temperature": config.get("temperature", 0.7),
            "top_p": config.get("top_p", 0.9),
            "top_k": config.get("top_k", 40),
            "repeat_penalty": config.get("repeat_penalty", 1.1),
        }
    }

    try:
        if stream:
            return stream_ollama_response(payload)
        else:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=300  # 5 minutes timeout
            )
            response.raise_for_status()

            result = response.json()
            return result.get("response", "")

    except requests.RequestException as e:
        raise ConnectionError(f"Failed to connect to Ollama: {str(e)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid response from Ollama: {str(e)}")


def stream_ollama_response(payload: Dict) -> str:
    """Stream response from Ollama and return complete text"""
    full_response = ""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            stream=True,
            timeout=300
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    if 'response' in chunk:
                        token = chunk['response']
                        full_response += token
                        print(token, end='', flush=True)  # Print tokens as they come

                    if chunk.get('done', False):
                        break

                except json.JSONDecodeError:
                    continue

    except requests.RequestException as e:
        raise ConnectionError(f"Streaming failed: {str(e)}")

    print()  # New line after streaming
    return full_response


def generate_response(system_prompt: str, user_input: str, config_name: str = "Middle-ground",
                      stream: bool = False, model_name: str = None) -> str:
    """
    Generate response using Ollama API with the given system prompt and user input.

    Args:
        system_prompt: System instruction for the model
        user_input: User's input message
        config_name: Configuration preset to use
        stream: Whether to stream the response
        model_name: Optional model name to use (for dynamic checkpoint switching)

    Returns:
        Generated response text
    """
    # Load/check model with optional model_name
    if not load_model(model_name):
        return f"Error: Model '{model_name or _current_model_name}' not available in Ollama. Please wait while it downloads or check the model name."

    config = INFERENCE_CONFIGS[config_name]

    # Prepare messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    start_time = time.time()

    try:
        # Generate response using Ollama with dynamic model
        generated_response = call_ollama_api(messages, config, model_name=model_name, stream=stream)

        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.2f} seconds")

        # Process any tool calls in the generated response
        processed_response = process_tool_calls(generated_response)

        return processed_response

    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return f"Error: {str(e)}"


# Example usage and testing functions
def test_connection():
    """Test Ollama connection and model availability"""
    print("Testing Ollama connection...")

    if not check_ollama_connection():
        print("Cannot connect to Ollama")
        print("Make sure Ollama is running: ollama serve")
        return False

    print("Ollama is running")

    models = list_ollama_models()
    print(f"Available models: {models}")

    if MODEL_NAME not in models:
        print(f"Model '{MODEL_NAME}' not found")
        print(f"Pull the model with: ollama pull {MODEL_NAME}")
        return False

    print(f"Model '{MODEL_NAME}' is available")
    return True


def example_usage():
    """Example of how to use the system"""
    if not test_connection():
        return

    system_prompt = """You are a helpful AI assistant with access to tools. When you need to perform mathematical calculations, use the available tools by calling them in this format: [TOOL_CALL:calculate_numbers(operation="add", num1="10", num2="5")]

Available tools:
- calculate_numbers: Perform basic math operations (add, subtract, multiply, divide)
"""

    user_input = "What is 125 + 675? Please calculate this for me."

    print("Generating response...")
    response = generate_response(system_prompt, user_input, "Middle-ground", stream=True)
    print(f"\nFinal response: {response}")


if __name__ == "__main__":
    example_usage()
