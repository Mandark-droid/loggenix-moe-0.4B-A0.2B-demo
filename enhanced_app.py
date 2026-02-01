import gradio as gr
import pandas as pd
from datasets import load_dataset, Dataset
import plotly.graph_objects as go
import datetime
import json
import random
import os
# Import both inference backends
from model_handler_ollama import generate_response as ollama_generate, get_inference_configs as ollama_configs
from model_handler import generate_response as transformers_generate, get_inference_configs as transformers_configs

def get_inference_configs():
    """Get inference configs (using Ollama configs as default)"""
    return ollama_configs()
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
from huggingface_hub import HfApi, hf_hub_download, upload_file
import tempfile

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

# ===== HUGGINGFACE DATASET FLAGGING CONFIGURATION =====
# Since HuggingFace Spaces are stateless Docker containers, we push flagged responses
# to a HuggingFace dataset for persistence and later analysis/refinement
FLAGGED_RESPONSES_DATASET = "kshitijthakkar/loggenix-0.4B-flagged-responses"
HF_API = HfApi()

# ===== CHECKPOINT REGISTRY =====
# Each checkpoint has its own model ID, GGUF path, and benchmark results
# Add new checkpoints here - they will appear in the UI dropdown
# Available quantization options
QUANTIZATION_OPTIONS = {
    "Q8_0": {"display_name": "Q8_0 (8-bit, balanced)", "description": "8-bit quantization - good balance of size and quality"},
    "f16": {"display_name": "F16 (16-bit, higher quality)", "description": "16-bit float - larger but higher quality"},
}
DEFAULT_QUANTIZATION = "f16"

CHECKPOINT_REGISTRY = {
    "s3.1": {
        "display_name": "v3.1 (SFT Stage 3.1)",
        "hf_model_id": "kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.1",
        "hf_repo_base": "hf.co/kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.1",
        "available_quantizations": ["Q8_0", "f16"],
        "benchmarks": {
            "MMLU": 25.26, "HellaSwag": 40.0, "PIQA": 70.0, "ARC": 10.0, "WinoGrande": 60.0,
            "BoolQ": 40.0, "OpenBookQA": 40.0, "GSM8K": 0.0,
            "synthetic_mean": 21.68, "tool_calling": 50.0, "programming": 25.88,
        },
        "eval_date": "2026-01-28",
    },
    "s3.2": {
        "display_name": "v3.2 (SFT Stage 3.2)",
        "hf_model_id": "kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.2",
        "hf_repo_base": "hf.co/kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.2",
        "available_quantizations": ["Q8_0", "f16"],
        "benchmarks": {
            "MMLU": 27.19, "HellaSwag": 40.0, "PIQA": 70.0, "ARC": 10.0, "WinoGrande": 50.0,
            "BoolQ": 30.0, "OpenBookQA": 40.0, "GSM8K": 0.0,
            "synthetic_mean": 22.0, "tool_calling": 30.0, "programming": 0.0,
        },
        "eval_date": "2026-01-28",
    },
    "s3.3": {
        "display_name": "v3.3 (SFT Stage 3.3)",
        "hf_model_id": "kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.3",
        "hf_repo_base": "hf.co/kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.3",
        "available_quantizations": ["Q8_0", "f16"],
        "benchmarks": {
            "MMLU": 26.0, "HellaSwag": 38.0, "PIQA": 68.0, "ARC": 12.0, "WinoGrande": 52.0,
            "BoolQ": 32.0, "OpenBookQA": 38.0, "GSM8K": 0.0,
            "synthetic_mean": 25.0, "tool_calling": 75.0, "programming": 0.0,
        },
        "eval_date": "2026-01-26",
    },
    "s3.4": {
        "display_name": "v3.4 (SFT Stage 3.4)",
        "hf_model_id": "kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.4",
        "hf_repo_base": "hf.co/kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.4",
        "available_quantizations": ["Q8_0", "f16"],
        "benchmarks": {
            "MMLU": 0.0, "HellaSwag": 0.0, "PIQA": 0.0, "ARC": 0.0, "WinoGrande": 0.0,
        },
        "eval_date": None,
    },
    "s3.5": {
        "display_name": "v3.5 (SFT Stage 3.5)",
        "hf_model_id": "kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.5",
        "hf_repo_base": "hf.co/kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.5",
        "available_quantizations": ["Q8_0", "f16"],
        "benchmarks": {
            "MMLU": 0.0, "HellaSwag": 0.0, "PIQA": 0.0, "ARC": 0.0, "WinoGrande": 0.0,
        },
        "eval_date": None,
    },
    "s2.5": {
        "display_name": "v2.5 (SFT Stage 2.5)",
        "hf_model_id": "kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s2.5",
        "hf_repo_base": "hf.co/kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s2.5",
        "available_quantizations": ["Q8_0", "f16"],
        "benchmarks": {
            "MMLU": 23.4, "HellaSwag": 45.0, "PIQA": 60.0, "ARC": 20.0, "WinoGrande": 55.0,
            "BoolQ": 30.0, "OpenBookQA": 30.0, "GSM8K": 0.0,
            "synthetic_mean": 30.0, "tool_calling": 30.0, "programming": 0.0,
        },
        "eval_date": "2026-01-19",
    },
    "s2.1": {
        "display_name": "v2.1 (SFT Stage 2.1)",
        "hf_model_id": "kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s2.1",
        "hf_repo_base": "hf.co/kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s2.1",
        "available_quantizations": ["Q8_0", "f16"],
        "benchmarks": {
            "MMLU": 0.0, "HellaSwag": 0.0, "PIQA": 0.0, "ARC": 0.0, "WinoGrande": 0.0,
        },
        "eval_date": None,
    },
    "s2.0": {
        "display_name": "v2.0 (SFT Stage 2.0)",
        "hf_model_id": "kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s2.0",
        "hf_repo_base": "hf.co/kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s2.0",
        "available_quantizations": ["Q8_0", "f16"],
        "benchmarks": {
            "MMLU": 0.0, "HellaSwag": 0.0, "PIQA": 0.0, "ARC": 0.0, "WinoGrande": 0.0,
        },
        "eval_date": None,
    },
}

# Default checkpoint
DEFAULT_CHECKPOINT = "s3.1"

# Current active checkpoint (will be updated by UI)
CURRENT_CHECKPOINT_KEY = DEFAULT_CHECKPOINT


def get_checkpoint_choices():
    """Get list of checkpoint choices for dropdown"""
    return [(v["display_name"], k) for k, v in CHECKPOINT_REGISTRY.items()]


def get_quantization_choices(checkpoint_key=None):
    """Get available quantization options for a checkpoint"""
    if checkpoint_key and checkpoint_key in CHECKPOINT_REGISTRY:
        available = CHECKPOINT_REGISTRY[checkpoint_key].get("available_quantizations", ["Q8_0"])
    else:
        available = list(QUANTIZATION_OPTIONS.keys())
    return [(QUANTIZATION_OPTIONS[q]["display_name"], q) for q in available if q in QUANTIZATION_OPTIONS]


def build_model_name(checkpoint_key, quantization):
    """Build the full Ollama model name from checkpoint and quantization"""
    checkpoint = CHECKPOINT_REGISTRY.get(checkpoint_key, CHECKPOINT_REGISTRY[DEFAULT_CHECKPOINT])
    base = checkpoint.get("hf_repo_base", "")
    quant = quantization if quantization else DEFAULT_QUANTIZATION
    return f"{base}:{quant}"


def get_current_checkpoint_info():
    """Get info for currently selected checkpoint"""
    return CHECKPOINT_REGISTRY.get(CURRENT_CHECKPOINT_KEY, CHECKPOINT_REGISTRY[DEFAULT_CHECKPOINT])


# Configuration for datasets
DATASET_CONFIGS = {
    'Loggenix Synthetic AI Tasks Eval (with outputs)-small': {
        'repo_id': 'kshitijthakkar/loggenix-synthetic-ai-tasks-eval-with-outputs',
        'split': 'train'
    },
    'Loggenix Synthetic AI Tasks Eval (with outputs) v5-large': {
        'repo_id': 'kshitijthakkar/loggenix-synthetic-ai-tasks-eval_v5-with-outputs',
        'split': 'train'
    },
    'Loggenix Synthetic AI Tasks Eval (with outputs) v6-large': {
        'repo_id': 'kshitijthakkar/loggenix-synthetic-ai-tasks-eval_v6-with-outputs',
        'split': 'train'
    },
    'Loggenix Synthetic AI Tasks Eval (with outputs) v7-large': {
        'repo_id': 'kshitijthakkar/loggenix-synthetic-ai-tasks-eval_v5-with-outputs-v7-sft-v1',
        'split': 'train'
    }
}


class BenchmarkPlotter:
    """Dynamic benchmark plotter that uses checkpoint-specific data from registry"""

    # Baseline models for comparison
    # Shows that our $200 budget tiny model can compete with/outperform larger models from big labs
    BASELINE_MODELS = {
        'zero_shot': {
            # Loggenix family
            'Loggenix 0.3B': {'params': 330, 'MMLU': 24.6, 'HellaSwag': 25.0, 'PIQA': 55.0, 'ARC': 15.0, 'WinoGrande': 40.0, 'BoolQ': 40.0, 'OpenBookQA': 28.0, 'is_loggenix_family': True},
            # Small models (< 200M)
            'Falcon-H1-90M': {'params': 90, 'MMLU': 32.3, 'HellaSwag': 37.52, 'PIQA': 65.13, 'ARC': 24.4, 'WinoGrande': 51.3, 'BoolQ': 58.96, 'OpenBookQA': 30.0},
            'Mobile-LLM-140M': {'params': 140, 'MMLU': 24.4, 'HellaSwag': 33.7, 'PIQA': 63.6, 'ARC': 22.6, 'WinoGrande': 52.2, 'BoolQ': 53.7, 'OpenBookQA': 28.0},
            'SmolLM-135M': {'params': 135, 'MMLU': 24.2, 'HellaSwag': 43.0, 'PIQA': 68.4, 'ARC': 28.1, 'WinoGrande': 52.6, 'BoolQ': 60.5, 'OpenBookQA': 32.0},
            'GPT2-137M': {'params': 137, 'MMLU': 26.29, 'HellaSwag': 29.76, 'PIQA': 62.51, 'ARC': 31.09, 'WinoGrande': 49.72, 'BoolQ': 50.0, 'OpenBookQA': 26.0},
            # Medium models (200M-500M)
            'Gemma3-270M': {'params': 270, 'MMLU': 26.2, 'HellaSwag': 41.5, 'PIQA': 68.3, 'ARC': 25.2, 'WinoGrande': 53.1, 'BoolQ': 58.1, 'OpenBookQA': 30.0},
            'SmolLM-360M': {'params': 360, 'MMLU': 34.17, 'HellaSwag': 53.8, 'PIQA': 72.0, 'ARC': 51.1, 'WinoGrande': 53.7, 'BoolQ': 62.0, 'OpenBookQA': 36.0},
            'Qwen2-500M': {'params': 500, 'MMLU': 31.92, 'HellaSwag': 47.61, 'PIQA': 69.31, 'ARC': 39.74, 'WinoGrande': 54.14, 'BoolQ': 60.0, 'OpenBookQA': 34.0},
            # Larger reference models
            'SmolLM-1.7B': {'params': 1700, 'MMLU': 39.97, 'HellaSwag': 64.1, 'PIQA': 77.3, 'ARC': 61.55, 'WinoGrande': 56.0, 'BoolQ': 65.0, 'OpenBookQA': 42.0},
        },
        'few_shot': {
            # Loggenix family
            'Loggenix 0.3B': {'params': 330, 'MMLU': 25.8, 'HellaSwag': 30.0, 'PIQA': 80.0, 'ARC': 10.0, 'WinoGrande': 50.0, 'BoolQ': 42.0, 'OpenBookQA': 30.0, 'is_loggenix_family': True},
            # Small instruct models
            'SmolLM2-135M-Inst': {'params': 135, 'MMLU': 24.64, 'HellaSwag': 40.21, 'PIQA': 65.0, 'ARC': 26.7, 'WinoGrande': 50.0, 'BoolQ': 55.0, 'OpenBookQA': 30.0},
            'Gemma3-270M-IT': {'params': 270, 'MMLU': 23.38, 'HellaSwag': 36.21, 'PIQA': 65.0, 'ARC': 23.8, 'WinoGrande': 50.0, 'BoolQ': 55.0, 'OpenBookQA': 28.0},
            'SmolLM2-350M-Inst': {'params': 350, 'MMLU': 25.75, 'HellaSwag': 40.93, 'PIQA': 68.0, 'ARC': 32.51, 'WinoGrande': 52.0, 'BoolQ': 58.0, 'OpenBookQA': 34.0},
            'LFM2-350M': {'params': 350, 'MMLU': 43.43, 'HellaSwag': 45.0, 'PIQA': 70.0, 'ARC': 35.0, 'WinoGrande': 55.0, 'BoolQ': 60.0, 'OpenBookQA': 36.0},
            # Large models - shows our tiny $200 model competing with billion-param models
            'Gemma 3 PT 1B': {'params': 1000, 'MMLU': 26.5, 'HellaSwag': 62.3, 'PIQA': 73.8, 'ARC': 38.4, 'WinoGrande': 58.2, 'BoolQ': 65.0, 'OpenBookQA': 38.0},
            'Gemma 3 PT 4B': {'params': 4000, 'MMLU': 59.6, 'HellaSwag': 77.2, 'PIQA': 79.6, 'ARC': 56.2, 'WinoGrande': 64.7, 'BoolQ': 75.0, 'OpenBookQA': 48.0},
            'Gemma 3 PT 12B': {'params': 12000, 'MMLU': 74.5, 'HellaSwag': 84.2, 'PIQA': 81.8, 'ARC': 68.9, 'WinoGrande': 74.3, 'BoolQ': 82.0, 'OpenBookQA': 56.0},
            'Gemma 3 PT 27B': {'params': 27000, 'MMLU': 78.6, 'HellaSwag': 85.6, 'PIQA': 83.3, 'ARC': 70.6, 'WinoGrande': 78.8, 'BoolQ': 85.0, 'OpenBookQA': 62.0},
        }
    }

    def __init__(self, checkpoint_key=None):
        self.checkpoint_key = checkpoint_key or CURRENT_CHECKPOINT_KEY
        self.refresh_data()

    def refresh_data(self, checkpoint_key=None):
        """Refresh benchmark data for a specific checkpoint"""
        if checkpoint_key:
            self.checkpoint_key = checkpoint_key

        checkpoint = CHECKPOINT_REGISTRY.get(self.checkpoint_key, CHECKPOINT_REGISTRY[DEFAULT_CHECKPOINT])
        benchmarks = checkpoint.get("benchmarks", {})

        # Build zero-shot comparison data
        zero_shot_data = {
            'Model': [f'Loggenix 0.4B {self.checkpoint_key}'],
            'Parameters': ['0.4B'],
            'Param_Numeric': [400],
            'MMLU': [benchmarks.get('MMLU', 0.0)],
            'HellaSwag': [benchmarks.get('HellaSwag', 0.0)],
            'PIQA': [benchmarks.get('PIQA', 0.0)],
            'ARC': [benchmarks.get('ARC', 0.0)],
            'WinoGrande': [benchmarks.get('WinoGrande', 0.0)],
            'BoolQ': [benchmarks.get('BoolQ', 0.0)],
            'OpenBookQA': [benchmarks.get('OpenBookQA', 0.0)],
            'IsLoggenix': [True],
            'IsLoggenixFamily': [True]
        }

        # Add baseline models
        for model_name, data in self.BASELINE_MODELS['zero_shot'].items():
            zero_shot_data['Model'].append(model_name)
            zero_shot_data['Parameters'].append(f"{data['params']}M" if data['params'] < 1000 else f"{data['params']/1000:.1f}B")
            zero_shot_data['Param_Numeric'].append(data['params'])
            zero_shot_data['MMLU'].append(data['MMLU'])
            zero_shot_data['HellaSwag'].append(data['HellaSwag'])
            zero_shot_data['PIQA'].append(data['PIQA'])
            zero_shot_data['ARC'].append(data['ARC'])
            zero_shot_data['WinoGrande'].append(data['WinoGrande'])
            zero_shot_data['BoolQ'].append(data.get('BoolQ', 0.0))
            zero_shot_data['OpenBookQA'].append(data.get('OpenBookQA', 0.0))
            zero_shot_data['IsLoggenix'].append(False)
            zero_shot_data['IsLoggenixFamily'].append(data.get('is_loggenix_family', False))

        # Build few-shot comparison data
        few_shot_data = {
            'Model': [f'Loggenix 0.4B {self.checkpoint_key}'],
            'Parameters': ['0.4B'],
            'Param_Numeric': [400],
            'MMLU': [benchmarks.get('MMLU', 0.0)],
            'HellaSwag': [benchmarks.get('HellaSwag', 0.0)],
            'PIQA': [benchmarks.get('PIQA', 0.0)],
            'ARC': [benchmarks.get('ARC', 0.0)],
            'WinoGrande': [benchmarks.get('WinoGrande', 0.0)],
            'BoolQ': [benchmarks.get('BoolQ', 0.0)],
            'OpenBookQA': [benchmarks.get('OpenBookQA', 0.0)],
            'IsLoggenix': [True],
            'IsLoggenixFamily': [True]
        }

        for model_name, data in self.BASELINE_MODELS['few_shot'].items():
            few_shot_data['Model'].append(model_name)
            few_shot_data['Parameters'].append(f"{data['params']}M" if data['params'] < 1000 else f"{data['params']/1000:.1f}B")
            few_shot_data['Param_Numeric'].append(data['params'])
            few_shot_data['MMLU'].append(data['MMLU'])
            few_shot_data['HellaSwag'].append(data['HellaSwag'])
            few_shot_data['PIQA'].append(data['PIQA'])
            few_shot_data['ARC'].append(data['ARC'])
            few_shot_data['WinoGrande'].append(data['WinoGrande'])
            few_shot_data['BoolQ'].append(data.get('BoolQ', 0.0))
            few_shot_data['OpenBookQA'].append(data.get('OpenBookQA', 0.0))
            few_shot_data['IsLoggenix'].append(False)
            few_shot_data['IsLoggenixFamily'].append(data.get('is_loggenix_family', False))

        self.df_zero = pd.DataFrame(zero_shot_data)
        self.df_few = pd.DataFrame(few_shot_data)
        self.checkpoint_benchmarks = benchmarks
        self.checkpoint_info = checkpoint

    def create_matplotlib_comparison(self, shot_type='zero'):
        """Create matplotlib comparison charts"""
        df = self.df_zero if shot_type == 'zero' else self.df_few

        # Create subplots - 3x3 grid for 6 benchmarks + scatter plot + legend
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle(f'{shot_type.title()}-Shot Benchmark - Loggenix 0.4B ({self.checkpoint_key})', fontsize=16, fontweight='bold')

        benchmarks = ['MMLU', 'HellaSwag', 'PIQA', 'ARC', 'WinoGrande', 'BoolQ', 'OpenBookQA']
        axes_flat = axes.flatten()

        # Color palette - highlight current model in red, 0.3B in orange, others in blue
        def get_color(row):
            if row['IsLoggenix']:
                return '#ff6b6b'  # Red for current 0.4B model
            elif row['IsLoggenixFamily']:
                return '#ffa500'  # Orange for Loggenix 0.3B
            else:
                return '#4a90e2'  # Blue for other models
        colors = [get_color(row) for _, row in df.iterrows()]

        for i, benchmark in enumerate(benchmarks):
            ax = axes_flat[i]
            bars = ax.bar(range(len(df)), df[benchmark], color=colors, alpha=0.8)

            # Highlight bars where current Loggenix outperforms
            loggenix_score = df[df['IsLoggenix']][benchmark].iloc[0]
            for j, (bar, score) in enumerate(zip(bars, df[benchmark])):
                if not df['IsLoggenix'].iloc[j] and not df['IsLoggenixFamily'].iloc[j] and score < loggenix_score:
                    bar.set_color('#90EE90')  # Light green for outperformed models

            ax.set_title(f'{benchmark}', fontweight='bold')
            ax.set_ylabel('Score (%)')
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df['Model'], rotation=45, ha='right')
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, df[benchmark]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                        f'{value:.1f}%', ha='center', va='bottom', fontsize=9)

        # Parameter efficiency scatter plot
        ax_scatter = axes_flat[7]
        scatter = ax_scatter.scatter(df['Param_Numeric'], df['MMLU'],
                                     c=colors, s=100, alpha=0.7, edgecolors='black')
        ax_scatter.set_xlabel('Parameters (M)')
        ax_scatter.set_ylabel('MMLU Score (%)')
        ax_scatter.set_title('Parameter Efficiency (MMLU)', fontweight='bold')
        ax_scatter.set_xscale('log')
        ax_scatter.grid(True, alpha=0.3)

        # Add model labels to scatter plot
        for idx, row in df.iterrows():
            ax_scatter.annotate(row['Parameters'],
                                (row['Param_Numeric'], row['MMLU']),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=8, ha='left')

        # Add color legend in axes_flat[8]
        ax_legend = axes_flat[8]
        ax_legend.axis('off')
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='#ff6b6b', label='Loggenix 0.4B (current)'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#ffa500', label='Loggenix 0.3B (family)'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#90EE90', label='Outperformed by 0.4B'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#4a90e2', label='Other models'),
        ]
        ax_legend.legend(handles=legend_elements, loc='center', fontsize=12, title='Color Legend')

        plt.tight_layout()
        return fig

    def create_plotly_interactive(self, shot_type='zero'):
        """Create interactive plotly charts"""
        df = self.df_zero if shot_type == 'zero' else self.df_few

        # Create subplots - 3x3 grid for 7 benchmarks + scatter
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('MMLU', 'HellaSwag', 'PIQA', 'ARC', 'WinoGrande', 'BoolQ', 'OpenBookQA', 'Parameter Efficiency', ''),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "scatter"}, {"secondary_y": False}]]
        )

        benchmarks = ['MMLU', 'HellaSwag', 'PIQA', 'ARC', 'WinoGrande', 'BoolQ', 'OpenBookQA']
        positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1)]

        for i, (benchmark, pos) in enumerate(zip(benchmarks, positions)):
            loggenix_score = df[df['IsLoggenix']][benchmark].iloc[0]

            # Create colors: red=current 0.4B, orange=0.3B family, green=outperformed, blue=others
            bar_colors = []
            for idx, row in df.iterrows():
                if row['IsLoggenix']:
                    bar_colors.append('#ff6b6b')  # Red for current 0.4B
                elif row['IsLoggenixFamily']:
                    bar_colors.append('#ffa500')  # Orange for Loggenix 0.3B
                elif row[benchmark] < loggenix_score:
                    bar_colors.append('#90EE90')  # Light green for outperformed
                else:
                    bar_colors.append('#4a90e2')  # Blue for others

            fig.add_trace(
                go.Bar(
                    x=df['Model'],
                    y=df[benchmark],
                    name=benchmark,
                    marker_color=bar_colors,
                    text=[f'{val:.1f}%' for val in df[benchmark]],
                    textposition='outside',
                    showlegend=False,

                ),
                row=pos[0], col=pos[1]
            )

        # Parameter efficiency scatter plot with color coding
        scatter_colors = []
        for _, row in df.iterrows():
            if row['IsLoggenix']:
                scatter_colors.append('#ff6b6b')  # Red for current 0.4B
            elif row['IsLoggenixFamily']:
                scatter_colors.append('#ffa500')  # Orange for Loggenix 0.3B
            else:
                scatter_colors.append('#4a90e2')  # Blue for others

        fig.add_trace(
            go.Scatter(
                x=df['Param_Numeric'],
                y=df['MMLU'],
                mode='markers+text',
                text=df['Parameters'],
                textposition='top right',
                marker=dict(
                    size=12,
                    color=scatter_colors,
                    line=dict(width=1, color='black')
                ),
                name='Models',
                showlegend=False
            ),
            row=3, col=2
        )

        # Update layout
        fig.update_layout(
            title_text=f'{shot_type.title()}-Shot Benchmark - Loggenix 0.4B ({self.checkpoint_key})',
            title_x=0.5,
            height=1000,
            showlegend=False
        )

        # Update x-axis for scatter plot to log scale
        fig.update_xaxes(type="log", row=3, col=2, title_text="Parameters (M)")
        fig.update_yaxes(title_text="MMLU Score (%)", row=3, col=2)

        # Update all y-axes for benchmark charts
        for col in range(1, 4):
            fig.update_yaxes(title_text="Score (%)", row=1, col=col)
            fig.update_yaxes(title_text="Score (%)", row=2, col=col)
        fig.update_yaxes(title_text="Score (%)", row=3, col=1)  # OpenBookQA

        return fig

    def create_competitive_analysis_summary(self, shot_type='zero'):
        """Create a summary of competitive areas"""
        df = self.df_zero if shot_type == 'zero' else self.df_few
        loggenix_row = df[df['IsLoggenix']].iloc[0]
        model_name = f"Loggenix {self.checkpoint_key}"

        # Get checkpoint info
        eval_date = self.checkpoint_info.get('eval_date', 'N/A')
        display_name = self.checkpoint_info.get('display_name', self.checkpoint_key)

        summary = f"## {display_name} ({shot_type.title()}-Shot)\n"
        summary += f"*Eval date: {eval_date}*\n\n"

        benchmarks = ['MMLU', 'HellaSwag', 'PIQA', 'ARC', 'WinoGrande', 'BoolQ', 'OpenBookQA']
        competitive_areas = []

        for benchmark in benchmarks:
            loggenix_score = loggenix_row[benchmark]
            if loggenix_score == 0:
                continue  # Skip if no benchmark data

            outperformed = df[(~df['IsLoggenix']) & (df[benchmark] < loggenix_score)]

            if len(outperformed) > 0:
                competitive_areas.append(
                    f"**{benchmark} ({loggenix_score:.1f}%)**: Outperforms {len(outperformed)} models"
                )

            # Check for near-competitive performance
            if shot_type == 'few' and benchmark == 'PIQA' and loggenix_score >= 80:
                competitive_areas.append(
                    f"**{benchmark} ({loggenix_score:.1f}%)**: Near 12B model performance!"
                )

        if competitive_areas:
            summary += "### Competitive Areas:\n"
            for area in competitive_areas:
                summary += f"- {area}\n"
        elif all(self.checkpoint_benchmarks.get(b, 0) == 0 for b in benchmarks):
            summary += "### No Benchmark Data\n"
            summary += "- Benchmark results not yet available for this checkpoint\n"
            summary += "- Run evaluations and update CHECKPOINT_REGISTRY\n"
        else:
            summary += "### Performance Notes:\n"
            summary += "- Model shows efficient performance for parameter count\n"

        # Additional metrics from checkpoint
        extra_metrics = []
        if self.checkpoint_benchmarks.get('synthetic_mean', 0) > 0:
            extra_metrics.append(f"Synthetic Tasks: {self.checkpoint_benchmarks['synthetic_mean']:.1f}%")
        if self.checkpoint_benchmarks.get('tool_calling', 0) > 0:
            extra_metrics.append(f"Tool Calling: {self.checkpoint_benchmarks['tool_calling']:.1f}%")
        if self.checkpoint_benchmarks.get('programming', 0) > 0:
            extra_metrics.append(f"Programming: {self.checkpoint_benchmarks['programming']:.1f}%")

        if extra_metrics:
            summary += f"\n### Additional Metrics:\n"
            for m in extra_metrics:
                summary += f"- {m}\n"

        summary += f"\n### Architecture:\n"
        summary += f"- MoE: 0.4B total, 0.2B active parameters\n"

        return summary


# ===== HUGGINGFACE DATASET FLAGGING FUNCTIONS =====

def ensure_dataset_exists():
    """
    Ensure the flagged responses dataset exists on HuggingFace Hub.
    Creates it if it doesn't exist.
    """
    try:
        # Check if dataset exists
        HF_API.repo_info(repo_id=FLAGGED_RESPONSES_DATASET, repo_type="dataset")
        return True
    except Exception:
        # Dataset doesn't exist, create it
        try:
            HF_API.create_repo(
                repo_id=FLAGGED_RESPONSES_DATASET,
                repo_type="dataset",
                private=True,
                exist_ok=True
            )
            print(f"Created new dataset: {FLAGGED_RESPONSES_DATASET}")
            return True
        except Exception as e:
            print(f"Failed to create dataset: {str(e)}")
            return False


def push_flagged_response_to_hub(flag_entry: dict) -> str:
    """
    Push a flagged response to HuggingFace Hub dataset.
    Creates the dataset if it doesn't exist.
    This ensures persistence since HF Spaces are stateless Docker containers.
    """
    try:
        # Ensure the dataset repo exists
        if not ensure_dataset_exists():
            return "error: Could not create or access dataset repository"

        # Try to load existing dataset
        existing_data = []
        try:
            existing_ds = load_dataset(FLAGGED_RESPONSES_DATASET, split='train')
            existing_data = existing_ds.to_pandas().to_dict('records')
            print(f"Loaded {len(existing_data)} existing entries from dataset")
        except Exception as e:
            # Dataset exists but is empty or has no data yet
            print(f"Starting fresh dataset (no existing data): {str(e)}")
            existing_data = []

        # Add new entry
        existing_data.append(flag_entry)

        # Create new dataset with all entries
        new_ds = Dataset.from_list(existing_data)

        # Push to hub (this will create the dataset files if they don't exist)
        new_ds.push_to_hub(
            FLAGGED_RESPONSES_DATASET,
            private=True,
            commit_message=f"Add flagged response: {flag_entry.get('flag_reason', 'N/A')[:50]}"
        )

        print(f"Successfully pushed {len(existing_data)} entries to {FLAGGED_RESPONSES_DATASET}")
        return "success"
    except Exception as e:
        print(f"Error pushing to HF Hub: {str(e)}")
        return f"error: {str(e)}"


def load_flagged_responses_from_hub() -> pd.DataFrame:
    """
    Load flagged responses from HuggingFace Hub dataset.
    """
    try:
        ds = load_dataset(FLAGGED_RESPONSES_DATASET, split='train')
        df = ds.to_pandas()

        # Format for display
        display_data = []
        for idx, row in df.iterrows():
            display_data.append({
                "Timestamp": row.get("timestamp", "N/A"),
                "Flag Reason": row.get("flag_reason", "N/A"),
                "Flagged Message": str(row.get("flagged_message", "N/A"))[:100] + "...",
                "Model": row.get("model_id", "N/A"),
                "Checkpoint": row.get("checkpoint", "N/A"),
                "Context Length": str(len(row.get("conversation_context", []))) + " messages"
            })
        return pd.DataFrame(display_data)
    except Exception as e:
        print(f"Error loading from HF Hub: {str(e)}")
        return pd.DataFrame()


# Load main dataset for inference tab
def load_inference_dataset():
    """Load the main dataset for inference use case"""
    try:
        print("Loading synthetic-ai-tasks-eval-v5 dataset...")
        dataset = load_dataset(
            'kshitijthakkar/synthetic-ai-tasks-eval-v5',
            split='train',
            trust_remote_code=True
        )
        df = dataset.to_pandas()
        print(f"Successfully loaded: {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return pd.DataFrame({'Error': [f'Failed to load: {str(e)}']})


# Load dataset for eval samples tab
def load_eval_datasets():
    """Load all datasets for evaluation samples"""
    datasets = {}
    for display_name, config in DATASET_CONFIGS.items():
        try:
            print(f"Loading {display_name}...")
            dataset = load_dataset(
                config['repo_id'],
                split=config['split'],
                trust_remote_code=True
            )
            df = dataset.to_pandas()
            datasets[display_name] = df
            print(f"Successfully loaded {display_name}: {len(df)} rows")
        except Exception as e:
            print(f"Error loading {display_name}: {str(e)}")
            datasets[display_name] = pd.DataFrame({
                'Error': [f'Failed to load: {str(e)}'],
                'Dataset': [config['repo_id']]
            })
    return datasets


# Load datasets
INFERENCE_DATASET = load_inference_dataset()
EVAL_DATASETS = load_eval_datasets()

# Note: Checkpoint info is now managed via CHECKPOINT_REGISTRY and CURRENT_CHECKPOINT_KEY


# ===== TAB 1: INFERENCE USE CASE WITH INTEGRATED FLAGGING =====

def get_task_types():
    """Get unique task types from inference dataset"""
    if 'task_type' in INFERENCE_DATASET.columns:
        task_types = INFERENCE_DATASET['task_type'].unique().tolist()
        return [str(t) for t in task_types if pd.notna(t)]
    return ["No task types available"]


def get_task_by_type(task_type):
    """Get task content by task type"""
    if 'task_type' in INFERENCE_DATASET.columns and 'task' in INFERENCE_DATASET.columns:
        filtered = INFERENCE_DATASET[INFERENCE_DATASET['task_type'] == task_type]
        if len(filtered) > 0:
            return str(filtered.iloc[0]['task'])
    return "No task found for this type"


def chat_interface_with_inference(prompt, history, system_prompt, inference_config, checkpoint_key, backend, quantization):
    """Enhanced chat interface with model inference and history"""
    global CURRENT_CHECKPOINT_KEY

    if not prompt.strip():
        return history, ""

    # Update current checkpoint
    CURRENT_CHECKPOINT_KEY = checkpoint_key

    # Add user message to history (Gradio 6 messages format)
    history.append({"role": "user", "content": prompt})

    try:
        # Set default backend label
        backend_label = backend.split(" ")[0] if backend else "Unknown"

        if not system_prompt.strip():
            response = "Please select a task type to load system prompt first."
        else:
            # Get checkpoint info
            checkpoint = CHECKPOINT_REGISTRY.get(checkpoint_key, CHECKPOINT_REGISTRY[DEFAULT_CHECKPOINT])

            if backend == "Ollama (GGUF)":
                # Build Ollama model name from checkpoint and quantization
                model_name = build_model_name(checkpoint_key, quantization)
                backend_label = f"Ollama/{quantization}"

                # Run inference using Ollama
                response = ollama_generate(
                    system_prompt=system_prompt,
                    user_input=prompt,
                    config_name=inference_config,
                    model_name=model_name
                )
            else:
                # Use Transformers backend with HF model ID
                model_name = checkpoint.get('hf_model_id', '')
                backend_label = "Transformers"

                # Run inference using Transformers
                response = transformers_generate(
                    system_prompt=system_prompt,
                    user_input=prompt,
                    config_name=inference_config,
                    model_name=model_name
                )

        # Format and add AI response to history (Gradio 6 messages format)
        formatted_response = f"**AI Assistant ({checkpoint_key}, {backend_label}):**\n{response}"
        history.append({"role": "assistant", "content": formatted_response})

    except Exception as e:
        error_msg = f"**AI Assistant:**\nError during inference: {str(e)}"
        history.append({"role": "assistant", "content": error_msg})

    return history, ""


def flag_response(history, flagged_message, flag_reason):
    """Flag a response and push to HuggingFace Hub dataset (Gradio 6 messages format)"""
    if not flagged_message or flagged_message == "No responses available":
        return "Invalid message selection."

    try:
        flagged_index = int(flagged_message.split()[1][:-1])
        if flagged_index >= len(history) or history[flagged_index].get("role") != "assistant":
            return "You can only flag assistant responses."

        flagged_message_content = history[flagged_index].get("content", "")

        # Get current checkpoint info
        checkpoint_info = CHECKPOINT_REGISTRY.get(CURRENT_CHECKPOINT_KEY, CHECKPOINT_REGISTRY[DEFAULT_CHECKPOINT])

        # Create flag entry with model metadata
        flag_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "flag_reason": str(flag_reason),
            "flagged_message": str(flagged_message_content),
            "conversation_context": json.dumps(history),  # Serialize for dataset
            "model_id": checkpoint_info.get("hf_model_id", "unknown"),
            "checkpoint": CURRENT_CHECKPOINT_KEY,
            "checkpoint_display_name": checkpoint_info.get("display_name", CURRENT_CHECKPOINT_KEY),
        }

        # Push to HuggingFace Hub dataset
        result = push_flagged_response_to_hub(flag_entry)

        if result == "success":
            return f"Response flagged successfully and pushed to HF Hub dataset: {flag_reason}"
        else:
            # Fallback to local logging if Hub push fails
            os.makedirs("logs", exist_ok=True)
            with open("logs/flagged_responses.log", "a") as f:
                f.write(json.dumps(flag_entry) + "\n")
            return f"Response flagged (local fallback - Hub push failed: {result})"

    except Exception as e:
        return f"Error flagging response: {str(e)}"


def get_assistant_responses(history):
    """Get dropdown options for assistant responses (Gradio 6 messages format)"""
    responses = [
        f"Response {i}: {str(msg.get('content', ''))[:50]}..."
        for i, msg in enumerate(history)
        if msg.get("role") == "assistant"
    ]

    if not responses:
        responses = ["No responses available"]

    return gr.update(choices=responses, value=responses[0] if responses else "No responses available")


def display_selected_message(selected_index, history):
    """Display the selected flagged message (Gradio 6 messages format)"""
    if selected_index == "No responses available":
        return "No responses available"

    try:
        flagged_index = int(selected_index.split()[1][:-1])
        if flagged_index < len(history) and history[flagged_index].get("role") == "assistant":
            return history[flagged_index].get("content", "")
        else:
            return "Invalid selection."
    except Exception as e:
        return f"Error: {str(e)}"


def clear_inference_history():
    """Clear chat history for inference tab"""
    return [], gr.update(choices=["No responses available"], value="No responses available")


# ===== TAB 2: EVAL SAMPLES =====

def update_eval_table(dataset_name):
    """Update eval table based on selected dataset"""
    if dataset_name in EVAL_DATASETS:
        return EVAL_DATASETS[dataset_name].head(100)
    return pd.DataFrame()


def get_eval_dataset_info(dataset_name):
    """Get info about selected eval dataset"""
    if dataset_name in EVAL_DATASETS:
        df = EVAL_DATASETS[dataset_name]
        return f"""
        **Dataset**: {dataset_name}
        - **Rows**: {len(df):,}
        - **Columns**: {len(df.columns)}
        - **Column Names**: {', '.join(df.columns.tolist())}
        """
    return "No dataset selected"


def get_task_types_for_eval(dataset_name):
    """Get unique task types from selected eval dataset"""
    if dataset_name in EVAL_DATASETS and 'task_type' in EVAL_DATASETS[dataset_name].columns:
        task_types = EVAL_DATASETS[dataset_name]['task_type'].unique().tolist()
        return [str(t) for t in task_types if pd.notna(t)]
    return ["No task types available"]


def get_selected_row_data_by_type(dataset_name, task_type):
    """Get all data for the first row of a selected dataset and task type"""
    if (dataset_name in EVAL_DATASETS and
            'task_type' in EVAL_DATASETS[dataset_name].columns and
            'task' in EVAL_DATASETS[dataset_name].columns):

        filtered = EVAL_DATASETS[dataset_name][EVAL_DATASETS[dataset_name]['task_type'] == task_type]
        if len(filtered) > 0:
            row = filtered.iloc[0]

            # Extract all fields with safe handling for missing columns
            task = str(row.get('task', 'N/A'))
            input_model = str(row.get('input_model', 'N/A'))
            expected_response = str(row.get('expected_response', 'N/A'))
            loggenix_output = str(row.get('loggenix_output', 'N/A'))
            output_model = str(row.get('output_model', 'N/A'))
            input_text = str(row.get('input', 'N/A'))

            return input_model, output_model, task, input_text, expected_response, loggenix_output

    return "", "", "", "", "", ""


# ===== TAB 3: VIEW FLAGGED RESPONSES (FROM HF HUB) =====

def read_flagged_messages():
    """Read flagged messages from HuggingFace Hub dataset"""
    return load_flagged_responses_from_hub()


def handle_row_select(evt: gr.SelectData):
    """Handle row selection in flagged messages table (Gradio 6 messages format)"""
    try:
        # Load from Hub
        ds = load_dataset(FLAGGED_RESPONSES_DATASET, split='train')
        df = ds.to_pandas()

        if evt.index[0] < len(df):
            row = df.iloc[evt.index[0]]
            context_str = row.get("conversation_context", "[]")
            try:
                conversation_context = json.loads(context_str)
                # Convert old tuple format to messages format if needed
                if conversation_context and isinstance(conversation_context[0], (list, tuple)):
                    conversation_context = [
                        {"role": "user" if msg[0] == "You" else "assistant", "content": msg[1]}
                        for msg in conversation_context
                    ]
            except:
                conversation_context = []
            return conversation_context
        return []
    except Exception as e:
        return [{"role": "assistant", "content": f"Error loading conversation: {str(e)}"}]


# ===== MAIN INTERFACE =====

def create_interface():
    with gr.Blocks(title="Loggenix MoE 0.4B Demo") as demo:
        gr.Markdown("# Loggenix MoE 0.4B-A0.2B Demo")
        gr.Markdown("Comprehensive platform for AI model evaluation and testing")

        with gr.Tabs():
            # TAB 1: ABOUT (First tab for documentation)
            with gr.Tab("About"):
                with gr.Group():
                    gr.Markdown("# Loggenix MoE 0.4B-A0.2B - User Guide")

                    with gr.Accordion("Application Overview", open=True):
                        gr.Markdown("""
                        The **Loggenix MoE 0.4B Demo** is a platform for testing, evaluating, and monitoring the Loggenix MoE 0.4B-A0.2B model.

                        **Model**: `kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.1`

                        **Key Features:**
                        - Interactive model testing with real-time inference
                        - Comprehensive dataset exploration and comparison
                        - Response quality monitoring with HuggingFace Hub persistence
                        - Performance analysis and evaluation metrics

                        **Architecture:**
                        - Total Parameters: 0.4B (400M)
                        - Active Parameters: 0.2B (200M) - MoE efficiency
                        - Supports both HuggingFace Transformers and GGUF (Ollama) backends
                        """)

                    with gr.Accordion("Model Specifications", open=False):
                        # Build checkpoint list
                        checkpoint_list_md = "\n".join([
                            f"- `{k}`: {v['display_name']}"
                            for k, v in CHECKPOINT_REGISTRY.items()
                        ])
                        gr.Markdown(f"""
                        **Default Checkpoint**: `{DEFAULT_CHECKPOINT}`

                        **Available Checkpoints:**
{checkpoint_list_md}

                        - **Architecture**: Mixture of Experts (MoE)
                        - **Total Parameters**: 0.4B
                        - **Active Parameters**: 0.2B
                        - **Context Length**: 8192 tokens
                        - **Precision**: FP16
                        - **Flash Attention**: Supported
                        - **Tool Calling**: Enabled

                        **Inference Configurations:**
                        - Optimized for Speed: 512 tokens max
                        - Middle-ground: 2048 tokens max
                        - Full Capacity: 4096 tokens max

                        *Switch checkpoints in the Inference or Benchmarks tabs*
                        """)

                    with gr.Accordion("Flagged Responses Dataset", open=False):
                        gr.Markdown(f"""
                        **Dataset**: `{FLAGGED_RESPONSES_DATASET}`

                        Flagged responses are automatically pushed to a HuggingFace Hub dataset for:
                        - **Persistence**: HF Spaces are stateless, so data is stored on the Hub
                        - **Analysis**: Review flagged responses for model behavior patterns
                        - **Refinement**: Use flagged data for future model improvements

                        Each flag entry includes:
                        - Timestamp
                        - Flag reason
                        - Flagged message content
                        - Full conversation context
                        - Model ID and checkpoint version
                        """)

                    gr.Markdown("""
                    ---
                    **Developed by**: Kshitij Thakkar
                    **Model Version**: 0.4B-A0.2B
                    """)

            # TAB 2: INFERENCE USE CASE
            with gr.Tab("Inference"):
                gr.Markdown("## Model Inference Testing with Response Flagging")

                with gr.Row():
                    with gr.Column(scale=1):
                        # Checkpoint selector for inference
                        inference_checkpoint_dropdown = gr.Dropdown(
                            choices=list(CHECKPOINT_REGISTRY.keys()),
                            value=DEFAULT_CHECKPOINT,
                            label="Model Checkpoint",
                            info="Select checkpoint for inference"
                        )

                        # Inference backend selector
                        inference_backend_dropdown = gr.Dropdown(
                            choices=["Ollama (GGUF)", "Transformers (HF)"],
                            value="Ollama (GGUF)",
                            label="Inference Backend",
                            info="Select inference backend"
                        )

                        # Quantization selector (for Ollama only)
                        quantization_dropdown = gr.Dropdown(
                            choices=list(QUANTIZATION_OPTIONS.keys()),
                            value=DEFAULT_QUANTIZATION,
                            label="Quantization (Ollama only)",
                            info="Select GGUF quantization level",
                            visible=True
                        )

                        task_type_dropdown = gr.Dropdown(
                            choices=get_task_types(),
                            value=get_task_types()[0] if get_task_types() else None,
                            label="Task Type",
                            info="Select task type to load system prompt"
                        )

                        inference_config = gr.Dropdown(
                            choices=list(get_inference_configs().keys()),
                            value="Optimized for Speed",
                            label="Inference Configuration",
                            info="Select inference optimization level"
                        )

                    with gr.Column(scale=2):
                        system_prompt = gr.Textbox(
                            label="System Prompt (Editable)",
                            lines=6,
                            max_lines=10,
                            placeholder="Select a task type to load system prompt...",
                            interactive=True
                        )

                gr.Markdown("### Chat Interface")
                with gr.Row():
                    with gr.Column(scale=2):
                        chat_display = gr.Chatbot(label="Conversation History", height=400)
                        chat_history_state = gr.State([])

                        with gr.Row():
                            chat_input = gr.Textbox(
                                placeholder="Enter your message here...",
                                label="Your Message",
                                scale=4
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)

                        with gr.Row():
                            clear_chat_btn = gr.Button("Clear History", variant="secondary")

                    with gr.Column(scale=1):
                        gr.Markdown("### Flag Response")
                        gr.Markdown(f"*Flags are pushed to `{FLAGGED_RESPONSES_DATASET}`*")

                        flagged_message_index = gr.Dropdown(
                            label="Select a response to flag",
                            choices=["No responses available"],
                            value="No responses available",
                            interactive=True
                        )

                        selected_message_display = gr.Textbox(
                            label="Selected Response",
                            interactive=False,
                            lines=4,
                            max_lines=6
                        )

                        flag_reason = gr.Textbox(
                            placeholder="Enter reason for flagging...",
                            label="Reason for Flagging"
                        )

                        flag_btn = gr.Button("Flag Response", variant="stop")
                        flag_output = gr.Textbox(label="Flagging Status", visible=True, lines=2)

                # Event handlers
                task_type_dropdown.change(
                    fn=get_task_by_type,
                    inputs=[task_type_dropdown],
                    outputs=[system_prompt]
                )

                # Toggle quantization visibility based on backend
                def toggle_quantization_visibility(backend):
                    return gr.update(visible=(backend == "Ollama (GGUF)"))

                inference_backend_dropdown.change(
                    fn=toggle_quantization_visibility,
                    inputs=[inference_backend_dropdown],
                    outputs=[quantization_dropdown]
                )

                send_btn.click(
                    chat_interface_with_inference,
                    inputs=[chat_input, chat_history_state, system_prompt, inference_config, inference_checkpoint_dropdown, inference_backend_dropdown, quantization_dropdown],
                    outputs=[chat_display, chat_input]
                ).then(
                    lambda x: x,
                    inputs=[chat_display],
                    outputs=[chat_history_state]
                ).then(
                    get_assistant_responses,
                    inputs=[chat_history_state],
                    outputs=[flagged_message_index]
                )

                chat_input.submit(
                    chat_interface_with_inference,
                    inputs=[chat_input, chat_history_state, system_prompt, inference_config, inference_checkpoint_dropdown, inference_backend_dropdown, quantization_dropdown],
                    outputs=[chat_display, chat_input]
                ).then(
                    lambda x: x,
                    inputs=[chat_display],
                    outputs=[chat_history_state]
                ).then(
                    get_assistant_responses,
                    inputs=[chat_history_state],
                    outputs=[flagged_message_index]
                )

                clear_chat_btn.click(
                    clear_inference_history,
                    outputs=[chat_display, flagged_message_index]
                ).then(
                    lambda: [],
                    outputs=[chat_history_state]
                )

                flagged_message_index.change(
                    display_selected_message,
                    inputs=[flagged_message_index, chat_history_state],
                    outputs=[selected_message_display]
                )

                flag_btn.click(
                    flag_response,
                    inputs=[chat_history_state, flagged_message_index, flag_reason],
                    outputs=[flag_output]
                )

            # TAB 3: EVAL SAMPLES
            with gr.Tab("Eval Samples"):
                gr.Markdown("## Dataset Evaluation Samples")

                with gr.Row():
                    with gr.Column(scale=1):
                        eval_dataset_dropdown = gr.Dropdown(
                            choices=list(EVAL_DATASETS.keys()),
                            value=list(EVAL_DATASETS.keys())[0] if EVAL_DATASETS else None,
                            label="Select Dataset"
                        )

                        eval_task_type_dropdown = gr.Dropdown(
                            choices=[],
                            label="Select Task Type",
                            allow_custom_value=True
                        )

                    with gr.Column(scale=1):
                        eval_dataset_info = gr.Markdown(
                            get_eval_dataset_info(list(EVAL_DATASETS.keys())[0] if EVAL_DATASETS else "")
                        )

                gr.Markdown("### Task Details")
                with gr.Row():
                    input_model_field = gr.Textbox(label="input_model", lines=1, interactive=False)
                    output_model_field = gr.Textbox(label="output_model", lines=1, interactive=False)

                with gr.Row():
                    task_field = gr.Textbox(label="Task", lines=2, max_lines=5, interactive=False)

                with gr.Row():
                    input_field = gr.Textbox(label="input", lines=12, max_lines=20, interactive=False)

                gr.Markdown("### Expected vs Actual Response Comparison")
                with gr.Row():
                    loggenix_output_field = gr.Textbox(label="Expected Response", lines=30, max_lines=40, interactive=False)
                    expected_response_field = gr.Textbox(label="Loggenix Output", lines=30, max_lines=40, interactive=False)

                def update_eval_components(dataset_name):
                    info = get_eval_dataset_info(dataset_name)
                    task_types = get_task_types_for_eval(dataset_name)
                    return info, gr.update(choices=task_types, value=task_types[0] if task_types else "No task types available")

                eval_dataset_dropdown.change(
                    fn=update_eval_components,
                    inputs=[eval_dataset_dropdown],
                    outputs=[eval_dataset_info, eval_task_type_dropdown]
                )

                eval_task_type_dropdown.change(
                    fn=get_selected_row_data_by_type,
                    inputs=[eval_dataset_dropdown, eval_task_type_dropdown],
                    outputs=[input_model_field, output_model_field, task_field, input_field,
                             loggenix_output_field, expected_response_field]
                )

            # TAB 4: VIEW FLAGGED RESPONSES
            with gr.Tab("Flagged Responses"):
                gr.Markdown("## Review Flagged Responses")
                gr.Markdown(f"*Data loaded from HuggingFace Hub: `{FLAGGED_RESPONSES_DATASET}`*")

                with gr.Row():
                    with gr.Column():
                        flagged_messages_display = gr.Dataframe(
                            headers=["Timestamp", "Flag Reason", "Flagged Message", "Model", "Checkpoint", "Context Length"],
                            interactive=False,
                            max_height=400
                        )
                        refresh_btn = gr.Button("Refresh", variant="primary")

                    with gr.Column():
                        conversation_context_display = gr.Chatbot(
                            label="Conversation Context",
                            height=400
                        )

                flagged_messages_display.select(
                    handle_row_select,
                    outputs=[conversation_context_display]
                )

                refresh_btn.click(
                    read_flagged_messages,
                    outputs=[flagged_messages_display]
                )

            # TAB 5: MODEL EVAL RESULTS
            with gr.Tab("Benchmarks"):
                gr.Markdown("## Model Evaluation Results")
                gr.Markdown("### Loggenix MoE 0.4B - Checkpoint Benchmark Comparison")

                with gr.Accordion("Chart Guide & Color Legend", open=False):
                    gr.Markdown("""
### Color Legend
| Color | Meaning |
|-------|---------|
| **Red** | Current Loggenix 0.4B model (selected checkpoint) |
| **Orange** | Loggenix 0.3B (previous version, family comparison) |
| **Green** | Models that Loggenix 0.4B **outperforms** |
| **Blue** | Other baseline models |

### Benchmark Descriptions
| Benchmark | Description |
|-----------|-------------|
| **MMLU** | Massive Multitask Language Understanding - 57 academic subjects |
| **HellaSwag** | Commonsense reasoning about everyday situations |
| **PIQA** | Physical Intuition QA - understanding physical world |
| **ARC** | AI2 Reasoning Challenge - grade-school science questions |
| **WinoGrande** | Commonsense pronoun resolution |
| **BoolQ** | Boolean (yes/no) reading comprehension questions |
| **OpenBookQA** | Science facts with open-book reasoning |
| **GSM8K** | Grade school math word problems |

### Chart Types
- **Zero-Shot**: Model answers without examples (tests raw knowledge)
- **Few-Shot**: Model given a few examples first (tests learning ability)

### Why This Matters
Our tiny **$200 budget** model (0.4B params) trained on consumer hardware competes with and sometimes **outperforms** models from major AI labs with billion-dollar budgets. Green bars show where Loggenix wins!
                    """)

                def plot_comparison(checkpoint_key, shot_type, plot_type):
                    """Generate benchmark comparison for selected checkpoint"""
                    plotter = BenchmarkPlotter(checkpoint_key)
                    if plot_type == "Interactive (Plotly)":
                        fig = plotter.create_plotly_interactive(shot_type.lower())
                    else:
                        fig = plotter.create_matplotlib_comparison(shot_type.lower())
                    summary = plotter.create_competitive_analysis_summary(shot_type.lower())

                    # Get checkpoint info for display
                    cp_info = CHECKPOINT_REGISTRY.get(checkpoint_key, {})
                    benchmarks = cp_info.get('benchmarks', {})

                    info_text = f"""**Checkpoint:** {cp_info.get('display_name', checkpoint_key)}
**Model ID:** `{cp_info.get('hf_model_id', 'N/A')}`
**Eval Date:** {cp_info.get('eval_date', 'Not evaluated')}

| Metric | Score |
|--------|-------|
| MMLU | {benchmarks.get('MMLU', 0):.2f}% |
| HellaSwag | {benchmarks.get('HellaSwag', 0):.2f}% |
| PIQA | {benchmarks.get('PIQA', 0):.2f}% |
| ARC | {benchmarks.get('ARC', 0):.2f}% |
| WinoGrande | {benchmarks.get('WinoGrande', 0):.2f}% |
| BoolQ | {benchmarks.get('BoolQ', 0):.2f}% |
| OpenBookQA | {benchmarks.get('OpenBookQA', 0):.2f}% |
| GSM8K | {benchmarks.get('GSM8K', 0):.2f}% |
"""
                    return fig, summary, info_text

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Checkpoint & Evaluation Controls")

                        benchmark_checkpoint_dropdown = gr.Dropdown(
                            choices=list(CHECKPOINT_REGISTRY.keys()),
                            value=DEFAULT_CHECKPOINT,
                            label="Select Checkpoint",
                            info="Choose checkpoint to view benchmarks"
                        )

                        shot_type = gr.Radio(
                            choices=["Zero", "Few"],
                            value="Zero",
                            label="Shot Type"
                        )
                        plot_type = gr.Radio(
                            choices=["Interactive (Plotly)", "Static (Matplotlib)"],
                            value="Interactive (Plotly)",
                            label="Plot Type"
                        )
                        plot_button = gr.Button("Generate Comparison", variant="primary", size="lg")

                        gr.Markdown("---")
                        checkpoint_info_display = gr.Markdown("*Select checkpoint and click Generate*")

                    with gr.Column(scale=2):
                        gr.Markdown("#### Available Checkpoints")
                        checkpoint_list = "\n".join([
                            f"- **{k}**: {v['display_name']}" + (" (has benchmarks)" if v['benchmarks'].get('MMLU', 0) > 0 else " (pending)")
                            for k, v in CHECKPOINT_REGISTRY.items()
                        ])
                        gr.Markdown(f"""
{checkpoint_list}

*Select a checkpoint and click Generate to view benchmarks*
                        """)

                with gr.Row():
                    with gr.Column(scale=3):
                        plot_output = gr.Plot(label="Performance Comparison")

                    with gr.Column(scale=1):
                        gr.Markdown("#### Analysis Summary")
                        summary_output = gr.Markdown(
                            value="Select evaluation parameters and click 'Generate Comparison' to see detailed analysis.")

                plot_button.click(
                    fn=plot_comparison,
                    inputs=[benchmark_checkpoint_dropdown, shot_type, plot_type],
                    outputs=[plot_output, summary_output, checkpoint_info_display]
                )

        # Load initial data
        demo.load(
            fn=read_flagged_messages,
            outputs=[flagged_messages_display]
        )

    return demo


# Launch the application
if __name__ == "__main__":
    print("Starting Loggenix MoE 0.4B Demo...")
    demo = create_interface()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        theme=gr.themes.Ocean()
    )
