#!/usr/bin/env python3
"""
Experiment 1: Format Understanding and Property Extraction
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from exp1_format_property.base_utils import CrystalDataLoader, PromptGenerator, PropertyCalculator, save_results

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: VLLM not available. Please install with: pip install vllm")

MODEL_CONFIGS = {
    "llama3.1-8b": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "tensor_parallel_size": 1,
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.85,
    },
    "llama3.1-70b": {
        "model_name": "meta-llama/Llama-3.1-70B-Instruct",
        "tensor_parallel_size": 4,
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.85,
    },
    "qwen2.5-7b": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "tensor_parallel_size": 1,
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.85,
    },
    "qwen2.5-14b": {
        "model_name": "Qwen/Qwen2.5-14B-Instruct",
        "tensor_parallel_size": 2,
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.85,
    }
}


class UnifiedVLLMInference:
    """Unified VLLM inference engine for format understanding tests."""

    def __init__(self, model_config_key: str = "llama3.1-70b", custom_model_name: str = None):
        self.model_config_key = model_config_key
        self.custom_model_name = custom_model_name

        if custom_model_name:
            self.model_name = custom_model_name
            self.model_config = MODEL_CONFIGS["llama3.1-70b"]
        else:
            if model_config_key not in MODEL_CONFIGS:
                raise ValueError(f"Unknown model config: {model_config_key}")
            self.model_config = MODEL_CONFIGS[model_config_key]
            self.model_name = self.model_config["model_name"]

        self.llm = None
        self.sampling_params = None
        self.tokenizer = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize VLLM model."""
        if not VLLM_AVAILABLE:
            raise RuntimeError("VLLM is required for this experiment")

        print(f"Initializing VLLM model: {self.model_name}")
        model_start = time.time()
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.model_config['tensor_parallel_size'],
            max_model_len=self.model_config['max_model_len'],
            gpu_memory_utilization=self.model_config['gpu_memory_utilization'],
        )
        print(f"Model loaded in {time.time() - model_start:.1f} seconds")

        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception as e:
            print(f"Tokenizer loading failed: {e}")
            self.tokenizer = None

        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=100,
            stop=["\n\n", "Question:", "Crystal Structure"]
        )

    def _create_unified_prompt(self, structure_text: str, prompt_template: str, test_type: str) -> str:
        """Create a formatted prompt for any test type."""
        return f"""You are a crystallography expert. Analyze the following crystal structure data and answer the question precisely.

Crystal Structure Data:
{structure_text}

Question: {prompt_template}

Answer:"""

    def _analyze_tokenization(self, structure_text: str) -> Dict[str, int]:
        """Analyze tokenization properties of structure text."""
        lines = structure_text.strip().split('\n')
        words = structure_text.split()
        chars = len(structure_text)

        tokens = self.tokenizer.encode(structure_text, add_special_tokens=False)
        actual_tokens = len(tokens)

        compression_ratio = actual_tokens / max(chars, 1)
        tokens_per_word = actual_tokens / max(len(words), 1)

        return {
            'num_lines': len(lines),
            'num_words': len(words),
            'num_chars': chars,
            'actual_tokens': actual_tokens,
            'compression_ratio': compression_ratio,
            'tokens_per_word': tokens_per_word,
        }

    def generate_all_prompts(self, paired_data: List[Dict[str, Any]],
                           sample_size: int = 200) -> Tuple[List[str], List[Dict[str, Any]], Dict[int, List[str]]]:
        """Generate all prompts for format understanding tests."""

        if len(paired_data) > sample_size:
            paired_data = np.random.choice(paired_data, size=sample_size, replace=False).tolist()

        print("Calculating ground truth properties...")
        paired_data = PropertyCalculator.calculate_ground_truth(paired_data)

        all_prompts = []
        prompt_metadata = []

        format_recognition_prompts = [
            ("format_type", "What file format is this crystal structure data in?"),
            ("cif_or_poscar", "Is this structure data in CIF or POSCAR format?"),
        ]

        property_prompts = PromptGenerator.property_extraction_prompts()

        tier_mapping = {
            1: ['has_oxygen', 'chemical_formula', 'num_elements', 'all_angles_90'],
            2: ['num_atoms', 'cell_volume'],
            3: ['lattice_a', 'space_group', 'lattice_system', 'density']
        }

        for item in paired_data:
            for format_type in ['cif', 'poscar']:
                structure_text = item['cif_text'] if format_type == 'cif' else item['poscar_text']
                tokenization = self._analyze_tokenization(structure_text)

                for prompt_id, prompt_template in format_recognition_prompts:
                    prompt = self._create_unified_prompt(structure_text, prompt_template, "format_recognition")
                    all_prompts.append(prompt)

                    prompt_metadata.append({
                        'test_type': 'format_recognition',
                        'mbid': item['mbid'],
                        'format': format_type,
                        'prompt_id': prompt_id,
                        'prompt_template': prompt_template,
                        'tokenization': tokenization,
                        'ground_truth_format': format_type.upper(),
                        'structure_data': item
                    })

                for tier, properties in tier_mapping.items():
                    for property_name in properties:
                        if property_name in property_prompts:
                            prompt_template = property_prompts[property_name][0]
                            prompt = self._create_unified_prompt(structure_text, prompt_template, "property_extraction")
                            all_prompts.append(prompt)

                            prompt_metadata.append({
                                'test_type': 'property_extraction',
                                'mbid': item['mbid'],
                                'format': format_type,
                                'tier': tier,
                                'property': property_name,
                                'prompt_template': prompt_template,
                                'tokenization': tokenization,
                                'ground_truth': item.get(f'gt_{property_name}'),
                                'structure_data': item
                            })

        print(f"Generated {len(all_prompts)} total prompts")
        return all_prompts, prompt_metadata, tier_mapping

    def run_unified_inference(self, paired_data: List[Dict[str, Any]],
                            sample_size: int = 200) -> Dict[str, Any]:
        """Run unified VLLM inference for all format understanding tests."""

        print("Starting Unified VLLM Inference")
        print(f"Input: {len(paired_data)} paired structures")
        print(f"Target sample size: {sample_size}")

        print("Generating prompts...")
        prompt_start = time.time()
        all_prompts, prompt_metadata, tier_mapping = self.generate_all_prompts(paired_data, sample_size)
        print(f"Generated {len(all_prompts)} prompts in {time.time() - prompt_start:.1f} seconds")

        print(f"Running VLLM batch inference on {len(all_prompts)} prompts...")
        start_time = time.time()
        outputs = self.llm.generate(all_prompts, self.sampling_params)
        inference_time = time.time() - start_time

        print(f"Batch inference completed in {inference_time:.2f} seconds")
        print(f"Throughput: {len(all_prompts)/inference_time:.1f} prompts/second")

        unified_results = []
        for i, output in enumerate(outputs):
            response = output.outputs[0].text.strip()
            metadata = prompt_metadata[i]

            result = {
                **metadata,
                'raw_response': response,
                'inference_time': inference_time,
                'prompt_index': i
            }
            unified_results.append(result)

        final_results = {
            'experiment_name': 'unified_format_understanding',
            'model': self.model_name,
            'num_structures': len(paired_data),
            'total_prompts': len(all_prompts),
            'inference_time_seconds': inference_time,
            'throughput_prompts_per_second': len(all_prompts) / inference_time,
            'results': unified_results,
            'metadata': {
                'sample_size': sample_size,
                'format_recognition_prompts': len([r for r in unified_results if r['test_type'] == 'format_recognition']),
                'property_extraction_prompts': len([r for r in unified_results if r['test_type'] == 'property_extraction']),
                'tier_mapping': tier_mapping
            },
            'timestamp': time.time()
        }

        return final_results


def main():
    parser = argparse.ArgumentParser(description="Run unified VLLM inference for crystal format understanding")
    parser.add_argument("--model", type=str, default="llama3.1-70b",
                       choices=list(MODEL_CONFIGS.keys()),
                       help="Model to use")
    parser.add_argument("--custom-model", type=str, default=None,
                       help="Custom model name (overrides --model)")
    parser.add_argument("--sample-size", type=int, default=200,
                       help="Number of structures to test")
    parser.add_argument("--csv-path", type=str, default="../data/band_gap_sample_1000.csv",
                       help="Path to input CSV file")
    parser.add_argument("--output-path", type=str, default="../results/experiment1/unified_vllm_results.json",
                       help="Path to output results file")

    args = parser.parse_args()

    CSV_PATH = args.csv_path
    OUTPUT_PATH = args.output_path
    SAMPLE_SIZE = args.sample_size

    print("=== Unified VLLM Inference ===")
    print(f"Model: {args.model if not args.custom_model else 'custom'}")
    if args.custom_model:
        print(f"Custom model: {args.custom_model}")
    print(f"Target sample size: {SAMPLE_SIZE}")

    print("\n[1/4] Loading crystal data...")
    loader = CrystalDataLoader(CSV_PATH)
    paired_data = loader.get_paired_formats(sample_size=SAMPLE_SIZE)
    print(f"Generated {len(paired_data)} paired structures")

    print(f"\n[2/4] Initializing unified VLLM inference...")
    if args.custom_model:
        inference_engine = UnifiedVLLMInference(custom_model_name=args.custom_model)
    else:
        inference_engine = UnifiedVLLMInference(model_config_key=args.model)

    print(f"\n[3/4] Running unified inference...")
    results = inference_engine.run_unified_inference(paired_data, sample_size=SAMPLE_SIZE)

    print(f"\n[4/4] Saving results...")
    save_results(results, OUTPUT_PATH)

    print("\n=== Unified Inference Complete ===")
    print(f"Total prompts processed: {results['total_prompts']}")
    print(f"Total inference time: {results['inference_time_seconds']:.1f} seconds")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

