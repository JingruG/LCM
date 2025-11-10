#!/usr/bin/env python3
"""
Experiment 2: Stability Judge
Activation patching to identify where the model evaluates thermodynamic stability.
Supports full, MLP-only, and attention-only patching for component decomposition.
"""

import os
import warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd
import json
import re
import argparse
import random
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from pathlib import Path

class StabilityJudge:
    """Activation patching for crystal stability assessment experiments."""

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-70B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        """Load the language model."""
        print(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print(f"Model loaded with {self.model.config.num_hidden_layers} layers")


    def create_clean_prompt(self, stable_poscar: str, strained_poscar: str) -> str:
        """Clean prompt: Stable structure in position A."""

        prompt = f"""[INST]
You are an expert crystallographer evaluating crystal stability. Compare these two crystal structures and determine which is more thermodynamically stable.

Structure A:
{stable_poscar}

Structure B:
{strained_poscar}

Based on crystallographic principles, which structure is more thermodynamically stable?

Answer with SINGLE LETTER either 'A' or 'B'.
[/INST]
Answer:"""
        return prompt

    def create_corrupted_prompt(self, stable_poscar: str, strained_poscar: str) -> str:
        """Corrupted prompt: Stable structure in position B."""

        prompt = f"""[INST]
You are an expert crystallographer evaluating crystal stability. Compare these two crystal structures and determine which is more thermodynamically stable.

Structure A:
{strained_poscar}

Structure B:
{stable_poscar}

Based on crystallographic principles, which structure is more thermodynamically stable?

Answer with SINGLE LETTER either 'A' or 'B'.
[/INST]
Answer:"""
        return prompt

    def extract_answer_choice(self, text: str) -> Optional[str]:
        """Extract answer choice (A or B) from model output."""
        text = text.strip()

        if text in ['A', 'B']:
            return text


        match = re.search(r'^([AB])', text)
        if match:
            return match.group(1)
        return None

    def cache_activations(self, prompt: str, patching_type: str = "full") -> Dict[int, torch.Tensor]:
        """Cache activations from a forward pass at the decision token position."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        cached_activations = {}


        answer_token_ids = self.tokenizer("Answer:", add_special_tokens=False)["input_ids"]
        input_ids = inputs["input_ids"][0].tolist()


        answer_pos = None
        for i in range(len(input_ids) - len(answer_token_ids) + 1):
            if input_ids[i:i+len(answer_token_ids)] == answer_token_ids:
                answer_pos = i + len(answer_token_ids) - 1

        if answer_pos is None:
            print("Warning: Could not find 'Answer:' token sequence, using last position")
            answer_pos = -1



        def hook_fn(layer_idx, component_type="full"):
            def hook(module, input, output):
                if component_type == "full":
                    hidden_states = output[0]
                    cached_activations[layer_idx] = hidden_states[:, answer_pos, :].clone()
                else:
                    if isinstance(output, tuple):
                        component_output = output[0]
                    else:
                        component_output = output
                    cached_activations[layer_idx] = component_output[:, answer_pos, :].clone()
            return hook


        handles = []

        if patching_type == "full":
            for i, layer in enumerate(self.model.model.layers):
                handle = layer.register_forward_hook(hook_fn(i, "full"))
                handles.append(handle)
        elif patching_type == "mlp_only":
            for i, layer in enumerate(self.model.model.layers):
                if hasattr(layer, 'mlp'):
                    handle = layer.mlp.register_forward_hook(hook_fn(i, "mlp"))
                    handles.append(handle)
        elif patching_type == "attention_only":
            for i, layer in enumerate(self.model.model.layers):
                if hasattr(layer, 'self_attn'):
                    handle = layer.self_attn.register_forward_hook(hook_fn(i, "attention"))
                    handles.append(handle)


        with torch.no_grad():
            _ = self.model(**inputs)


        for handle in handles:
            handle.remove()

        return cached_activations

    def patch_and_predict(self, prompt: str, patch_activations: Dict[int, torch.Tensor],
                          patch_layer: int, patching_type: str = "full") -> str:
        """Perform a single forward pass with patching and extract the first predicted token."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)


        answer_token_ids = self.tokenizer("Answer:", add_special_tokens=False)["input_ids"]
        input_ids = inputs["input_ids"][0].tolist()


        answer_pos = None
        for i in range(len(input_ids) - len(answer_token_ids) + 1):
            if input_ids[i:i+len(answer_token_ids)] == answer_token_ids:
                answer_pos = i + len(answer_token_ids) - 1

        if answer_pos is None:
            print("Warning: Could not find 'Answer:' token sequence, using last position")
            answer_pos = -1



        def hook_fn_full(module, input, output):
            if patch_layer in patch_activations:
                hidden_states = output[0]
                hidden_states[:, answer_pos, :] = patch_activations[patch_layer]
                return (hidden_states,) + output[1:]
            return output

        def hook_fn_component(module, input, output):
            if patch_layer in patch_activations:
                if isinstance(output, tuple):
                    component_output = output[0]
                    component_output[:, answer_pos, :] = patch_activations[patch_layer]
                    return (component_output,) + output[1:]
                else:
                    output[:, answer_pos, :] = patch_activations[patch_layer]
                    return output
            return output


        handle = None
        if patch_layer < len(self.model.model.layers):
            if patching_type == "full":
                handle = self.model.model.layers[patch_layer].register_forward_hook(hook_fn_full)
            elif patching_type == "mlp_only":
                if hasattr(self.model.model.layers[patch_layer], 'mlp'):
                    handle = self.model.model.layers[patch_layer].mlp.register_forward_hook(hook_fn_component)
            elif patching_type == "attention_only":
                if hasattr(self.model.model.layers[patch_layer], 'self_attn'):
                    handle = self.model.model.layers[patch_layer].self_attn.register_forward_hook(hook_fn_component)

        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :]
                predicted_token_id = torch.argmax(logits, dim=-1)
                predicted_text = self.tokenizer.decode(predicted_token_id, skip_special_tokens=True)

        finally:
            if handle:
                handle.remove()

        return predicted_text.strip()

    def get_token_logits(self, prompt: str, target_tokens: List[str]) -> Dict[str, float]:
        """Get logits for specific target tokens."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]


        token_logits = {}
        for token in target_tokens:
            token_id = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_id) == 1:
                token_logits[token] = logits[0, token_id[0]].item()
            else:
                token_logits[token] = float('-inf')

        return token_logits

    def patch_and_get_logits(self, prompt: str, patch_activations: Dict[int, torch.Tensor],
                           patch_layer: int, target_tokens: List[str],
                           patching_type: str = "full") -> Dict[str, float]:
        """Perform patching and return logits for target tokens."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)


        answer_token_ids = self.tokenizer("Answer:", add_special_tokens=False)["input_ids"]
        input_ids = inputs["input_ids"][0].tolist()


        answer_pos = None
        for i in range(len(input_ids) - len(answer_token_ids) + 1):
            if input_ids[i:i+len(answer_token_ids)] == answer_token_ids:
                answer_pos = i + len(answer_token_ids) - 1

        if answer_pos is None:
            answer_pos = -1

        def hook_fn_full(module, input, output):
            if patch_layer in patch_activations:
                hidden_states = output[0]
                hidden_states[:, answer_pos, :] = patch_activations[patch_layer]
                return (hidden_states,) + output[1:]
            return output

        def hook_fn_component(module, input, output):
            if patch_layer in patch_activations:
                if isinstance(output, tuple):
                    component_output = output[0]
                    component_output[:, answer_pos, :] = patch_activations[patch_layer]
                    return (component_output,) + output[1:]
                else:
                    output[:, answer_pos, :] = patch_activations[patch_layer]
                    return output
            return output


        handle = None
        if patch_layer < len(self.model.model.layers):
            if patching_type == "full":
                handle = self.model.model.layers[patch_layer].register_forward_hook(hook_fn_full)
            elif patching_type == "mlp_only":
                if hasattr(self.model.model.layers[patch_layer], 'mlp'):
                    handle = self.model.model.layers[patch_layer].mlp.register_forward_hook(hook_fn_component)
            elif patching_type == "attention_only":
                if hasattr(self.model.model.layers[patch_layer], 'self_attn'):
                    handle = self.model.model.layers[patch_layer].self_attn.register_forward_hook(hook_fn_component)

        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :]


            token_logits = {}
            for token in target_tokens:
                token_id = self.tokenizer.encode(token, add_special_tokens=False)
                if len(token_id) == 1:
                    token_logits[token] = logits[0, token_id[0]].item()
                else:
                    token_logits[token] = float('-inf')

        finally:
            if handle:
                handle.remove()

        return token_logits

    def run_stability_experiment(self, stable_poscar: str, strained_poscar: str,
                               patching_types: List[str] = ["full", "mlp_only", "attention_only"]) -> Dict:
        """Run the complete stability judgment experiment with multiple patching types."""
        print(f"Running stability judgment experiment with patching types: {patching_types}")


        clean_prompt = self.create_clean_prompt(stable_poscar, strained_poscar)
        corrupted_prompt = self.create_corrupted_prompt(stable_poscar, strained_poscar)

        print(f"Clean prompt length: {len(self.tokenizer(clean_prompt).input_ids)} tokens")
        print(f"Corrupted prompt length: {len(self.tokenizer(corrupted_prompt).input_ids)} tokens")

        print('\nClean prompt:', clean_prompt)


        target_tokens = ['A', 'B']


        print("\nGetting baseline outputs and logits...")
        clean_output = self.patch_and_predict(clean_prompt, {}, -1, "full")
        corrupted_output = self.patch_and_predict(corrupted_prompt, {}, -1, "full")


        clean_logits = self.get_token_logits(clean_prompt, target_tokens)
        corrupted_logits = self.get_token_logits(corrupted_prompt, target_tokens)




        clean_logit_diff = clean_logits['A'] - clean_logits['B']
        corrupted_logit_diff = corrupted_logits['A'] - corrupted_logits['B']

        clean_choice = self.extract_answer_choice(clean_output)
        corrupted_choice = self.extract_answer_choice(corrupted_output)

        print(f"Clean output: '{clean_output}' -> {clean_choice}")
        print(f"Corrupted output: '{corrupted_output}' -> {corrupted_choice}")
        print(f"Clean logit diff (A-B): {clean_logit_diff:.3f}")
        print(f"Corrupted logit diff (A-B): {corrupted_logit_diff:.3f}")

        print(f"\nDEBUG - Expected behavior:")
        print(f"Clean should choose 'A' (Structure A: stable)")
        print(f"Corrupted should choose 'B' (Structure B: stable)")


        baseline_valid = True
        skip_reason = None

        if clean_choice != 'A':
            print(f"❌ Clean prompt failed: expected 'A', got '{clean_choice}'. Skipping activation patching.")
            baseline_valid = False
            skip_reason = f"Clean prompt answered '{clean_choice}' instead of 'A'"

        if corrupted_choice != 'B':
            print(f"❌ Corrupted prompt failed: expected 'B' (stable structure), got '{corrupted_choice}'. Skipping activation patching.")
            baseline_valid = False
            skip_reason = f"Corrupted prompt answered '{corrupted_choice}' instead of 'B'"


        all_results = {}

        if not baseline_valid:

            for patching_type in patching_types:
                all_results[patching_type] = {
                    'baseline': {
                        'clean_output': clean_output,
                        'corrupted_output': corrupted_output,
                        'clean_choice': clean_choice,
                        'corrupted_choice': corrupted_choice,
                        'clean_logits': clean_logits,
                        'corrupted_logits': corrupted_logits,
                        'clean_logit_diff': clean_logit_diff,
                        'corrupted_logit_diff': corrupted_logit_diff,
                        'validation_passed': False,
                        'skip_reason': skip_reason
                    },
                    'layer_results': [],
                    'onset_layer': None,
                    'skipped': True,
                    'patching_type': patching_type
                }
            return all_results

        print("✅ Baseline validation passed: Clean chose 'A', corrupted chose 'B'. Proceeding with activation patching...")


        print(f"\nCaching clean activations for all patching types...")
        cached_activations = {}
        for patching_type in patching_types:
            print(f"Caching {patching_type} activations...")
            cached_activations[patching_type] = self.cache_activations(clean_prompt, patching_type)

        num_layers = self.model.config.num_hidden_layers


        for patching_type in patching_types:
            print(f"\nRunning layer-by-layer {patching_type} patching...")
            if patching_type == "full":
                print("Note: Each layer test patches the full residual stream at the 'Answer:' token position")
            elif patching_type == "mlp_only":
                print("Note: Each layer test patches only MLP output at the 'Answer:' token position")
            elif patching_type == "attention_only":
                print("Note: Each layer test patches only attention output at the 'Answer:' token position")

            results = {
                'baseline': {
                    'clean_output': clean_output,
                    'corrupted_output': corrupted_output,
                    'clean_choice': clean_choice,
                    'corrupted_choice': corrupted_choice,
                    'clean_logits': clean_logits,
                    'corrupted_logits': corrupted_logits,
                    'clean_logit_diff': clean_logit_diff,
                    'corrupted_logit_diff': corrupted_logit_diff
                },
                'layer_results': [],
                'patching_type': patching_type
            }

            onset_layer = None
            clean_activations = cached_activations[patching_type]

            for layer in tqdm(range(num_layers), desc=f"Patching layers ({patching_type})"):
                patched_output = self.patch_and_predict(
                    corrupted_prompt, clean_activations, layer, patching_type
                )
                patched_choice = self.extract_answer_choice(patched_output)


                patched_logits = self.patch_and_get_logits(
                    corrupted_prompt, clean_activations, layer, target_tokens, patching_type
                )



                patched_logit_diff = patched_logits['A'] - patched_logits['B']
                logit_diff_improvement = patched_logit_diff - corrupted_logit_diff



                choice_changed = (patched_choice != corrupted_choice)
                matches_clean = (patched_choice == clean_choice)

                layer_result = {
                    'layer': layer,
                    'output': patched_output,
                    'choice': patched_choice,
                    'choice_changed': choice_changed,
                    'matches_clean': matches_clean,
                    'logits': patched_logits,
                    'logit_diff': patched_logit_diff,
                    'logit_diff_improvement': logit_diff_improvement
                }
                results['layer_results'].append(layer_result)



                if onset_layer is None and patched_choice == 'A':
                    onset_layer = layer
                    print(f"Stability reasoning onset detected at layer {layer}: model now chooses like clean prompt (A) (logit_diff: {patched_logit_diff:.3f}, improvement: {logit_diff_improvement:.3f})")

            results['onset_layer'] = onset_layer
            all_results[patching_type] = results

        return all_results

    def analyze_results(self, results: Dict) -> Dict:
        """Analyze patching results with logit difference analysis."""
        if results.get('skipped', False):
            return {
                'onset_layer': None,
                'total_layers': 0,
                'choice_progression': [],
                'logit_analysis': {'skipped': True}
            }

        analysis = {
            'onset_layer': results['onset_layer'],
            'total_layers': len(results['layer_results']),
            'choice_progression': [],
            'logit_analysis': {}
        }


        logit_diffs = [lr['logit_diff'] for lr in results['layer_results']]
        improvements = [lr['logit_diff_improvement'] for lr in results['layer_results']]

        if logit_diffs:
            analysis['logit_analysis'] = {
                'max_logit_diff': max(logit_diffs),
                'max_logit_diff_layer': logit_diffs.index(max(logit_diffs)),
                'max_improvement': max(improvements),
                'max_improvement_layer': improvements.index(max(improvements)),
                'mean_logit_diff': np.mean(logit_diffs),
                'mean_improvement': np.mean(improvements),
                'baseline_clean_logit_diff': results['baseline']['clean_logit_diff'],
                'baseline_corrupted_logit_diff': results['baseline']['corrupted_logit_diff']
            }

        prev_choice = results['baseline']['corrupted_choice']

        for layer_result in results['layer_results']:
            choice = layer_result['choice']
            changed = choice != prev_choice

            analysis['choice_progression'].append({
                'layer': layer_result['layer'],
                'choice': choice,
                'changed_from_previous': changed,
                'logit_diff': layer_result['logit_diff'],
                'logit_diff_improvement': layer_result['logit_diff_improvement']
            })
            prev_choice = choice

        return analysis


def load_stability_test_data(num_samples: int = None, csv_path: str = None) -> List[Dict]:
    """Load test cases from CSV file for stability experiments."""
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"Warning: CSV file not found at {csv_path}")
        print(f"Please run: python utils/paired_data_prepare_stability.py --n_samples {num_samples or 100}")
        return []

    df = pd.read_csv(csv_path)


    stability_df = df[df['perturbation_type'] == 'lattice_strain']
    if num_samples:
        stability_df = stability_df.head(num_samples)

    print(f"Loading {len(stability_df)} stability judgment test cases")

    test_cases = []
    for _, row in stability_df.iterrows():
        test_case = {
            'pair_id': row['pair_id'],
            'stable_poscar': row['original_poscar'],
            'strained_poscar': row['strained_poscar'],
            'perturbation_description': row['perturbation_description'],
            'energy_shift': row['energy_shift'],
            'strain_magnitude': row['strain_magnitude'],
            'original_e_hull': row['original_e_hull_distance'],
            'strained_e_hull': row['strained_e_hull_distance'],
            'material_id': row.get('original_mbid', 'Unknown'),
            'formula': row.get('original_composition', 'Unknown')
        }
        test_cases.append(test_case)

    return test_cases


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run stability judgment experiment with component decomposition.')
    parser.add_argument('--output_dir', type=str, default='results/experiment3_2/stability_judge',
                       help='Output directory for saving results (default: results/experiment3_2/stability_judge)')
    parser.add_argument('--patching_type', type=str, default='full', choices=['full', 'mlp_only', 'attention_only'],
                       help='Type of patching to perform: full (residual stream), mlp_only, or attention_only (default: full)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of test samples to run (default: all available)')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.1-70B-Instruct",
                       help='Model name or path (default: meta-llama/Llama-3.1-70B-Instruct)')
    parser.add_argument('--csv_path', type=str, default='data/stability_pairs_1000.csv',
                       help='Path to CSV file with stability pairs (default: data/stability_pairs_1000.csv)')
    return parser.parse_args()


def main():
    """Run the stability judgment experiment with all patching types."""
    args = parse_args()


    patching_types = ["full", "mlp_only", "attention_only"]

    print(f"Running stability judgment experiment with:")
    print(f"  Patching types: {patching_types}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Model: {args.model_name}")
    print(f"  Number of samples: {args.num_samples or 'all'}")


    judge = StabilityJudge(model_name=args.model_name)
    judge.load_model()


    test_cases = load_stability_test_data(num_samples=args.num_samples, csv_path=args.csv_path)


    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    all_results = {ptype: [] for ptype in patching_types}
    validation_stats = {ptype: {'passed': 0, 'failed_clean': 0, 'failed_corrupted': 0, 'total': 0}
                       for ptype in patching_types}
    start_idx = 0


    for patching_type in patching_types:
        temp_filename = f"temp_stability_judge_{patching_type}_results.json"
        temp_path = output_dir / temp_filename
        if temp_path.exists():
            print(f"Found existing temporary results for {patching_type} at {temp_path}")
            try:
                with open(temp_path, "r") as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, dict) and 'results' in existing_data:
                        all_results[patching_type] = existing_data['results']
                        validation_stats[patching_type] = existing_data.get('validation_stats', validation_stats[patching_type])
                        start_idx = max(start_idx, existing_data.get('completed_test_cases', 0))
            except Exception as e:
                print(f"Could not load existing results for {patching_type}: {e}. Starting fresh for this type.")
                all_results[patching_type] = []

    if start_idx > 0:
        print(f"Resuming from test case {start_idx + 1}/{len(test_cases)}")

    for i, test_case in enumerate(test_cases):

        if i < start_idx:
            continue
        print(f"\n{'='*60}")
        print(f"Running test case {i+1}/{len(test_cases)}")
        print(f"Material: {test_case.get('formula', 'Unknown')}")
        print(f"Strain: {test_case.get('perturbation_description', 'Unknown')}")
        print(f"Energy shift: {test_case.get('energy_shift', 0):.4f} eV/atom")
        print(f"{'='*60}")


        results_by_type = judge.run_stability_experiment(
            test_case['stable_poscar'],
            test_case['strained_poscar'],
            patching_types
        )


        for patching_type in patching_types:
            results = results_by_type[patching_type]


            validation_stats[patching_type]['total'] += 1
            if results.get('skipped', False):
                baseline = results['baseline']
                if baseline['clean_choice'] != 'A':
                    validation_stats[patching_type]['failed_clean'] += 1
                elif baseline['corrupted_choice'] != 'B':
                    validation_stats[patching_type]['failed_corrupted'] += 1
            else:
                validation_stats[patching_type]['passed'] += 1


            analysis = judge.analyze_results(results)
            results['analysis'] = analysis


            all_results[patching_type].append({
                'test_case': i,
                'results': results,
                'test_metadata': test_case
            })


        print(f"\nSummary for test case {i+1}:")
        for patching_type in patching_types:
            results = results_by_type[patching_type]
            if results.get('skipped', False):
                print(f"  {patching_type}: ⚠️ Skipped - {results['baseline']['skip_reason']}")
            else:
                onset = results.get('onset_layer')
                if onset is not None:
                    print(f"  {patching_type}: Onset at layer {onset}")
                else:
                    print(f"  {patching_type}: No onset detected")


        print(f"Saving intermediate results after test case {i+1}...")
        for patching_type in patching_types:
            temp_filename = f"temp_stability_judge_{patching_type}_results.json"
            temp_result_path = output_dir / temp_filename


            intermediate_data = {
                'results': all_results[patching_type],
                'validation_stats': validation_stats[patching_type],
                'completed_test_cases': i + 1,
                'total_test_cases': len(test_cases),
                'timestamp': str(pd.Timestamp.now())
            }

            with open(temp_result_path, "w") as f:
                json.dump(intermediate_data, f, indent=2, default=str)

        print(f"Intermediate results saved for all patching types")


    for patching_type in patching_types:
        result_filename = f"stability_judge_{patching_type}_results.json"
        final_result_path = output_dir / result_filename

        with open(final_result_path, "w") as f:
            json.dump(all_results[patching_type], f, indent=2, default=str)


        print(f"\n{'='*60}")
        print(f"VALIDATION STATISTICS - {patching_type.upper()}")
        print(f"{'='*60}")
        stats = validation_stats[patching_type]
        print(f"Total test cases: {stats['total']}")
        print(f"Passed validation: {stats['passed']} ({stats['passed']/stats['total']*100:.1f}%)")
        print(f"Failed - Clean prompt wrong: {stats['failed_clean']}")
        print(f"Failed - Corrupted prompt wrong: {stats['failed_corrupted']}")


        valid_onsets = []
        for result in all_results[patching_type]:
            if not result['results'].get('skipped', False):
                onset = result['results']['onset_layer']
                if onset is not None:
                    valid_onsets.append(onset)

        if valid_onsets:
            print(f"\nONSET LAYER STATISTICS (n={len(valid_onsets)}):")
            print(f"Mean onset layer: {np.mean(valid_onsets):.1f}")
            print(f"Median onset layer: {np.median(valid_onsets):.1f}")
            print(f"Range: {min(valid_onsets)} - {max(valid_onsets)}")
        else:
            print(f"No onsets detected for {patching_type}")

    print(f"\nAll results saved to {output_dir}")
    print("Files created:")
    for patching_type in patching_types:
        print(f"  - stability_judge_{patching_type}_results.json")


if __name__ == "__main__":
    main()
