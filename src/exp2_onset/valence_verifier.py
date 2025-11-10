#!/usr/bin/env python3
"""
Experiment 2: Valence Verifier
Activation patching to test the model's concept of oxidation states and charge neutrality.
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

class ValenceVerifier:
    """Activation patching for oxidation state and charge neutrality analysis."""

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

    def create_clean_prompt(self, ions_text: str) -> str:
        """Clean prompt: Contains charge-neutral ions, model should predict 'A' (Yes)."""
        return ions_text

    def create_corrupted_prompt(self, ions_text: str) -> str:
        """Corrupted prompt: Contains charge-imbalanced ions, model should predict 'B' (No)."""
        return ions_text

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

    def run_valence_experiment(self, clean_prompt: str, corrupted_prompt: str,
                             patching_types: List[str] = ["full", "mlp_only", "attention_only"]) -> Dict:
        """Run the complete valence verification experiment with multiple patching types."""
        print(f"Running valence verification experiment with patching types: {patching_types}")

        print(f"Clean prompt length: {len(self.tokenizer(clean_prompt).input_ids)} tokens")
        print(f"Corrupted prompt length: {len(self.tokenizer(corrupted_prompt).input_ids)} tokens")


        target_tokens = ['A', 'B']


        print("\nGetting baseline outputs and logits...")
        clean_output = self.patch_and_predict(clean_prompt, {}, -1, "full")
        corrupted_output = self.patch_and_predict(corrupted_prompt, {}, -1, "full")


        clean_logits = self.get_token_logits(clean_prompt, target_tokens)
        corrupted_logits = self.get_token_logits(corrupted_prompt, target_tokens)



        clean_neutrality_margin = clean_logits['A'] - clean_logits['B']
        corrupted_neutrality_margin = corrupted_logits['A'] - corrupted_logits['B']

        clean_choice = self.extract_answer_choice(clean_output)
        corrupted_choice = self.extract_answer_choice(corrupted_output)

        print(f"Clean output: '{clean_output}' -> {clean_choice}")
        print(f"Corrupted output: '{corrupted_output}' -> {corrupted_choice}")
        print(f"Clean neutrality margin (A-B): {clean_neutrality_margin:.3f}")
        print(f"Corrupted neutrality margin (A-B): {corrupted_neutrality_margin:.3f}")

        print(f"\nDEBUG - Expected behavior:")
        print(f"Clean should choose 'A' (charge neutral ions)")
        print(f"Corrupted should choose 'B' (charge imbalanced ions)")


        baseline_valid = True
        skip_reason = None

        if clean_choice != 'A':
            print(f"❌ Clean prompt failed: expected 'A', got '{clean_choice}'. Skipping activation patching.")
            baseline_valid = False
            skip_reason = f"Clean prompt answered '{clean_choice}' instead of 'A'"

        if corrupted_choice != 'B':
            print(f"❌ Corrupted prompt failed: expected 'B' (imbalanced ions), got '{corrupted_choice}'. Skipping activation patching.")
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
                        'clean_neutrality_margin': clean_neutrality_margin,
                        'corrupted_neutrality_margin': corrupted_neutrality_margin,
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
                    'clean_neutrality_margin': clean_neutrality_margin,
                    'corrupted_neutrality_margin': corrupted_neutrality_margin
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


                patched_neutrality_margin = patched_logits['A'] - patched_logits['B']
                margin_improvement = patched_neutrality_margin - corrupted_neutrality_margin



                choice_changed = (patched_choice != corrupted_choice)
                matches_clean = (patched_choice == clean_choice)

                layer_result = {
                    'layer': layer,
                    'output': patched_output,
                    'choice': patched_choice,
                    'choice_changed': choice_changed,
                    'matches_clean': matches_clean,
                    'logits': patched_logits,
                    'neutrality_margin': patched_neutrality_margin,
                    'margin_improvement': margin_improvement
                }
                results['layer_results'].append(layer_result)



                if onset_layer is None and patched_choice == 'A':
                    onset_layer = layer
                    print(f"Charge neutrality reasoning onset detected at layer {layer}: model now chooses like clean prompt (A) (neutrality_margin: {patched_neutrality_margin:.3f}, improvement: {margin_improvement:.3f})")

            results['onset_layer'] = onset_layer
            all_results[patching_type] = results

        return all_results

    def analyze_results(self, results: Dict) -> Dict:
        """Analyze patching results with charge neutrality margin analysis."""
        if results.get('skipped', False):
            return {
                'onset_layer': None,
                'total_layers': 0,
                'choice_progression': [],
                'neutrality_margin_analysis': {'skipped': True}
            }

        analysis = {
            'onset_layer': results['onset_layer'],
            'total_layers': len(results['layer_results']),
            'choice_progression': [],
            'neutrality_margin_analysis': {}
        }


        neutrality_margins = [lr['neutrality_margin'] for lr in results['layer_results']]
        margin_improvements = [lr['margin_improvement'] for lr in results['layer_results']]

        if neutrality_margins:
            analysis['neutrality_margin_analysis'] = {
                'max_neutrality_margin': max(neutrality_margins),
                'max_neutrality_margin_layer': neutrality_margins.index(max(neutrality_margins)),
                'max_margin_improvement': max(margin_improvements),
                'max_margin_improvement_layer': margin_improvements.index(max(margin_improvements)),
                'mean_neutrality_margin': np.mean(neutrality_margins),
                'mean_margin_improvement': np.mean(margin_improvements),
                'baseline_clean_neutrality_margin': results['baseline']['clean_neutrality_margin'],
                'baseline_corrupted_neutrality_margin': results['baseline']['corrupted_neutrality_margin']
            }

        prev_choice = results['baseline']['corrupted_choice']

        for layer_result in results['layer_results']:
            choice = layer_result['choice']
            changed = choice != prev_choice

            analysis['choice_progression'].append({
                'layer': layer_result['layer'],
                'choice': choice,
                'changed_from_previous': changed,
                'neutrality_margin': layer_result['neutrality_margin'],
                'margin_improvement': layer_result['margin_improvement']
            })
            prev_choice = choice

        return analysis


def load_valence_test_data(num_samples: int = None, csv_path: str = None) -> List[Dict]:
    """Load test cases from CSV file for valence verification experiments."""
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"Warning: CSV file not found at {csv_path}")
        print(f"Please ensure the valence pairs CSV file exists")
        return []

    df = pd.read_csv(csv_path)

    if num_samples:
        df = df.head(num_samples)

    print(f"Loading {len(df)} valence verification test cases")

    test_cases = []
    for _, row in df.iterrows():
        test_case = {
            'pair_id': row['pair_id'],
            'clean_prompt': row['clean_prompt'],
            'corrupted_prompt': row['corrupted_prompt'],
            'clean_ions': row['clean_ions'],
            'corrupted_ions': row['corrupted_ions'],
            'clean_charge_sum': row['clean_charge_sum'],
            'corrupted_charge_sum': row['corrupted_charge_sum'],
            'charge_violation_magnitude': row['charge_violation_magnitude'],
            'clean_expected_answer': row['clean_expected_answer'],
            'corrupted_expected_answer': row['corrupted_expected_answer'],
            'original_formula': row['original_formula'],
            'clean_ion_count': row['clean_ion_count'],
            'corrupted_ion_count': row['corrupted_ion_count'],
        }
        test_cases.append(test_case)

    return test_cases


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run valence verifier experiment with component decomposition.')
    parser.add_argument('--output_dir', type=str, default='results/experiment3/3_6_valence_verifier/1000',
                       help='Output directory for saving results')
    parser.add_argument('--patching_type', type=str, default='full', choices=['full', 'mlp_only', 'attention_only'],
                       help='Type of patching to perform: full (residual stream), mlp_only, or attention_only (default: full)')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of test samples to run (default: all available)')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.1-70B-Instruct",
                       help='Model name or path (default: meta-llama/Llama-3.1-70B-Instruct)')
    parser.add_argument('--csv_path', type=str, default='data/valence_pairs_1000.csv',
                       help='Path to CSV file with valence pairs (default: data/valence_pairs_1000.csv)')
    return parser.parse_args()


def main():
    """Run the valence verifier experiment with all patching types."""
    args = parse_args()


    patching_types = ["full", "mlp_only", "attention_only"]

    print(f"Running valence verifier experiment with:")
    print(f"  Patching types: {patching_types}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Model: {args.model_name}")
    print(f"  Number of samples: {args.num_samples or 'all'}")


    verifier = ValenceVerifier(model_name=args.model_name)
    verifier.load_model()


    test_cases = load_valence_test_data(num_samples=args.num_samples, csv_path=args.csv_path)


    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    all_results = {ptype: [] for ptype in patching_types}
    validation_stats = {ptype: {'passed': 0, 'failed_clean': 0, 'failed_corrupted': 0, 'total': 0}
                       for ptype in patching_types}
    start_idx = 0


    for patching_type in patching_types:
        temp_filename = f"temp_valence_verifier_{patching_type}_results.json"
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
        print(f"\n{'='*80}")
        print(f"Running test case {i+1}/{len(test_cases)}")
        print(f"Pair ID: {test_case.get('pair_id', 'Unknown')}")
        print(f"Clean ions: {test_case.get('clean_ions', 'Unknown')} (sum: {test_case.get('clean_charge_sum', 0)})")
        print(f"Corrupted ions: {test_case.get('corrupted_ions', 'Unknown')} (sum: {test_case.get('corrupted_charge_sum', 0)})")
        print(f"Charge violation: {test_case.get('charge_violation_magnitude', 0)}")
        print(f"Original formula: {test_case.get('original_formula', 'Unknown')}")
        print(f"{'='*80}")


        results_by_type = verifier.run_valence_experiment(
            test_case['clean_prompt'],
            test_case['corrupted_prompt'],
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


            analysis = verifier.analyze_results(results)
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
                    print(f"  {patching_type}: Charge neutrality reasoning onset at layer {onset}")
                else:
                    print(f"  {patching_type}: No charge neutrality reasoning onset detected")


        print(f"Saving intermediate results after test case {i+1}...")
        for patching_type in patching_types:
            temp_filename = f"temp_valence_verifier_{patching_type}_results.json"
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
        result_filename = f"valence_verifier_{patching_type}_results.json"
        final_result_path = output_dir / result_filename

        with open(final_result_path, "w") as f:
            json.dump(all_results[patching_type], f, indent=2, default=str)


        print(f"\n{'='*80}")
        print(f"VALIDATION STATISTICS - {patching_type.upper()}")
        print(f"{'='*80}")
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
            print(f"\nCHARGE NEUTRALITY REASONING ONSET STATISTICS (n={len(valid_onsets)}):")
            print(f"Mean onset layer: {np.mean(valid_onsets):.1f}")
            print(f"Median onset layer: {np.median(valid_onsets):.1f}")
            print(f"Range: {min(valid_onsets)} - {max(valid_onsets)}")
        else:
            print(f"No charge neutrality reasoning onsets detected for {patching_type}")

    print(f"\nAll results saved to {output_dir}")
    print("Files created:")
    for patching_type in patching_types:
        print(f"  - valence_verifier_{patching_type}_results.json")

    print(f"\n🎯 Experiment 3_6 'Valence Verifier' Complete!")
    print(f"📊 This experiment tests the model's internal concept of oxidation states")
    print(f"   and charge neutrality through simple ion list reasoning.")
    print(f"🔬 Expected: MLP patching should show higher success rates than attention patching")
    print(f"   since this requires parametric chemical knowledge + mathematical reasoning.")
    print(f"⚖️ Attention-easy (ion parsing) vs MLP-hard (chemical knowledge + math)")


if __name__ == "__main__":
    main()
