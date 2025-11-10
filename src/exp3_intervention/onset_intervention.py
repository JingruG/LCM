#!/usr/bin/env python3
"""
Experiment 3: Onset Layer Intervention
Injects stability vectors into the generation process for unstable crystals.
The stability vector is created by averaging stable reference structures.
"""

import torch
import numpy as np
import pandas as pd
import json
import pickle
import os
import re
from typing import Dict, List, Tuple, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from pathlib import Path
from contextlib import contextmanager
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.structure_parser import parse_structure_string, structure_to_string
from utils.chgnet_evaluator import CHGNetEvaluator


class OnsetLayerIntervention:
    """Stability vector injection at onset layer for crystal generation control."""

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-70B-Instruct",
                 chgnet_timeout: int = 60, target_layer: int = 39):
        self.model_name = model_name
        self.target_layer = target_layer
        self.tokenizer = None
        self.model = None
        self.evaluator = CHGNetEvaluator(chgnet_timeout=chgnet_timeout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.stored_activations = {}
        self.parent_a_stability_vectors = {}

    def initialize_model(self):
        """Initialize tokenizer and model."""
        print(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side='left',
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
        )

        self.model.eval()
        print(f"Model loaded on device: {self.device}")

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load crystal parent data without filtering."""
        print(f"Loading data from {csv_path}")

        df = pd.read_csv(csv_path)
        print(f"Total pairs loaded: {len(df)}")


        df['parent_b_index'] = df['pair_id'].str.split('_').str[1].astype(int)
        df['parent_a_id'] = df['pair_id'].str.rsplit('_', n=1).str[0]


        def count_elements(composition_str):
            """Count number of unique elements in composition string."""
            try:
                from pymatgen.core import Composition
                comp = Composition(composition_str)
                return len(comp.elements)
            except:
                import re
                elements = re.findall(r'[A-Z][a-z]?', composition_str)
                return len(set(elements))

        df['parent_a_element_count'] = df['parent_A_composition'].apply(count_elements)

        print(f"Unique Parent A structures: {df['parent_a_id'].nunique()}")
        print(f"Data loaded successfully with derived columns")

        return df

    def apply_standard_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply standard filtering criteria for the general experiment."""
        print(f"\n=== APPLYING STANDARD FILTERS ===")
        print(f"Starting with {len(df)} pairs")


        df = df[df['parent_A_e_hull'] >= 0.1].copy()
        print(f"After unstable Parent A filter (ehull >= 0.1): {len(df)}")


        df = df[df['parent_B_e_hull'] <= 0.05].copy()
        print(f"After stable Parent B filter (ehull <= 0.05): {len(df)}")


        df = df[df['parent_a_element_count'] == 2].copy()
        print(f"After Parent A element count filter (== 2): {len(df)}")


        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"Data shuffled with random seed 42")
        print(f"Final filtered dataset: {len(df)} pairs")

        return df

    def filter_for_specific_parent_a_ids(self, df: pd.DataFrame, target_parent_a_ids: List[str]) -> pd.DataFrame:
        """Filter data for specific parent A IDs without other filtering criteria."""
        print(f"\n=== FILTERING FOR SPECIFIC PARENT A IDs ===")
        print(f"Target parent A IDs: {target_parent_a_ids}")
        print(f"Starting with {len(df)} pairs")


        df = df[df['parent_a_id'].isin(target_parent_a_ids)].copy()
        print(f"After filtering for specific parent A IDs: {len(df)}")


        df = df[df['parent_B_e_hull'] <= 0.05].copy()
        print(f"After stable Parent B filter (ehull <= 0.05): {len(df)}")


        found_parent_a_ids = df['parent_a_id'].unique()
        print(f"Found parent A IDs: {sorted(found_parent_a_ids)}")

        for target_id in target_parent_a_ids:
            if target_id not in found_parent_a_ids:
                print(f"WARNING: Target parent A ID {target_id} not found!")
            else:
                count = len(df[df['parent_a_id'] == target_id])
                target_data = df[df['parent_a_id'] == target_id]
                parent_a_info = target_data.iloc[0]
                print(f"✓ Parent A ID {target_id}: {count} pairs")
                print(f"  Composition: {parent_a_info['parent_A_composition']}")
                print(f"  E_hull: {parent_a_info['parent_A_e_hull']:.6f}")
                print(f"  Element count: {parent_a_info['parent_a_element_count']}")

        return df

    def preprocess_poscar(self, poscar_content: str) -> str:
        """Preprocess POSCAR content with structure_to_string using precision 8."""
        try:

            structure = parse_structure_string(poscar_content, format_type='poscar')
            if structure is not None:
                return structure_to_string(structure, precision=8, fmt='poscar')
            else:
                print("  → Warning: Could not parse POSCAR, using original")
                return poscar_content
        except Exception as e:
            print(f"  → Warning: Error preprocessing POSCAR: {e}, using original")
            return poscar_content

    def format_crystal_prompt(self, unstable_poscar: str, unstable_comp: str) -> str:
        """Format crystal structure prompt for unstable Parent A with intervention vector."""
        return f"""Generate a new, more thermodynamically stable crystal structure by improving the following unstable crystal structure for {unstable_comp}.

```poscar
{unstable_poscar}
```

REQUIREMENTS:
- Primary goal: Lower deformation energy
- Allow compositional changes for better stability
- Avoid copying parent coordinates
- Ensure valid coordination and no overlapping atoms

OUTPUT ONLY the POSCAR string:"""

    @contextmanager
    def activation_hook(self, layer_idx: int, storage_key: str):
        """Context manager for capturing activations at specific layer."""

        def hook_fn(module, input, output):

            if isinstance(output, tuple):
                self.stored_activations[storage_key] = output[0].detach().clone()
            else:
                self.stored_activations[storage_key] = output.detach().clone()


        layer = self.model.model.layers[layer_idx]
        handle = layer.register_forward_hook(hook_fn)

        try:
            yield
        finally:
            handle.remove()

    @contextmanager
    def intervention_hook(self, layer_idx: int, intervention_vector: torch.Tensor, alpha: float = 1.0):
        """Context manager for applying intervention at specific layer."""

        def hook_fn(module, input, output):

            if isinstance(output, tuple):
                hidden_states = output[0]
                batch_size, seq_len, hidden_dim = hidden_states.shape



                modified_hidden = hidden_states.clone()
                if intervention_vector.dim() == 2:
                    modified_hidden[:, -1, :] = modified_hidden[:, -1, :] + alpha * intervention_vector
                else:
                    modified_hidden[:, -1, :] = modified_hidden[:, -1, :] + alpha * intervention_vector.squeeze()


                return (modified_hidden,) + output[1:]
            else:

                modified_output = output.clone()
                if intervention_vector.dim() == 2:
                    modified_output[:, -1, :] = modified_output[:, -1, :] + alpha * intervention_vector
                else:
                    modified_output[:, -1, :] = modified_output[:, -1, :] + alpha * intervention_vector.squeeze()
                return modified_output


        layer = self.model.model.layers[layer_idx]
        handle = layer.register_forward_hook(hook_fn)

        try:
            yield
        finally:
            handle.remove()

    def create_parent_a_stability_vector(self, parent_a_data: pd.DataFrame) -> torch.Tensor:
        """Create a stability vector for a specific Parent A by averaging its 5 Parent B structures.
        Uses FULL FORMATTED PROMPTS for meaningful activations.
        """
        parent_a_id = parent_a_data.iloc[0]['parent_a_id']
        print(f"Computing stability vector for Parent A: {parent_a_id}")
        print(f"Using {len(parent_a_data)} Parent B structures")


        parent_a_row = parent_a_data.iloc[0]
        parent_a_poscar = self.preprocess_poscar(parent_a_row['parent_A_poscar'])
        parent_a_comp = parent_a_row['parent_A_composition']


        print(f"  → Processing unstable Parent A with full prompt")
        unstable_prompt = self.format_crystal_prompt(parent_a_poscar, parent_a_comp)
        unstable_inputs = self.tokenizer(unstable_prompt, return_tensors="pt", truncation=True, max_length=2048)
        unstable_inputs = {k: v.to(self.device) for k, v in unstable_inputs.items()}

        with torch.no_grad():
            with self.activation_hook(self.target_layer, f"unstable_{parent_a_id}"):
                _ = self.model(**unstable_inputs)

        unstable_activation = self.stored_activations[f"unstable_{parent_a_id}"][:, -1, :]


        print(f"  → Processing {len(parent_a_data)} stable Parent B structures with full prompts")
        stable_activations = []
        for idx, row in parent_a_data.iterrows():
            processed_poscar = self.preprocess_poscar(row['parent_B_poscar'])
            stable_prompt = self.format_crystal_prompt(processed_poscar, row['parent_B_composition'])

            inputs = self.tokenizer(stable_prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                with self.activation_hook(self.target_layer, f"stable_{row['pair_id']}"):
                    _ = self.model(**inputs)

            activation = self.stored_activations[f"stable_{row['pair_id']}"]
            stable_activations.append(activation[:, -1, :])


        stable_tensor = torch.stack(stable_activations)
        avg_stable = stable_tensor.mean(dim=0, keepdim=True)


        stability_vector = avg_stable - unstable_activation
        stability_vector = torch.nn.functional.normalize(stability_vector, p=2, dim=-1)

        print(f"  → Stability vector computed, shape: {stability_vector.shape}")
        return stability_vector

    def save_stability_vectors(self, output_dir: str):
        """Save pre-computed stability vectors to file."""
        vectors_path = f"{output_dir}/stability_vectors.pt"
        torch.save(self.parent_a_stability_vectors, vectors_path)
        print(f"Stability vectors saved to: {vectors_path}")

    def load_stability_vectors(self, output_dir: str) -> bool:
        """Load pre-computed stability vectors from file, with fallback paths."""

        vectors_path = f"{output_dir}/stability_vectors.pt"


        fallback_path = "results/experiment4/onset_intervention/39_gt2_0.6/stability_vectors.pt"


        if os.path.exists(vectors_path):
            loaded_vectors = torch.load(vectors_path, map_location=self.device)

            self.parent_a_stability_vectors = {k: v.to(self.device) for k, v in loaded_vectors.items()}
            print(f"Loaded {len(self.parent_a_stability_vectors)} stability vectors from: {vectors_path}")
            return True


        elif os.path.exists(fallback_path):
            loaded_vectors = torch.load(fallback_path, map_location=self.device)

            self.parent_a_stability_vectors = {k: v.to(self.device) for k, v in loaded_vectors.items()}
            print(f"Loaded {len(self.parent_a_stability_vectors)} stability vectors from fallback: {fallback_path}")

            self.save_stability_vectors(output_dir)
            print(f"Copied stability vectors to current output directory: {output_dir}")
            return True

        print("No pre-computed stability vectors found")
        return False

    def generate_with_intervention(self, prompt: str, stability_vector: torch.Tensor,
                                 alpha: float = 1.0, temperature: float = 0.6,
                                 max_new_tokens: int = 5000) -> str:
        """Generate crystal structure with stability vector intervention."""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4000)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            with self.intervention_hook(self.target_layer, stability_vector, alpha):
                if temperature == 0.0:

                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                else:

                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.85,
                        pad_token_id=self.tokenizer.eos_token_id
                    )


        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text

    def run_parent_a_intervention(self, parent_a_data: pd.DataFrame, stability_vector: torch.Tensor, output_dir: str = None) -> Dict:
        """Run intervention experiment for a Parent A using stability vector averaged from all its Parent B structures."""


        parent_a_row = parent_a_data.iloc[0]
        parent_a_id = parent_a_row['parent_a_id']
        parent_a_poscar = parent_a_row['parent_A_poscar']
        parent_a_comp = parent_a_row['parent_A_composition']
        parent_a_ehull = parent_a_row['parent_A_e_hull']


        parent_b_compositions = parent_a_data['parent_B_composition'].tolist()
        parent_b_ehulls = parent_a_data['parent_B_e_hull'].tolist()
        avg_parent_b_ehull = np.mean(parent_b_ehulls)

        print(f"    Processing Parent A {parent_a_id}")
        print(f"    Parent A (unstable): {parent_a_comp}, ehull = {parent_a_ehull:.3f}")
        print(f"    Used {len(parent_a_data)} Parent B structures for stability vector:")
        for i, (comp, ehull) in enumerate(zip(parent_b_compositions, parent_b_ehulls)):
            print(f"      Parent B {i}: {comp}, ehull = {ehull:.3f}")
        print(f"    Average Parent B ehull: {avg_parent_b_ehull:.3f}")


        print(f"      → Preprocessing Parent A POSCAR")
        processed_parent_a_poscar = self.preprocess_poscar(parent_a_poscar)


        print(f"      → Creating Parent A prompt for intervention")
        unstable_prompt = self.format_crystal_prompt(processed_parent_a_poscar, parent_a_comp)

        results = {
            'parent_a_id': parent_a_id,
            'parent_a_composition': parent_a_comp,
            'parent_a_ehull': parent_a_ehull,
            'num_parent_b_used': len(parent_a_data),
            'parent_b_compositions': parent_b_compositions,
            'parent_b_ehulls': parent_b_ehulls,
            'avg_parent_b_ehull': avg_parent_b_ehull,
            'target_layer': self.target_layer,
            'intervention_results': {}
        }

        try:



            alphas = [0.0, 1.0, 2.0]
            temperatures = [0.0, 0.6]
            total_conditions = len(alphas) * len(temperatures)
            condition_idx = 0

            print(f"      → Testing {total_conditions} conditions with pre-computed stability vector...")

            for alpha in alphas:
                for temp in temperatures:
                    condition_idx += 1
                    setting_key = f'alpha_{alpha}_temp_{temp}'
                    print(f"        [{condition_idx}/{total_conditions}] Alpha={alpha}, Temp={temp}")


                    generated_text = self.generate_with_intervention(
                        unstable_prompt, stability_vector, alpha=alpha, temperature=temp
                    )
                    print(f"        → Generation completed ({len(generated_text)} chars)")


                    try:
                        structure = parse_structure_string(generated_text, format_type='poscar')
                        if structure is not None:
                            print(f"        → Structure parsed successfully!")
                            print(f"           Composition: {structure.composition}")
                            print(f"           Atoms: {len(structure)}, Volume: {structure.volume:.2f} Å³")


                            try:
                                e_hull = self.evaluator.calculate_e_hull(structure)

                                if e_hull is not None:
                                    print(f"        → E_hull: {e_hull:.4f} eV/atom")
                                    stability_improvement = parent_a_ehull - e_hull
                                    formation_energy = self.evaluator.calculate_formation_energy(structure)

                                    evaluation = {
                                        'e_hull': e_hull,
                                        'formation_energy': formation_energy,
                                        'valid': abs(formation_energy) < 3.0,
                                        'stable': e_hull < 0.1,
                                        'chgnet_success': True
                                    }

                                    results['intervention_results'][setting_key] = {
                                        'alpha': alpha,
                                        'temperature': temp,
                                        'generated_text': generated_text,
                                        'parsed_successfully': True,
                                        'evaluation': evaluation,
                                        'stability_improvement': stability_improvement,
                                        'structure_data': {
                                            'composition': str(structure.composition),
                                            'num_atoms': len(structure),
                                            'lattice_abc': structure.lattice.abc,
                                            'lattice_angles': structure.lattice.angles,
                                            'volume': structure.volume,
                                            'density': structure.density
                                        }
                                    }
                                    print(f"        → Improvement: {stability_improvement:.4f}")
                                else:
                                    results['intervention_results'][setting_key] = {
                                        'alpha': alpha,
                                        'temperature': temp,
                                        'generated_text': generated_text,
                                        'parsed_successfully': True,
                                        'evaluation': None,
                                        'stability_improvement': 0.0,
                                        'note': 'Structure parsed but no e_hull available'
                                    }

                            except Exception as eval_error:
                                print(f"        → E_hull calculation failed: {eval_error}")
                                results['intervention_results'][setting_key] = {
                                    'alpha': alpha,
                                    'temperature': temp,
                                    'generated_text': generated_text,
                                    'parsed_successfully': True,
                                    'evaluation': None,
                                    'stability_improvement': 0.0,
                                    'error': str(eval_error)
                                }
                        else:
                            print(f"        → Structure parsing failed")
                            results['intervention_results'][setting_key] = {
                                'alpha': alpha,
                                'temperature': temp,
                                'generated_text': generated_text,
                                'parsed_successfully': False,
                                'evaluation': None,
                                'stability_improvement': 0.0
                            }

                    except Exception as e:
                        print(f"        → Error parsing/evaluating: {e}")
                        results['intervention_results'][setting_key] = {
                            'alpha': alpha,
                            'temperature': temp,
                            'generated_text': generated_text,
                            'parsed_successfully': False,
                            'evaluation': None,
                            'error': str(e),
                            'stability_improvement': 0.0
                        }

        except Exception as e:
            print(f"    Error in intervention experiment: {e}")
            results['error'] = str(e)

        return results

    def run_experiment(self, csv_path: str, output_dir: str, max_parent_a: int = 20,
                      specific_parent_a_ids: Optional[List[str]] = None):
        """Run the complete onset intervention experiment with true sequential Parent A processing.

        For each Parent A:
        1. Compute stability vector (once)
        2. Run all generation experiments (multiple alpha/temp combinations)
        3. Move to next Parent A

        Args:
            csv_path: Path to the CSV data file
            output_dir: Directory to save results
            max_parent_a: Maximum number of Parent A structures to process (ignored if specific_parent_a_ids is provided)
            specific_parent_a_ids: List of specific parent A IDs to test (if provided, no other filtering is applied)
        """


        self.initialize_model()


        df = self.load_data(csv_path)


        if specific_parent_a_ids is not None:
            print(f"\nTesting specific parent A IDs: {specific_parent_a_ids}")
            df = self.filter_for_specific_parent_a_ids(df, specific_parent_a_ids)
        else:
            print(f"\nApplying standard filters for general experiment")
            df = self.apply_standard_filters(df)

        if len(df) == 0:
            print("No suitable crystal pairs found after filtering!")
            return


        os.makedirs(output_dir, exist_ok=True)


        unique_parent_a_ids = df['parent_a_id'].unique()[:max_parent_a]
        print(f"\nFound {len(df['parent_a_id'].unique())} total unique Parent A structures")
        print(f"Will process first {len(unique_parent_a_ids)} Parent A structures")

        all_results = []


        for parent_a_idx, parent_a_id in enumerate(unique_parent_a_ids):
            print(f"\n" + "="*80)
            print(f"PROCESSING PARENT A {parent_a_idx + 1}/{len(unique_parent_a_ids)}: {parent_a_id}")
            print("="*80)


            parent_a_data = df[df['parent_a_id'] == parent_a_id].copy()
            print(f"Found {len(parent_a_data)} Parent B structures for this Parent A")


            print(f"\n[STEP 1/2] Computing stability vector for Parent A {parent_a_id}")
            print("─" * 40)
            stability_vector = self.create_parent_a_stability_vector(parent_a_data)
            if stability_vector is None:
                print(f"✗ Failed to compute stability vector, skipping Parent A {parent_a_id}")
                continue
            print(f"✓ Stability vector computed, shape: {stability_vector.shape}, device: {stability_vector.device}")


            print(f"\n[STEP 2/2] Running generation experiment for Parent A {parent_a_id}")
            print("─" * 40)
            print("Using the averaged stability vector from all 5 Parent B structures")


            parent_a_row = parent_a_data.iloc[0].to_dict()

            print(f"\n  → Running generation experiment for Parent A {parent_a_id}")

            try:
                results = self.run_parent_a_intervention(parent_a_data, stability_vector, output_dir=output_dir)
                all_results.append(results)
                parent_a_results = [results]


                temp_file = f"{output_dir}/temp_results_parent_a_{parent_a_id}.json"
                with open(temp_file, 'w') as f:
                    json.dump(results, f, indent=2)

            except Exception as e:
                print(f"  → Error in experiment: {e}")
                parent_a_results = []


            progress_file = f"{output_dir}/parent_a_{parent_a_idx + 1}_{parent_a_id}_complete.json"
            parent_a_summary = {
                'parent_a_id': parent_a_id,
                'parent_a_index': parent_a_idx + 1,
                'total_experiments': len(parent_a_results),
                'results': parent_a_results
            }
            with open(progress_file, 'w') as f:
                json.dump(parent_a_summary, f, indent=2)

            print(f"\n✓ Parent A {parent_a_id} COMPLETED:")
            print(f"  - Stability vector: ✓ Computed/Cached")
            print(f"  - Experiments run: {len(parent_a_results)}")
            print(f"  - Progress saved: {progress_file}")
            print(f"  - Moving to next Parent A...")


            self.stored_activations.clear()


        final_output = {
            'experiment_config': {
                'model_name': self.model_name,
                'target_layer': self.target_layer,
                'csv_path': csv_path,
                'total_parent_a_processed': len(unique_parent_a_ids),
                'total_experiments': len(all_results),
                'processing_method': 'sequential_per_parent_a'
            },
            'results': all_results
        }

        final_file = f"{output_dir}/onset_intervention_results_final.json"
        with open(final_file, 'w') as f:
            json.dump(final_output, f, indent=2)

        print(f"\n" + "="*80)
        print(f"🎉 EXPERIMENT COMPLETED!")
        print("="*80)
        print(f"📊 SUMMARY:")
        print(f"  • Parent A structures processed: {len(unique_parent_a_ids)}")
        print(f"  • Total experiments completed: {len(all_results)}")
        print(f"  • Stability vectors cached: {len(self.parent_a_stability_vectors)}")
        print(f"  • Results saved to: {output_dir}")
        print(f"  • Final results: {final_file}")
        print("="*80)


        self.generate_summary(all_results, output_dir)

    def generate_summary(self, results: List[Dict], output_dir: str):
        """Generate summary statistics for the intervention experiment."""

        summary = {
            'total_pairs': len(results),
            'successful_pairs': 0,
            'intervention_effectiveness': {},
            'average_improvements': {},
            'best_interventions': []
        }


        all_settings = set()
        for result in results:
            for setting_key in result.get('intervention_results', {}):
                all_settings.add(setting_key)


        alphas = set()
        temperatures = set()
        for setting in all_settings:
            if setting.startswith('alpha_') and '_temp_' in setting:
                parts = setting.split('_')
                if len(parts) >= 4:
                    alpha = float(parts[1])
                    temp = float(parts[3])
                    alphas.add(alpha)
                    temperatures.add(temp)

        alphas = sorted(list(alphas))
        temperatures = sorted(list(temperatures))

        for alpha in alphas:
            for temp in temperatures:
                setting_key = f'alpha_{alpha}_temp_{temp}'
                improvements = []
                successful_count = 0

                for result in results:
                    if setting_key in result.get('intervention_results', {}):
                        intervention = result['intervention_results'][setting_key]
                        if intervention.get('parsed_successfully', False):
                            successful_count += 1
                            improvement = intervention.get('stability_improvement', 0.0)
                            improvements.append(improvement)

                            summary['best_interventions'].append({
                                'parent_a_id': result['parent_a_id'],
                                'alpha': alpha,
                                'temperature': temp,
                                'improvement': improvement,
                                'original_ehull': result['parent_a_ehull'],
                                'new_ehull': result['parent_a_ehull'] - improvement
                            })

                summary['intervention_effectiveness'][setting_key] = {
                    'successful_generations': successful_count,
                    'average_improvement': np.mean(improvements) if improvements else 0.0,
                    'std_improvement': np.std(improvements) if improvements else 0.0,
                    'max_improvement': np.max(improvements) if improvements else 0.0,
                    'positive_improvements': sum(1 for imp in improvements if imp > 0)
                }


        summary['best_interventions'].sort(key=lambda x: x['improvement'], reverse=True)
        summary['best_interventions'] = summary['best_interventions'][:10]


        with open(f"{output_dir}/experiment_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Total pairs processed: {summary['total_pairs']}")

        for alpha in alphas:
            for temp in temperatures:
                setting_key = f'alpha_{alpha}_temp_{temp}'
                if setting_key in summary['intervention_effectiveness']:
                    stats = summary['intervention_effectiveness'][setting_key]
                    print(f"\nAlpha = {alpha}, Temperature = {temp}:")
                    print(f"  Successful generations: {stats['successful_generations']}")
                    print(f"  Average improvement: {stats['average_improvement']:.4f}")
                    print(f"  Positive improvements: {stats['positive_improvements']}")
                    print(f"  Max improvement: {stats['max_improvement']:.4f}")

        if summary['best_interventions']:
            print(f"\nTop 5 improvements:")
            for i, best in enumerate(summary['best_interventions'][:5]):
                print(f"  {i+1}. Parent A {best['parent_a_id']}: {best['improvement']:.4f} (α={best['alpha']}, T={best['temperature']})")


def main():
    """Main function to run the onset intervention experiment."""

    import argparse

    parser = argparse.ArgumentParser(description='Onset Layer Intervention Experiment - Improved')
    parser.add_argument('--csv_path', type=str,
                       default='data/lcm_exp5_parents_5k.csv',
                       help='Path to crystal pairs CSV file')
    parser.add_argument('--output_dir', type=str,
                       default='results/experiment4_onset_intervention_improved',
                       help='Output directory for results')
    parser.add_argument('--max_parent_a', type=int, default=1000,
                       help='Maximum number of Parent A structures to process (ignored if --specific_parent_a_ids is used)')
    parser.add_argument('--specific_parent_a_ids', type=str, nargs='+',
                       help='Specific parent A IDs to test (e.g., --specific_parent_a_ids 991 450). If provided, no other filtering is applied.')
    parser.add_argument('--target_layer', type=int, default=39,
                       help='Target layer for intervention')
    parser.add_argument('--model_name', type=str,
                       default="meta-llama/Llama-3.1-70B-Instruct",
                       help='Model name to use')

    args = parser.parse_args()


    experiment = OnsetLayerIntervention(
        model_name=args.model_name,
        target_layer=args.target_layer
    )

    experiment.run_experiment(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        max_parent_a=args.max_parent_a,
        specific_parent_a_ids=args.specific_parent_a_ids
    )


if __name__ == "__main__":
    main()
