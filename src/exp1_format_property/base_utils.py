#!/usr/bin/env python3
"""
Base utilities for crystal format understanding experiments.
Integrates with existing src/utils for structure parsing, CHGNet, and e_hull calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys
import json

src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from utils.structure_parser import structure_to_string, parse_structure_string
from utils.chgnet_evaluator import CHGNetEvaluator
from utils.e_hull_calculator import EHullCalculator
from pymatgen.core import Structure, Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def analyze_poscar_string(poscar_string, mbid=None):
    """
    Analyze POSCAR string and extract properties using existing utilities.
    This replaces the missing experiment1 module.
    """
    structure = parse_structure_string(poscar_string, 'poscar')
    if structure is None:
        return None



    comp = structure.composition
    atom_counts = dict(comp.as_dict())

    analyzer = SpacegroupAnalyzer(structure)
    lattice_system = analyzer.get_lattice_type()
    space_group = analyzer.get_space_group_symbol()


    props = {
        'chemical_formula': comp.formula,
        'num_atoms': len(structure.sites),
        'tier1_num_elements': len(comp.elements),
        'atom_counts': atom_counts,


        'a': structure.lattice.a,
        'b': structure.lattice.b,
        'c': structure.lattice.c,
        'alpha': structure.lattice.alpha,
        'beta': structure.lattice.beta,
        'gamma': structure.lattice.gamma,


        'tier2_cell_volume': structure.lattice.volume,
        'lattice_system': lattice_system,
        'density_g_cm3': structure.density,


        'tier3_space_group': space_group,
    }

    return props


class CrystalDataLoader:
    """Load and prepare crystal data for experiments, using existing utilities."""

    def __init__(self, csv_path: str, max_samples: Optional[int] = None):
        self.csv_path = csv_path
        self.max_samples = max_samples
        self.df = None

    @staticmethod
    def truncate_coordinates(text: str, decimal_places: int = 4) -> str:
        """Truncate floating point numbers in structure text to reduce token count.

        Args:
            text: POSCAR or CIF format text
            decimal_places: Number of decimal places to keep (default: 4)

        Returns:
            Text with truncated coordinates
        """
        import re



        float_pattern = r'(-?\d+\.\d+)'

        def truncate_float(match):
            """Truncate a float to specified decimal places."""
            try:
                value = float(match.group(1))

                formatted = f"{value:.{decimal_places}f}"

                formatted = formatted.rstrip('0').rstrip('.')
                return formatted
            except:
                return match.group(1)


        truncated = re.sub(float_pattern, truncate_float, text)
        return truncated

    def load_data(self) -> pd.DataFrame:
        """Load crystal data from CSV."""
        import time

        print(f"  Reading CSV file: {self.csv_path}")
        start_time = time.time()
        self.df = pd.read_csv(self.csv_path)
        print(f"  ✓ Raw CSV loaded: {len(self.df)} rows in {time.time() - start_time:.1f}s")


        required_cols = ['mbid', 'poscar_string']
        original_size = len(self.df)
        self.df = self.df.dropna(subset=required_cols)
        print(f"  ✓ Filtered valid structures: {len(self.df)}/{original_size} rows")

        if self.max_samples:
            self.df = self.df.sample(n=min(self.max_samples, len(self.df)),
                                   random_state=42)
            print(f"  ✓ Sampled to max_samples: {len(self.df)} rows")

        print(f"  ✓ Final dataset: {len(self.df)} crystal structures ready")
        return self.df

    def get_paired_formats(self, sample_size: int = 100) -> List[Dict[str, Any]]:
        """Get paired CIF/POSCAR data using existing structure utilities."""
        import time

        if self.df is None:
            self.load_data()


        actual_sample_size = min(sample_size, len(self.df))
        print(f"  Sampling {actual_sample_size} structures from {len(self.df)} available")
        sample_df = self.df.sample(n=actual_sample_size, random_state=42)

        paired_data = []
        start_time = time.time()
        total_rows = len(sample_df)

        for i, (_, row) in enumerate(sample_df.iterrows()):
            if i % 50 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (total_rows - i) / rate if rate > 0 else 0
                print(f"  Progress: {i}/{total_rows} structures ({i/total_rows*100:.1f}%) - {rate:.1f} struct/s - ETA: {eta:.0f}s")
            try:

                poscar_text = row['poscar_string']
                structure = None
                try:
                    structure = Structure.from_str(poscar_text, fmt='poscar')
                except Exception:
                    structure = parse_structure_string(poscar_text, 'poscar')

                if structure is None:
                    continue


                cif_text = structure_to_string(structure, fmt='cif')


                poscar_text = self.truncate_coordinates(poscar_text, decimal_places=4)
                cif_text = self.truncate_coordinates(cif_text, decimal_places=4)


                ground_truth = self._extract_ground_truth_properties(row, structure)

                pair = {
                    'mbid': row['mbid'],
                    'poscar_text': poscar_text,
                    'cif_text': cif_text,
                    'structure': structure,
                    'band_gap': row.get('band_gap', None),
                    'composition_formula': row.get('composition_formula', ''),
                    'lattice_a': row.get('lattice_a', None),
                    'lattice_b': row.get('lattice_b', None),
                    'lattice_c': row.get('lattice_c', None),
                    'lattice_alpha': row.get('lattice_alpha', None),
                    'lattice_beta': row.get('lattice_beta', None),
                    'lattice_gamma': row.get('lattice_gamma', None),
                    'crystal_system': row.get('crystal_system', ''),
                    'ground_truth': ground_truth
                }
                paired_data.append(pair)

            except Exception as e:
                print(f"  ✗ Failed to convert {row['mbid']}: {e}")
                continue

        elapsed = time.time() - start_time
        success_rate = len(paired_data) / total_rows * 100
        print(f"  ✓ Conversion completed in {elapsed:.1f}s")
        print(f"  ✓ Success rate: {success_rate:.1f}% ({len(paired_data)}/{total_rows} structures)")
        print(f"  ✓ Final dataset: {len(paired_data)} paired format samples ready")
        return paired_data

    def _extract_ground_truth_properties(self, row, structure) -> Dict[str, Any]:
        """Extract all available ground truth properties from dataset row."""
        import re
        import numpy as np

        ground_truth = {}

        try:

            formula = row.get('composition_formula', '')
            ground_truth['chemical_formula'] = formula


            if formula:
                elements = set(re.findall(r'([A-Z][a-z]?)', formula))
                ground_truth['num_elements'] = len(elements)


                ground_truth['has_oxygen'] = 'O' in elements
                ground_truth['has_sodium'] = 'Na' in elements
                ground_truth['has_carbon'] = 'C' in elements
            else:
                ground_truth['num_elements'] = None
                ground_truth['has_oxygen'] = None
                ground_truth['has_sodium'] = None
                ground_truth['has_carbon'] = None


            if structure:
                ground_truth['num_atoms'] = len(structure.sites)
            else:
                ground_truth['num_atoms'] = None


            ground_truth['lattice_a'] = row.get('lattice_a', None)


            if structure:
                ground_truth['cell_volume'] = structure.lattice.volume
            else:

                try:
                    a, b, c = row.get('lattice_a'), row.get('lattice_b'), row.get('lattice_c')
                    alpha, beta, gamma = row.get('lattice_alpha'), row.get('lattice_beta'), row.get('lattice_gamma')
                    if all(x is not None for x in [a, b, c, alpha, beta, gamma]):
                        ground_truth['cell_volume'] = self._calculate_cell_volume(a, b, c, alpha, beta, gamma)
                    else:
                        ground_truth['cell_volume'] = None
                except:
                    ground_truth['cell_volume'] = None


            if structure and formula:
                try:
                    ground_truth['density'] = structure.density
                except:
                    ground_truth['density'] = None
            else:
                ground_truth['density'] = None


            ground_truth['lattice_system'] = row.get('crystal_system', None)


            ground_truth['space_group'] = row.get('space_group', None)


            try:
                alpha = row.get('lattice_alpha', 90)
                beta = row.get('lattice_beta', 90)
                gamma = row.get('lattice_gamma', 90)
                tolerance = 0.1
                ground_truth['all_angles_90'] = (abs(alpha - 90) < tolerance and
                                               abs(beta - 90) < tolerance and
                                               abs(gamma - 90) < tolerance)
            except:
                ground_truth['all_angles_90'] = None

        except Exception as e:
            print(f"  ⚠ Error extracting ground truth for {row.get('mbid', 'unknown')}: {e}")

            for prop in ['chemical_formula', 'num_atoms', 'num_elements', 'has_oxygen', 'has_sodium', 'has_carbon',
                        'lattice_a', 'cell_volume', 'density', 'lattice_system', 'space_group', 'all_angles_90']:
                if prop not in ground_truth:
                    ground_truth[prop] = None

        return ground_truth

    def _calculate_cell_volume(self, a: float, b: float, c: float,
                              alpha: float, beta: float, gamma: float) -> float:
        """Calculate unit cell volume from lattice parameters."""
        import numpy as np
        try:

            alpha_rad = np.radians(alpha)
            beta_rad = np.radians(beta)
            gamma_rad = np.radians(gamma)


            cos_alpha = np.cos(alpha_rad)
            cos_beta = np.cos(beta_rad)
            cos_gamma = np.cos(gamma_rad)

            volume = a * b * c * np.sqrt(
                1 + 2 * cos_alpha * cos_beta * cos_gamma -
                cos_alpha**2 - cos_beta**2 - cos_gamma**2
            )
            return volume
        except Exception:
            return None


class PropertyCalculator:
    """Calculate ground truth properties using existing utilities and enhanced experiment1."""

    def __init__(self, use_chgnet: bool = False, ppd_path: Optional[str] = None):
        self.use_chgnet = use_chgnet
        self.chgnet_evaluator = None
        self.ehull_calculator = None

        if use_chgnet:
            try:
                self.chgnet_evaluator = CHGNetEvaluator()
                if ppd_path and Path(ppd_path).exists():
                    self.ehull_calculator = EHullCalculator(ppd_path)
                print("✅ CHGNet evaluator initialized")
            except Exception as e:
                print(f"⚠️  CHGNet initialization failed: {e}")
                self.use_chgnet = False

    @staticmethod
    def calculate_ground_truth(paired_data: List[Dict[str, Any]],
                             include_advanced: bool = False) -> List[Dict[str, Any]]:
        """Calculate ground truth properties using enhanced experiment1 analyzer."""

        success_count = 0
        fail_count = 0

        for item in paired_data:
            try:

                if 'poscar_text' not in item or not item['poscar_text']:
                    raise ValueError(f"Missing or empty poscar_text for {item.get('mbid', 'unknown')}")


                props = analyze_poscar_string(item['poscar_text'], item.get('mbid'))

                if props is None:
                    raise ValueError(f"analyze_poscar_string returned None for {item.get('mbid', 'unknown')}")


                item['gt_chemical_formula'] = props.get('chemical_formula', '')
                item['gt_num_elements'] = props.get('tier1_num_elements', 0)

                alpha, beta, gamma = props.get('alpha'), props.get('beta'), props.get('gamma')
                if all(x is not None for x in [alpha, beta, gamma]):
                    angles_90 = all(abs(angle - 90.0) < 1.0 for angle in [alpha, beta, gamma])
                    item['gt_all_angles_90'] = angles_90
                else:
                    item['gt_all_angles_90'] = None

                atom_counts = props.get('atom_counts', {})
                item['gt_has_oxygen'] = 'O' in atom_counts
                item['gt_has_sodium'] = 'Na' in atom_counts
                item['gt_has_carbon'] = 'C' in atom_counts


                item['gt_num_atoms'] = props.get('num_atoms', 0)
                item['gt_cell_volume'] = props.get('tier2_cell_volume', None)


                item['gt_lattice_a'] = props.get('a', None)
                item['gt_lattice_system'] = props.get('lattice_system', '')
                item['gt_density'] = props.get('density_g_cm3', None)
                item['gt_space_group'] = props.get('tier3_space_group', '')


                if include_advanced and 'structure' in item:
                    structure = item['structure']


                    try:
                        from pymatgen.analysis.local_env import VoronoiNN
                        voronoi = VoronoiNN()
                        coord_nums = []
                        for i, site in enumerate(structure):
                            try:
                                cn = voronoi.get_cn(structure, i)
                                coord_nums.append(cn)
                            except:
                                pass
                        item['gt_avg_coordination'] = np.mean(coord_nums) if coord_nums else None
                    except Exception as e:
                        item['gt_avg_coordination'] = None

                success_count += 1

            except Exception as e:
                fail_count += 1
                print(f"  ⚠ Failed to calculate ground truth for {item.get('mbid', 'unknown')}: {e}")

                for key in ['gt_chemical_formula', 'gt_num_atoms', 'gt_num_elements',
                           'gt_cell_volume', 'gt_lattice_a', 'gt_all_angles_90',
                           'gt_lattice_system', 'gt_density', 'gt_space_group',
                           'gt_has_oxygen', 'gt_has_sodium', 'gt_has_carbon',
                           'gt_avg_coordination']:
                    if key not in item:
                        item[key] = None

        print(f"  Ground truth calculation: {success_count} succeeded, {fail_count} failed")
        return paired_data

    def calculate_chgnet_properties(self, paired_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate CHGNet-based properties using existing CHGNet evaluator."""

        if not self.use_chgnet or self.chgnet_evaluator is None:
            print("CHGNet not available, skipping CHGNet properties")
            return paired_data

        print("Calculating CHGNet-based properties...")

        for item in paired_data:
            try:
                structure = item.get('structure')
                if structure is None:
                    continue


                chgnet_result = self.chgnet_evaluator.evaluate_structure(structure)

                if chgnet_result:
                    item['gt_chgnet_energy'] = chgnet_result.get('energy', None)
                    item['gt_chgnet_forces'] = chgnet_result.get('forces', None)
                    item['gt_chgnet_stress'] = chgnet_result.get('stress', None)


                    if self.ehull_calculator and 'energy' in chgnet_result:
                        try:
                            se_list = [{'structure': structure, 'energy': chgnet_result['energy']}]
                            ehull_results = self.ehull_calculator(se_list)
                            if ehull_results:
                                item['gt_e_hull'] = ehull_results[0].get('e_hull', None)
                        except Exception as e:
                            print(f"E_hull calculation failed for {item['mbid']}: {e}")
                            item['gt_e_hull'] = None

            except Exception as e:
                print(f"CHGNet evaluation failed for {item['mbid']}: {e}")
                item['gt_chgnet_energy'] = None
                item['gt_chgnet_forces'] = None
                item['gt_chgnet_stress'] = None
                item['gt_e_hull'] = None

        return paired_data


class PromptGenerator:
    """Generate prompts for different test types."""

    @staticmethod
    def format_recognition_prompts() -> List[str]:
        """Prompts for format recognition test."""
        return [
            "What file format is this crystal structure data in?",
            "Is this structure data in CIF or POSCAR format?",
            "What type of crystallographic file format is shown?",
            "Does this use fractional coordinates? Answer yes or no.",
            "Does this file contain explicit parameter labels like '_cell_length_a'? Answer yes or no.",
        ]

    @staticmethod
    def property_extraction_prompts() -> Dict[str, List[str]]:
        """Prompts for property extraction by tier."""
        return {

            'has_oxygen': [
                "Does this structure contain oxygen atoms? Answer yes or no.",
                "Is oxygen present in this crystal? Answer yes or no.",
                "Are there any O atoms in this structure? Answer yes or no."
            ],
            'chemical_formula': [
                "What is the chemical formula of this crystal structure?",
                "What is the composition of this material?",
                "What chemical formula does this structure represent?"
            ],
            'num_elements': [
                "How many different types of elements are in this structure?",
                "How many unique elements does this crystal contain?",
                "Count the number of distinct element types."
            ],
            'all_angles_90': [
                "Are all lattice angles equal to 90 degrees? Answer yes or no.",
                "Do all unit cell angles equal 90°? Answer yes or no.",
                "Are alpha, beta, and gamma all 90 degrees? Answer yes or no."
            ],


            'num_atoms': [
                "How many atoms are in the unit cell?",
                "What is the total number of atoms in this unit cell?",
                "Count the number of atoms in the unit cell."
            ],
            'cell_volume': [
                "What is the unit cell volume in cubic Angstroms?",
                "Calculate the volume of the unit cell.",
                "What is the cell volume?"
            ],


            'lattice_a': [
                "What is the lattice parameter 'a' in Angstroms?",
                "What is the value of the 'a' lattice parameter?",
                "What is the length of the 'a' unit cell edge?"
            ],
            'space_group': [
                "What is the space group of this crystal structure?",
                "Determine the space group symbol.",
                "What space group does this structure have?"
            ],
            'lattice_system': [
                "What crystal system does this structure belong to?",
                "Classify this crystal into one of the 7 crystal systems.",
                "What is the lattice system (cubic, tetragonal, etc.)?"
            ],
            'density': [
                "What is the density of this crystal in g/cm³?",
                "Calculate the density of this material.",
                "What is the crystal density?"
            ],
        }


def save_results(results: Dict[str, Any], output_path: str):
    """Save experiment results to JSON file."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {output_path}")


def load_results(input_path: str) -> Dict[str, Any]:
    """Load experiment results from JSON file."""
    with open(input_path, 'r') as f:
        return json.load(f)
