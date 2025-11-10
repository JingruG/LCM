#!/usr/bin/env python3
"""
CHGNet energy evaluator compatible with the workshop scripts.
"""

import torch
import numpy as np
import signal
from typing import Dict, Optional
from pymatgen.core import Structure

try:
    from chgnet.model import CHGNet
    CHGNET_AVAILABLE = True
    print("✅ CHGNet imported successfully")
except ImportError:
    CHGNET_AVAILABLE = False
    print("⚠️  CHGNet not available. Energy calculations will fail.")

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("CHGNet calculation timed out")

class CHGNetEvaluator:
    """CHGNet-based structure evaluator matching LCM implementation."""

    def __init__(self, chgnet_timeout: int = 60, ppd_path: str = "data/2023-02-07-ppd-mp.pkl.gz"):
        self.chgnet_timeout = chgnet_timeout
        self.chgnet = None
        self.e_hull_calculator = None


        if CHGNET_AVAILABLE:
            try:
                print("🔋 Loading CHGNet model...")
                self.chgnet = CHGNet.load()


                from chgnet.model.dynamics import StructOptimizer
                self.relaxer = StructOptimizer(model=self.chgnet)

                print("✅ CHGNet and relaxer loaded successfully")
            except Exception as e:
                print(f"⚠️  Failed to load CHGNet: {e}")
                self.chgnet = None
                self.relaxer = None
        else:
            print("⚠️  CHGNet not available - energy calculations will fail")
            self.relaxer = None


        try:
            from utils.e_hull_calculator import EHullCalculator
            print(f"🔋 Loading e_hull calculator from {ppd_path}")
            self.e_hull_calculator = EHullCalculator(ppd_path)
            print("✅ E_hull calculator loaded successfully")
        except Exception as e:
            print(f"⚠️  Failed to load e_hull calculator: {e}")
            print(f"     E_hull calculations will not be available")
            self.e_hull_calculator = None

    def calculate_formation_energy(self, structure: Structure) -> float:
        """
        Calculate formation energy using CHGNet.

        Args:
            structure: Pymatgen Structure object

        Returns:
            Formation energy in eV/atom

        Raises:
            RuntimeError: If CHGNet is not available or calculation fails
        """
        if self.chgnet is None:
            raise RuntimeError("CHGNet not available - cannot calculate formation energy")

        try:

            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.chgnet_timeout)

            try:
                print(f"      🔋 Running CHGNet prediction (timeout: {self.chgnet_timeout}s)...")


                prediction = self.chgnet.predict_structure(structure)


                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)


                formation_energy = prediction['e']

                print(f"      ✅ CHGNet completed: {formation_energy:.4f} eV/atom")

                return float(formation_energy)

            except TimeoutError:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                print(f"      ⏰ CHGNet calculation timed out after {self.chgnet_timeout}s")
                raise RuntimeError("CHGNet calculation timed out")

        except Exception as e:

            signal.alarm(0)
            if 'old_handler' in locals():
                signal.signal(signal.SIGALRM, old_handler)

            print(f"      ❌ CHGNet prediction failed: {e}")
            raise RuntimeError(f"CHGNet calculation failed: {e}")

    def relax_structure(self, structure: Structure) -> Optional[Dict]:
        """Relax structure using CHGNet following MatLLMSearch pattern."""
        if self.relaxer is None:
            print(f"      ❌ CHGNet relaxer not available")
            return None

        try:
            print(f"      🔋 Relaxing structure with CHGNet...")
            relaxation_result = self.relaxer.relax(structure)
            print(f"      ✅ Structure relaxation completed")
            return relaxation_result
        except Exception as e:
            print(f"      ❌ Structure relaxation failed: {e}")
            return None

    def calculate_e_hull(self, structure: Structure, e_hull_calculator=None) -> Optional[float]:
        """
        Calculate energy above hull following MatLLMSearch pipeline:
        1. Relax structure with CHGNet
        2. Calculate energy of relaxed structure
        3. Compute e_hull with relaxed structure and energy

        Args:
            structure: Pymatgen Structure object
            e_hull_calculator: Optional e_hull calculator object (overrides instance calculator)

        Returns:
            Energy above hull in eV/atom (positive = unstable) or None if calculation fails
        """

        calculator = e_hull_calculator or self.e_hull_calculator

        if calculator is None:
            print(f"      ⚠️  No e_hull calculator provided")
            return None

        try:

            relaxation = self.relax_structure(structure)
            if not relaxation or not relaxation.get('final_structure'):
                print(f"      ❌ Structure relaxation failed")
                return None

            final_structure = relaxation['final_structure']


            trajectory = relaxation.get('trajectory', {})
            if hasattr(trajectory, 'energies'):
                energies = trajectory.energies
            else:
                energies = trajectory.get('energies', [])

            if not energies:
                print(f"      ❌ No energies found in relaxation trajectory")
                return None

            final_energy = energies[-1]
            print(f"      ✅ Relaxed structure energy: {final_energy:.4f} eV ({final_energy/len(final_structure):.4f} eV/atom)")


            hull_data = [{
                'structure': final_structure,
                'energy': final_energy
            }]

            e_hull_result = calculator.get_e_hull(hull_data)
            e_hull = e_hull_result[0]['e_hull']

            print(f"      ✅ e_hull calculated: {e_hull:.4f} eV/atom")
            return float(e_hull)

        except Exception as e:
            print(f"      ❌ e_hull calculation failed: {e}")
            return None

    def evaluate_structure(self, structure: Structure) -> Dict[str, float]:
        """
        Comprehensive structure evaluation using CHGNet.
        Args:
            structure: Pymatgen Structure object
        Returns:
            Dict with evaluation metrics
        """
        try:

            formation_energy = self.calculate_formation_energy(structure)


            e_hull = self.calculate_e_hull(structure)


            if e_hull is not None:
                return {
                    'e_hull': e_hull,
                    'formation_energy': formation_energy,
                    'valid': abs(formation_energy) < 3.0,
                    'stable': e_hull < 0.1,
                    'chgnet_success': True
                }
            else:
                return {
                    'e_hull': None,
                    'formation_energy': formation_energy,
                    'valid': abs(formation_energy) < 3.0,
                    'stable': None,
                    'chgnet_success': True,
                    'note': 'No e_hull calculator available'
                }

        except Exception as e:
            print(f"Structure evaluation failed: {e}")
            return {
                'e_hull': float('inf'),
                'formation_energy': float('inf'),
                'valid': False,
                'stable': False,
                'chgnet_success': False
            }

    def calculate_energy_similarity(self, struct1: Structure, struct2: Structure) -> Dict:
        """
        Calculate energy-based similarity between structures using CHGNet.
        Matches LCM implementation exactly.
        """
        similarities = {}

        if self.chgnet is None:
            similarities.update({
                'chgnet_energy_similarity': None,
                'chgnet_energy_difference': None,
                'energy_calculation_successful': False
            })
            return similarities

        try:

            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.chgnet_timeout)

            try:
                print(f"      🔋 Running CHGNet predictions (timeout: {self.chgnet_timeout}s)...")
                pred1 = self.chgnet.predict_structure(struct1)
                pred2 = self.chgnet.predict_structure(struct2)
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

                energy1 = pred1['e']
                energy2 = pred2['e']

                energy_diff = abs(energy1 - energy2)
                similarities['chgnet_energy_difference'] = energy_diff
                similarities['chgnet_energy_similarity'] = np.exp(-energy_diff / 0.5)
                similarities['energy_calculation_successful'] = True

                print(f"      ✅ CHGNet completed successfully")

            except TimeoutError:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                print(f"      ⏰ CHGNet calculation timed out after {self.chgnet_timeout}s")
                similarities.update({
                    'chgnet_energy_similarity': None,
                    'chgnet_energy_difference': None,
                    'energy_calculation_successful': False
                })

        except Exception as e:

            signal.alarm(0)
            if 'old_handler' in locals():
                signal.signal(signal.SIGALRM, old_handler)

            print(f"      ❌ CHGNet prediction failed: {e}")
            similarities.update({
                'chgnet_energy_similarity': None,
                'chgnet_energy_difference': None,
                'energy_calculation_successful': False
            })

        return similarities







