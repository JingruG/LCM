#!/usr/bin/env python3
"""
Experiment 1: Analysis of format understanding and property extraction results.
"""

import json
import sys
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent.parent))
from exp1_format_property.base_utils import load_results, save_results


class UnifiedResultsAnalyzer:
    """Analyze unified_vllm_results.json directly."""

    def __init__(self, unified_results: Dict[str, Any]):
        self.unified_results = unified_results
        self.model_name = unified_results.get('model', 'unknown')
        self.results = unified_results.get('results', [])

    def extract_format_answer(self, response: str, prompt_id: str) -> str:
        """Extract format recognition answer."""
        response = response.lower().strip()

        if prompt_id in ["format_type", "cif_or_poscar"]:
            if "cif" in response:
                return "CIF"
            elif "poscar" in response or "vasp" in response:
                return "POSCAR"
            else:
                return "Unknown"
        return response

    def extract_property_answer(self, response: str, property_name: str) -> Union[str, float, bool, None]:
        """Extract property answer with robust parsing."""
        response = response.strip()
        response_lower = response.lower()

        if property_name == 'chemical_formula':
            return self._extract_chemical_formula(response)
        elif property_name in ['num_atoms', 'num_elements']:
            return self._extract_integer_count(response, property_name)
        elif property_name in ['has_oxygen', 'has_sodium', 'has_carbon', 'all_angles_90']:
            return self._extract_boolean(response, property_name)
        elif property_name in ['lattice_a', 'cell_volume', 'density']:
            return self._extract_numerical_value(response, property_name)
        elif property_name == 'lattice_system':
            return self._extract_lattice_system(response)
        elif property_name == 'space_group':
            return self._extract_space_group(response)

        return response

    def _extract_chemical_formula(self, response: str) -> str:
        """Extract chemical formula, including formulas with spaces like 'Mg14 Nb1 Ga1'."""


        spaced_patterns = [
            r'^([A-Z][a-z]?\d+(?:\s+[A-Z][a-z]?\d+)+)',
            r'(?:chemical\s+formula[:\s]+is\s+)?([A-Z][a-z]?\d+(?:\s+[A-Z][a-z]?\d+)+)',
            r'([A-Z][a-z]?\d+(?:\s+[A-Z][a-z]?\d+){2,})',
        ]

        for pattern in spaced_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            if matches:
                formula = matches[0].strip()

                if len(re.findall(r'[A-Z][a-z]?\d+', formula)) >= 2:
                    return formula


        no_space_patterns = [
            r'(?:chemical\s+formula[:\s]+)?([A-Z][a-z]?\d+[A-Z][a-z]?\d+[A-Z][a-z]?\d+)',
            r'["\']([A-Z][a-z]?\d+[A-Z][a-z]?\d+[A-Z][a-z]?\d+)["\']',
            r'\b([A-Z][a-z]?\d+[A-Z][a-z]?\d+[A-Z][a-z]?\d+)\b',
        ]

        for pattern in no_space_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                formula = matches[0].strip()
                if re.match(r'^[A-Z][a-z]?', formula):
                    return formula


        patterns = [
            r'\b([A-Z][a-z]?\d+)+\b',
            r'\b[A-Z][a-z]?(?:\d+[A-Z][a-z]?\d*)*\b',
            r'["\']([A-Z][a-z]?\d*)+["\']',
            r'formula[:\s]+([A-Z][a-z]?\d*)+',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                formula = matches[0] if isinstance(matches[0], str) else ''.join(matches[0])
                if re.match(r'^[A-Z][a-z]?', formula):
                    return self._normalize_formula(formula)

        words = response.split()
        for word in words:
            if re.match(r'^[A-Z][a-z]?\d*', word):
                return self._normalize_formula(word)

        return None

    def _extract_integer_count(self, response: str, property_name: str) -> int:
        """Extract integer counts, prioritizing totals and sum expressions."""
        if property_name == 'num_atoms':


            sum_patterns = [
                r'(\d+(?:\s*\+\s*\d+)+)\s*=\s*(\d+)',
                r'total[^=]*=\s*(\d+)',
                r'equals?\s+(\d+)\s*[\.\n]',
            ]

            for pattern in sum_patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                if matches:

                    if isinstance(matches[0], tuple) and len(matches[0]) == 2:
                        try:
                            result = int(matches[0][1])
                            if 1 <= result <= 1000:
                                return result
                        except (ValueError, IndexError, TypeError):
                            pass
                    else:
                        try:
                            num_str = matches[0] if isinstance(matches[0], str) else matches[0][0]
                            num = int(num_str)
                            if 1 <= num <= 1000:
                                return num
                        except (ValueError, TypeError, IndexError):
                            continue


            total_patterns = [
                r'total\s+number\s+of\s+atoms?\s+(?:in\s+the\s+unit\s+cell\s+)?(?:is\s+)?(\d+)',
                r'total\s+atoms?\s+(?:in\s+the\s+unit\s+cell\s+)?(?:is\s+)?(\d+)',
                r'(\d+)\s+total\s+atoms?',
                r'total[:\s]+(\d+)\s+atoms?',
            ]

            for pattern in total_patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                if matches:
                    try:
                        num = int(matches[0])
                        if 1 <= num <= 1000:
                            return num
                    except (ValueError, TypeError):
                        continue


            patterns = [
                r'(\d+)\s+(?:total\s+)?atoms?',
                r'(?:contains|has|total)\s+(\d+)\s+atoms?',
                r'atoms?[:\s]+(\d+)',
            ]

            for pattern in patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                if matches:
                    try:
                        num = int(matches[0])
                        if 1 <= num <= 1000:
                            return num
                    except (ValueError, TypeError):
                        continue


            numbers = re.findall(r'\b(\d+)\b', response)
            if numbers:
                valid_numbers = []
                for num_str in numbers:
                    num = int(num_str)
                    if 1 <= num <= 1000:
                        valid_numbers.append(num)

                if valid_numbers:



                    if re.search(r'\d+\s*\+\s*\d+', response):

                        return max(valid_numbers)
                    else:
                        return valid_numbers[0]
        else:
            patterns = [
                r'(\d+)\s+(?:different\s+)?elements?',
                r'(?:contains|has|total)\s+(\d+)\s+elements?',
                r'elements?[:\s]+(\d+)',
                r'total\s+number\s+of\s+elements?\s+(?:is\s+)?(\d+)',
            ]

            for pattern in patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                if matches:
                    try:
                        num = int(matches[0])
                        if 1 <= num <= 50:
                            return num
                    except (ValueError, TypeError):
                        continue

            numbers = re.findall(r'\b(\d+)\b', response)
            if numbers:
                valid_numbers = []
                for num_str in numbers:
                    num = int(num_str)
                    if 1 <= num <= 50:
                        valid_numbers.append(num)

                if valid_numbers:
                    return valid_numbers[0]

        return None

    def _extract_boolean(self, response: str, property_name: str) -> bool:
        """Extract boolean values."""
        response_lower = response.lower().strip()



        explicit_yes_patterns = [
            r'^\s*yes\b',
            r'^\s*yes[.,]',
            r'^\s*yes\s+[.,]',
            r'\byes\s*[.,]\s+',
        ]
        explicit_no_patterns = [
            r'^\s*no\b',
            r'^\s*no[.,]',
            r'^\s*no\s+[.,]',
            r'\bno\s*[.,]\s+',
        ]


        for pattern in explicit_no_patterns:
            if re.search(pattern, response_lower):
                return False

        for pattern in explicit_yes_patterns:
            if re.search(pattern, response_lower):
                return True

        if property_name.startswith('has_'):
            element = property_name.split('_')[1]
            positive_patterns = [
                f'yes.*{element}', f'{element}.*yes', f'contains.*{element}',
                f'{element}.*present', f'true.*{element}', f'{element}.*true'
            ]
            negative_patterns = [
                f'no.*{element}', f'{element}.*no', f'without.*{element}',
                f'{element}.*absent', f'false.*{element}', f'{element}.*false'
            ]
        else:


            negative_patterns = [
                r'not\s+all\s+(?:equal\s+to\s+)?90',
                r'not\s+all\s+90',
                r'angles?\s+are\s+not\s+(?:all\s+)?(?:equal\s+to\s+)?90',
                r'not\s+(?:all\s+)?90\s+degrees?',
                r'no\s*[.,]\s+',
                r'\bno\b.*90',
                r'90.*no',
                r'oblique',
                r'acute',
                r'obtuse',
                r'angle\s+\w+\s+is\s+(?:[1-8]\d|[1-8]\d\d|2[0-4]\d|25[0-9]|26[0-8])\s+degrees?',
                r'angle\s+\w+\s+is\s+[1-9](?!0)\s+degrees?',
            ]

            positive_patterns = [
                r'all\s+angles?\s+(?:are\s+)?(?:equal\s+to\s+)?90',
                r'all\s+90\s+degrees?',
                r'angles?\s+are\s+all\s+90',
                r'right\s+angles?',
                r'orthogonal',
                r'cubic',
            ]


        for pattern in negative_patterns:
            if re.search(pattern, response_lower):
                return False


        for pattern in positive_patterns:
            if re.search(pattern, response_lower):
                return True


        if re.search(r'\byes\b|\btrue\b', response_lower):
            return True
        elif re.search(r'\bno\b|\bfalse\b', response_lower):
            return False

        return None

    def _extract_numerical_value(self, response: str, property_name: str) -> float:
        """Extract numerical values."""
        if property_name == 'lattice_a':
            patterns = [r'(\d+\.?\d*)\s*[AÅ]', r'a\s*=\s*(\d+\.?\d*)', r'(\d+\.?\d*)\s*angstrom']
            reasonable_range = (1.0, 50.0)
        elif property_name == 'cell_volume':
            patterns = [r'(\d+\.?\d*)\s*[AÅ]³?', r'volume[:\s]*(\d+\.?\d*)', r'(\d+\.?\d*)\s*cubic']
            reasonable_range = (10.0, 10000.0)
        elif property_name == 'density':
            patterns = [r'(\d+\.?\d*)\s*g/cm³?', r'density[:\s]*(\d+\.?\d*)', r'(\d+\.?\d*)\s*g/cc']
            reasonable_range = (0.1, 25.0)
        else:
            patterns = [r'(\d+\.?\d*)']
            reasonable_range = (0.0, float('inf'))

        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                try:
                    value = float(matches[0])
                    if reasonable_range[0] <= value <= reasonable_range[1]:
                        return value
                except ValueError:
                    continue

        numbers = re.findall(r'\b(\d+\.?\d*)\b', response)
        for num_str in numbers:
            try:
                value = float(num_str)
                if reasonable_range[0] <= value <= reasonable_range[1]:
                    return value
            except ValueError:
                continue

        return None

    def _extract_lattice_system(self, response: str) -> str:
        """Extract lattice system."""
        response_lower = response.lower()

        systems = {
            'cubic': ['cubic', 'cube'],
            'tetragonal': ['tetragonal'],
            'orthorhombic': ['orthorhombic', 'orthogonal'],
            'hexagonal': ['hexagonal', 'hex'],
            'trigonal': ['trigonal', 'rhombohedral'],
            'monoclinic': ['monoclinic'],
            'triclinic': ['triclinic']
        }

        for system, variants in systems.items():
            for variant in variants:
                if variant in response_lower:
                    return system.title()

        return None

    def _extract_space_group(self, response: str) -> str:
        """Extract space group, handling variations like 'P 1', 'P-1', 'P1'."""


        symbol_patterns = [
            r'space\s+group[^.]*is\s+([A-Z]\s*[-]?\s*\d+(?:\s*[/]\s*\w+)?)',
            r'space\s+group[:\s]+([A-Z]\s*[-]?\s*\d+(?:\s*[/]\s*\w+)?)',
            r'([A-Z]\s*[-]?\s*\d+)\s+space\s+group',
            r'\b([A-Z]\s*[-]?\s*\d+(?:\s*[/]\s*\w+)?)\b',
            r'["\']([A-Z]\s*[-]?\s*\d+(?:\s*[/]\s*\w+)?)["\']',
        ]

        for pattern in symbol_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                sg = matches[0].strip()

                sg_normalized = re.sub(r'\s+|-', '', sg)

                if re.match(r'^[A-Z]\d+', sg_normalized, re.IGNORECASE):
                    return sg


        number_patterns = [
            r'space\s+group\s+(?:number\s+)?(?:#|is\s+)?(\d+)',
            r'#\s*(\d+)',
        ]

        for pattern in number_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                num_str = matches[0].strip()
                try:
                    num = int(num_str)
                    if 1 <= num <= 230:
                        return str(num)
                except ValueError:
                    continue



        numbers = re.findall(r'\b(\d+)\b', response)
        for num_str in numbers:
            num = int(num_str)
            if 1 <= num <= 230:


                num_pos = response.find(num_str)

                context = response[max(0, num_pos-50):num_pos+50].lower()
                if 'space' in context and 'group' in context:
                    return str(num)

        return None

    def _normalize_formula(self, formula: str) -> str:
        """Normalize chemical formula."""
        formula = re.sub(r'\s+', '', formula)
        result = ""
        i = 0
        while i < len(formula):
            if formula[i].isupper():
                result += formula[i]
                i += 1
                if i < len(formula) and formula[i].islower():
                    result += formula[i]
                    i += 1
                while i < len(formula) and formula[i].isdigit():
                    result += formula[i]
                    i += 1
            else:
                i += 1
        return result if result else formula

    def evaluate_format_answer(self, answer: str, ground_truth_format: str, prompt_id: str) -> bool:
        """Evaluate format recognition answer."""
        if prompt_id in ["format_type", "cif_or_poscar"]:
            return answer.upper() == ground_truth_format.upper()
        return False

    def evaluate_property_answer(self, extracted: Any, ground_truth: Any, property_name: str, raw_response: str = None) -> bool:
        """Evaluate property answer."""
        if extracted is None or ground_truth is None:
            return False


        if property_name == 'chemical_formula':
            if isinstance(ground_truth, list):

                extracted_normalized = self._normalize_formula(str(extracted)) if extracted else None
                if extracted_normalized:
                    if any(extracted_normalized == self._normalize_formula(str(gt_item))
                          for gt_item in ground_truth if gt_item is not None):
                        return True



                if raw_response:
                    raw_response_normalized = self._normalize_formula(raw_response)
                    for gt_item in ground_truth:
                        if gt_item is None:
                            continue
                        gt_normalized = self._normalize_formula(str(gt_item))

                        if gt_normalized in raw_response_normalized:
                            return True

                        if str(gt_item) in raw_response:
                            return True

                return False
            else:
                extracted_normalized = self._normalize_formula(str(extracted)) if extracted else None
                if extracted_normalized:
                    return extracted_normalized == self._normalize_formula(str(ground_truth))

                if raw_response:
                    gt_normalized = self._normalize_formula(str(ground_truth))
                    raw_response_normalized = self._normalize_formula(raw_response)
                    if gt_normalized in raw_response_normalized or str(ground_truth) in raw_response:
                        return True
                return False
        elif property_name == 'space_group':
            if isinstance(ground_truth, list):

                if extracted:
                    if any(self._compare_space_group(extracted, gt_item)
                          for gt_item in ground_truth if gt_item is not None):
                        return True



                if raw_response:
                    for gt_item in ground_truth:
                        if gt_item is None:
                            continue
                        gt_str = str(gt_item)

                        if self._compare_space_group_in_text(raw_response, gt_str):
                            return True

                return False
            else:
                if extracted:
                    if self._compare_space_group(extracted, ground_truth):
                        return True


                if raw_response:
                    if self._compare_space_group_in_text(raw_response, str(ground_truth)):
                        return True

                return False

        if property_name in ['has_oxygen', 'has_sodium', 'has_carbon', 'all_angles_90']:
            return bool(extracted) == bool(ground_truth)
        elif property_name in ['num_atoms', 'num_elements']:
            try:
                return int(extracted) == int(ground_truth)
            except (ValueError, TypeError):
                return False
        elif property_name == 'lattice_a':
            return self._compare_numerical(extracted, ground_truth, tolerance=0.01)
        elif property_name == 'cell_volume':
            return self._compare_numerical(extracted, ground_truth, tolerance=0.05)
        elif property_name == 'density':
            return self._compare_numerical(extracted, ground_truth, tolerance=0.10)
        elif property_name == 'lattice_system':
            return str(extracted).lower().strip() == str(ground_truth).lower().strip()

        return False

    def _compare_numerical(self, extracted: Any, ground_truth: Any, tolerance: float) -> bool:
        """Compare numerical values with tolerance."""
        try:
            ext_val = float(extracted)
            gt_val = float(ground_truth)
            if gt_val == 0:
                return abs(ext_val) < tolerance
            return abs(ext_val - gt_val) / abs(gt_val) < tolerance
        except (ValueError, TypeError, ZeroDivisionError):
            return False

    def _normalize_space_group(self, sg: str) -> str:
        """Normalize space group string by removing spaces, hyphens, and converting to uppercase."""
        if not sg:
            return ""

        normalized = re.sub(r'[\s\-]', '', str(sg)).upper()
        return normalized

    def _compare_space_group_in_text(self, text: str, space_group: str) -> bool:
        """Check if a space group appears in the text, handling variations."""
        if not text or not space_group:
            return False

        gt_normalized = self._normalize_space_group(space_group)
        text_normalized = self._normalize_space_group(text)


        if gt_normalized in text_normalized:
            return True


        if str(space_group) in text:
            return True


        if space_group.isdigit():
            num = int(space_group)
            if 1 <= num <= 230:

                pattern = rf'\b{num}\b'
                matches = list(re.finditer(pattern, text))
                for match in matches:

                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end].lower()
                    if 'space' in context and 'group' in context:
                        return True



        gt_letter_match = re.search(r'([A-Z])', gt_normalized, re.IGNORECASE)
        gt_num_match = re.search(r'(\d+)', gt_normalized)

        if gt_letter_match and gt_num_match:
            letter = gt_letter_match.group(1).upper()
            num = gt_num_match.group(1)

            patterns = [
                rf'{letter}\s*[-]?\s*{num}',
                rf'{letter}\s+{num}',
                rf'{letter}-{num}',
            ]
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return True

        return False

    def _compare_space_group(self, extracted: Any, ground_truth: Any) -> bool:
        """Compare space groups, handling variations like 'P 1', 'P-1', 'P1'."""
        try:
            ext_sg = str(extracted).strip() if extracted else ""
            gt_sg = str(ground_truth).strip() if ground_truth else ""

            if not ext_sg or not gt_sg:
                return False


            if ext_sg.lower() == gt_sg.lower():
                return True


            ext_normalized = self._normalize_space_group(ext_sg)
            gt_normalized = self._normalize_space_group(gt_sg)

            if ext_normalized == gt_normalized:
                return True



            ext_num_match = re.search(r'(\d+)', ext_sg)
            gt_num_match = re.search(r'(\d+)', gt_sg)

            if ext_num_match and gt_num_match:
                try:
                    ext_num = int(ext_num_match.group(1))
                    gt_num = int(gt_num_match.group(1))

                    if 1 <= ext_num <= 230 and 1 <= gt_num <= 230:
                        return ext_num == gt_num
                except (ValueError, AttributeError):
                    pass



            ext_letter = re.search(r'([A-Z])', ext_normalized, re.IGNORECASE)
            gt_letter = re.search(r'([A-Z])', gt_normalized, re.IGNORECASE)

            if ext_letter and gt_letter:
                ext_letter_val = ext_letter.group(1).upper()
                gt_letter_val = gt_letter.group(1).upper()
                if ext_letter_val == gt_letter_val:

                    ext_num_match = re.search(r'(\d+)', ext_normalized)
                    gt_num_match = re.search(r'(\d+)', gt_normalized)
                    if ext_num_match and gt_num_match:
                        return ext_num_match.group(1) == gt_num_match.group(1)

            return False
        except Exception:
            return False

    def analyze_format_recognition(self) -> Dict[str, Any]:
        """Analyze format recognition results."""
        format_results = [r for r in self.results if r.get('test_type') == 'format_recognition']

        if len(format_results) == 0:
            return {}

        processed = []
        for result in format_results:
            extracted = self.extract_format_answer(result['raw_response'], result.get('prompt_id', ''))
            correct = self.evaluate_format_answer(
                extracted,
                result.get('ground_truth_format', ''),
                result.get('prompt_id', '')
            )

            processed.append({
                **result,
                'extracted_answer': extracted,
                'correct': correct
            })

        df = pd.DataFrame(processed)

        summary = {
            'overall_accuracy': df['correct'].mean(),
            'by_format': {},
            'by_prompt': {},
            'total_samples': len(processed)
        }

        for fmt in ['cif', 'poscar']:
            fmt_results = df[df['format'] == fmt]
            if len(fmt_results) > 0:
                summary['by_format'][fmt] = {
                    'accuracy': fmt_results['correct'].mean(),
                    'count': len(fmt_results)
                }

        for prompt_id in df['prompt_id'].unique():
            prompt_results = df[df['prompt_id'] == prompt_id]
            summary['by_prompt'][prompt_id] = {
                'accuracy': prompt_results['correct'].mean(),
                'count': len(prompt_results)
            }

        return {
            'summary': summary,
            'processed_results': processed
        }

    def analyze_property_extraction(self) -> Dict[str, Any]:
        """Analyze property extraction results."""
        property_results = [r for r in self.results if r.get('test_type') == 'property_extraction']

        if len(property_results) == 0:
            return {}

        processed = []
        for result in property_results:
            extracted = self.extract_property_answer(
                result['raw_response'],
                result.get('property', '')
            )

            ground_truth = result.get('ground_truth_property')
            correct = self.evaluate_property_answer(
                extracted,
                ground_truth,
                result.get('property', ''),
                raw_response=result.get('raw_response', '')
            )

            processed.append({
                **result,
                'extracted_answer': extracted,
                'correct': correct
            })

        df = pd.DataFrame(processed)

        summary = {
            'overall_accuracy': df['correct'].mean(),
            'by_tier': {},
            'by_format': {},
            'by_property': {},
            'total_samples': len(processed),
            'valid_ground_truth_count': df['ground_truth_property'].notna().sum()
        }

        for tier in df['tier'].unique():
            tier_results = df[df['tier'] == tier]
            summary['by_tier'][int(tier)] = {
                'accuracy': tier_results['correct'].mean(),
                'count': len(tier_results)
            }

        for fmt in ['cif', 'poscar']:
            fmt_results = df[df['format'] == fmt]
            if len(fmt_results) > 0:
                summary['by_format'][fmt] = {
                    'accuracy': fmt_results['correct'].mean(),
                    'count': len(fmt_results)
                }

        for prop in df['property'].unique():
            prop_results = df[df['property'] == prop]
            summary['by_property'][prop] = {
                'accuracy': prop_results['correct'].mean(),
                'count': len(prop_results),
                'tier': prop_results['tier'].iloc[0] if len(prop_results) > 0 else None
            }

        return {
            'summary': summary,
            'processed_results': processed
        }

    def analyze_consistency(self) -> Dict[str, Any]:
        """Analyze cross-format consistency."""
        property_results = [r for r in self.results if r.get('test_type') == 'property_extraction']

        if len(property_results) == 0:
            return {}


        processed_results = []
        for result in property_results:
            extracted = self.extract_property_answer(
                result['raw_response'],
                result.get('property', '')
            )

            ground_truth = result.get('ground_truth_property')
            correct = self.evaluate_property_answer(
                extracted,
                ground_truth,
                result.get('property', ''),
                raw_response=result.get('raw_response', '')
            )
            processed_results.append({
                **result,
                'extracted_answer': extracted,
                'correct': correct
            })

        df = pd.DataFrame(processed_results)


        paired_comparisons = []
        for mbid in df['mbid'].unique():
            structure_data = df[df['mbid'] == mbid]

            for prop in structure_data['property'].unique():
                prop_data = structure_data[structure_data['property'] == prop]

                cif_responses = prop_data[prop_data['format'] == 'cif']
                poscar_responses = prop_data[prop_data['format'] == 'poscar']

                if len(cif_responses) > 0 and len(poscar_responses) > 0:
                    cif_resp = cif_responses.iloc[0]
                    poscar_resp = poscar_responses.iloc[0]

                    cif_answer = cif_resp.get('extracted_answer', cif_resp['raw_response'])
                    poscar_answer = poscar_resp.get('extracted_answer', poscar_resp['raw_response'])

                    agreement = self._check_agreement(cif_answer, poscar_answer, prop)

                    paired_comparisons.append({
                        'mbid': mbid,
                        'property': prop,
                        'tier': cif_resp['tier'],
                        'cif_answer': cif_answer,
                        'poscar_answer': poscar_answer,
                        'cif_correct': cif_resp.get('correct'),
                        'poscar_correct': poscar_resp.get('correct'),
                        'agreement': agreement
                    })

        comparison_df = pd.DataFrame(paired_comparisons)

        if len(comparison_df) == 0:
            return {}

        summary = {
            'overall_agreement_rate': comparison_df['agreement'].mean(),
            'total_comparisons': len(comparison_df),
            'by_tier': {}
        }

        for tier in comparison_df['tier'].unique():
            tier_data = comparison_df[comparison_df['tier'] == tier]
            summary['by_tier'][int(tier)] = {
                'agreement_rate': tier_data['agreement'].mean(),
                'count': len(tier_data)
            }

        return summary

    def _check_agreement(self, answer1: Any, answer2: Any, property_name: str) -> bool:
        """Check if two answers agree."""
        if answer1 is None or answer2 is None:
            return answer1 == answer2

        str1 = str(answer1).lower().strip()
        str2 = str(answer2).lower().strip()

        if property_name in ['has_oxygen', 'has_sodium', 'has_carbon', 'all_angles_90']:
            bool1 = 'yes' in str1 or 'true' in str1
            bool2 = 'yes' in str2 or 'true' in str2
            return bool1 == bool2
        elif property_name in ['num_atoms', 'num_elements']:
            nums1 = re.findall(r'\d+', str1)
            nums2 = re.findall(r'\d+', str2)
            if nums1 and nums2:
                return int(nums1[0]) == int(nums2[0])
            return str1 == str2
        elif property_name in ['lattice_a', 'cell_volume', 'density']:
            nums1 = re.findall(r'\d+\.?\d*', str1)
            nums2 = re.findall(r'\d+\.?\d*', str2)
            if nums1 and nums2:
                try:
                    val1, val2 = float(nums1[0]), float(nums2[0])
                    return abs(val1 - val2) / max(abs(val1), abs(val2), 1e-6) < 0.05
                except:
                    return str1 == str2
            return str1 == str2
        else:
            return str1 == str2

    def generate_full_report(self) -> str:
        """Generate comprehensive analysis report."""
        report = []
        report.append("="*80)
        report.append(f"COMPREHENSIVE ANALYSIS: {self.model_name}")
        report.append("="*80)


        report.append("\n" + "="*80)
        report.append("TEST 1: FORMAT RECOGNITION")
        report.append("="*80)
        test1 = self.analyze_format_recognition()
        if test1:
            summary = test1['summary']
            report.append(f"\nOverall Accuracy: {summary['overall_accuracy']:.4f} ({summary['overall_accuracy']*100:.2f}%)")
            report.append(f"Total Samples: {summary['total_samples']}")

            report.append("\nBy Format:")
            for fmt, stats in summary['by_format'].items():
                report.append(f"  {fmt.upper()}: {stats['accuracy']:.4f} ({stats['count']} samples)")

            report.append("\nBy Prompt Type:")
            for prompt_id, stats in summary['by_prompt'].items():
                report.append(f"  {prompt_id}: {stats['accuracy']:.4f} ({stats['count']} samples)")


        report.append("\n" + "="*80)
        report.append("TEST 2: PROPERTY EXTRACTION")
        report.append("="*80)
        test2 = self.analyze_property_extraction()
        if test2:
            summary = test2['summary']
            report.append(f"\nOverall Accuracy: {summary['overall_accuracy']:.4f} ({summary['overall_accuracy']*100:.2f}%)")
            report.append(f"Total Samples: {summary['total_samples']}")
            report.append(f"Valid Ground Truth: {summary['valid_ground_truth_count']}/{summary['total_samples']}")

            if summary['valid_ground_truth_count'] == 0:
                report.append("  ⚠ WARNING: No valid ground truth values found!")

            report.append("\nBy Tier:")
            for tier in sorted(summary['by_tier'].keys()):
                stats = summary['by_tier'][tier]
                report.append(f"  Tier {tier}: {stats['accuracy']:.4f} ({stats['count']} samples)")

            report.append("\nBy Format:")
            for fmt, stats in summary['by_format'].items():
                report.append(f"  {fmt.upper()}: {stats['accuracy']:.4f} ({stats['count']} samples)")

            report.append("\nBy Property:")
            for prop, stats in sorted(summary['by_property'].items(), key=lambda x: x[1].get('tier', 0)):
                report.append(f"  {prop} (Tier {stats['tier']}): {stats['accuracy']:.4f} ({stats['count']} samples)")


        report.append("\n" + "="*80)
        report.append("TEST 3: CROSS-FORMAT CONSISTENCY")
        report.append("="*80)
        test3 = self.analyze_consistency()
        if test3:
            report.append(f"\nOverall Agreement Rate: {test3['overall_agreement_rate']:.4f} ({test3['overall_agreement_rate']*100:.2f}%)")
            report.append(f"Total Comparisons: {test3['total_comparisons']}")

            report.append("\nBy Tier:")
            for tier in sorted(test3['by_tier'].keys()):
                stats = test3['by_tier'][tier]
                report.append(f"  Tier {tier}: {stats['agreement_rate']:.4f} ({stats['count']} comparisons)")

        return "\n".join(report)


def compare_models(model1_results: Dict[str, Any], model2_results: Dict[str, Any],
                   model1_name: str, model2_name: str) -> str:
    """Compare results between two models."""
    analyzer1 = UnifiedResultsAnalyzer(model1_results)
    analyzer2 = UnifiedResultsAnalyzer(model2_results)

    report = []
    report.append("="*80)
    report.append("MODEL COMPARISON")
    report.append(f"{model1_name} vs {model2_name}")
    report.append("="*80)


    test1_1 = analyzer1.analyze_format_recognition()
    test1_2 = analyzer2.analyze_format_recognition()

    if test1_1 and test1_2:
        report.append("\n" + "="*80)
        report.append("TEST 1: FORMAT RECOGNITION COMPARISON")
        report.append("="*80)

        s1 = test1_1['summary']
        s2 = test1_2['summary']

        report.append(f"\nOverall Accuracy:")
        report.append(f"  {model1_name}: {s1['overall_accuracy']:.4f}")
        report.append(f"  {model2_name}: {s2['overall_accuracy']:.4f}")
        report.append(f"  Difference: {abs(s1['overall_accuracy'] - s2['overall_accuracy']):.4f}")

        report.append("\nBy Format:")
        for fmt in ['cif', 'poscar']:
            if fmt in s1['by_format'] and fmt in s2['by_format']:
                acc1 = s1['by_format'][fmt]['accuracy']
                acc2 = s2['by_format'][fmt]['accuracy']
                report.append(f"  {fmt.upper()}:")
                report.append(f"    {model1_name}: {acc1:.4f}")
                report.append(f"    {model2_name}: {acc2:.4f}")
                report.append(f"    Difference: {abs(acc1 - acc2):.4f}")


    test2_1 = analyzer1.analyze_property_extraction()
    test2_2 = analyzer2.analyze_property_extraction()

    if test2_1 and test2_2:
        report.append("\n" + "="*80)
        report.append("TEST 2: PROPERTY EXTRACTION COMPARISON")
        report.append("="*80)

        s1 = test2_1['summary']
        s2 = test2_2['summary']

        report.append(f"\nOverall Accuracy:")
        report.append(f"  {model1_name}: {s1['overall_accuracy']:.4f}")
        report.append(f"  {model2_name}: {s2['overall_accuracy']:.4f}")
        report.append(f"  Difference: {abs(s1['overall_accuracy'] - s2['overall_accuracy']):.4f}")

        report.append(f"\nValid Ground Truth:")
        report.append(f"  {model1_name}: {s1['valid_ground_truth_count']}/{s1['total_samples']}")
        report.append(f"  {model2_name}: {s2['valid_ground_truth_count']}/{s2['total_samples']}")

        report.append("\nBy Tier:")
        for tier in sorted(set(list(s1['by_tier'].keys()) + list(s2['by_tier'].keys()))):
            if tier in s1['by_tier'] and tier in s2['by_tier']:
                acc1 = s1['by_tier'][tier]['accuracy']
                acc2 = s2['by_tier'][tier]['accuracy']
                report.append(f"  Tier {tier}:")
                report.append(f"    {model1_name}: {acc1:.4f}")
                report.append(f"    {model2_name}: {acc2:.4f}")
                report.append(f"    Difference: {abs(acc1 - acc2):.4f}")


    test3_1 = analyzer1.analyze_consistency()
    test3_2 = analyzer2.analyze_consistency()

    if test3_1 and test3_2:
        report.append("\n" + "="*80)
        report.append("TEST 3: CONSISTENCY COMPARISON")
        report.append("="*80)

        report.append(f"\nOverall Agreement Rate:")
        report.append(f"  {model1_name}: {test3_1['overall_agreement_rate']:.4f}")
        report.append(f"  {model2_name}: {test3_2['overall_agreement_rate']:.4f}")
        report.append(f"  Difference: {abs(test3_1['overall_agreement_rate'] - test3_2['overall_agreement_rate']):.4f}")

    return "\n".join(report)


def analyze_single_model(model_name: str, base_path: Path) -> None:
    """Analyze a single model and save combined results."""
    model_path = base_path / model_name / "unified_vllm_results.json"
    if not model_path.exists():
        print(f"Error: {model_path} not found")
        return

    print(f"\n{'='*80}")
    print(f"Analyzing {model_name}...")
    print(f"{'='*80}")
    print(f"Loading {model_name}...")
    model_results = load_results(str(model_path))
    print(f"  Loaded {len(model_results.get('results', []))} results")

    analyzer = UnifiedResultsAnalyzer(model_results)
    report = analyzer.generate_full_report()
    print(report)


    test1 = analyzer.analyze_format_recognition()
    test2 = analyzer.analyze_property_extraction()
    test3 = analyzer.analyze_consistency()









    combined_results = {
        'model': model_name,
        'source_file': str(model_path),
        'timestamp': time.time(),
        'test1_format_recognition': test1,
        'test2_property_extraction': test2,
        'test3_consistency': test3,
        'report': report
    }


    model_dir = base_path / model_name
    output_path = model_dir / "unified_analysis_results.json"
    save_results(combined_results, str(output_path))
    print(f"\n✓ Saved combined analysis to {output_path}")


def main():
    """Main analysis function."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze unified VLLM results")
    parser.add_argument("--model", type=str, default=None, help="Model to analyze (e.g., llama3.1-70b)")
    parser.add_argument("--all-models", action="store_true", help="Analyze all 4 models")
    args = parser.parse_args()

    base_path = Path(__file__).parent.parent.parent.parent / "results" / "experiment1"

    models = ['llama3.1-70b', 'llama3.1-8b', 'qwen2.5-7b', 'qwen2.5-14b']

    if args.all_models:
        print("="*80)
        print("Running analysis for all models")
        print("="*80)
        for model in models:
            analyze_single_model(model, base_path)
        print(f"\n{'='*80}")
        print("All analyses completed")
        print(f"{'='*80}")
    elif args.model:
        analyze_single_model(args.model, base_path)
    else:
        parser.print_help()
        print("\nError: Must specify either --model or --all-models")


if __name__ == "__main__":
    main()

