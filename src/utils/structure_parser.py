import re
import json
from typing import Optional, List, Tuple
from pymatgen.core import Structure
import numpy as np
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity


def structure_to_string(structure: Structure, precision: int = 8, fmt: str = 'poscar') -> str:
    """Convert Structure to formatted string with specified decimal precision"""
    fmt = fmt.lower()
    if fmt not in ['poscar', 'cif']:
        raise ValueError("Format must be either 'poscar' or 'cif'")
    if fmt == 'poscar' and precision < 12:
        species = []
        counts = []
        current_sp = None
        count = 0
        for site in structure:
            if site.species_string != current_sp:
                if current_sp is not None:
                    species.append(current_sp)
                    counts.append(count)
                current_sp = site.species_string
                count = 1
            else:
                count += 1
        species.append(current_sp)
        counts.append(count)
        fmt_str = f"{{:.{precision}f}}"
        lines = [
            " ".join(f"{sp}{cnt}" for sp, cnt in zip(species, counts)),
            "1.0",

            "\n".join("  " + " ".join(fmt_str.format(x) for x in row)
                      for row in structure.lattice.matrix),
            " ".join(species),
            " ".join(map(str, counts)),
            "direct",

            "\n".join("   " + " ".join(fmt_str.format(x) for x in site.frac_coords) +
                      f" {site.species_string}" for site in structure)
        ]
        return "\n".join(lines)



    if fmt == 'cif':
        from pymatgen.io.cif import CifWriter
        writer = CifWriter(structure, symprec=1e-3)
        return str(writer)

    return str(structure.to(fmt=fmt))


def parse_structure_string(input_str: str, format_type: str) -> Optional[Structure]:
    """
    Parse a noisy structure string into a pymatgen Structure object.
    Handles JSON format, POSCAR format, CIF format, and mixed coordinate formats.

    Args:
        input_str: String containing structure information
        format_type: Format type ('poscar' or 'cif')

    Returns:
        pymatgen Structure object or None if parsing fails
    """
    format_type = format_type.lower()
    if not input_str or not input_str.strip():
        return None
    try:

        structure = _parse_json_structure(input_str)
        if structure:
            return structure

        if format_type == 'cif':
            structure = _parse_cif_structure(input_str)
        elif format_type == 'poscar':
            structure = _parse_poscar_structure(input_str)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
        if structure:
            return structure


        structure = _parse_mixed_coordinates(input_str)
        if structure:
            return structure

    except Exception as e:
        print(f"Error parsing structure: {e}")

    return None


def _parse_json_structure(input_str: str) -> Optional[Structure]:
    """Try to parse JSON format with formula and structure fields."""
    try:

        cleaned = _clean_json_string(input_str)


        json_match = re.search(r'\{.*?"formula".*?"structure".*?\}', cleaned, re.DOTALL)
        if not json_match:
            return None

        json_str = json_match.group(0)

        data = json.loads(json_str)

        if 'formula' not in data or 'structure' not in data:
            return None


        struct_str = data['structure'].strip()

        struct_str = struct_str.replace('\\n', '\n').replace('\\r', '\r').replace('\\t', '\t')


        lines = [line.strip() for line in struct_str.split('\n') if line.strip()]


        if _is_simple_coordinate_list(lines):
            return _parse_simple_coordinate_list(lines)


        return _parse_poscar_structure(struct_str)

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return None


def _is_simple_coordinate_list(lines: List[str]) -> bool:
    """Check if lines look like a simple 'Element X Y Z' coordinate list."""
    coord_line_count = 0

    for line in lines:
        parts = line.split()
        if len(parts) >= 4:

            element = _clean_element_name(parts[0])
            if element and _is_valid_element(element):
                try:
                    [float(x) for x in parts[1:4]]
                    coord_line_count += 1
                except ValueError:
                    pass
            else:

                try:
                    [float(x) for x in parts[:3]]
                    element = _clean_element_name(parts[3])
                    if element and _is_valid_element(element):
                        coord_line_count += 1
                except ValueError:
                    pass


    return coord_line_count >= len(lines) * 0.6 and coord_line_count >= 2


def _parse_simple_coordinate_list(lines: List[str]) -> Optional[Structure]:
    """Parse simple list of 'Element X Y Z' lines."""
    try:
        coordinates = []
        site_species = []

        for line in lines:
            parts = line.split()
            if len(parts) >= 4:

                element = _clean_element_name(parts[0])
                if element and _is_valid_element(element):
                    try:
                        coords = [float(x) for x in parts[1:4]]
                        coordinates.append(coords)
                        site_species.append(element)
                        continue
                    except ValueError:
                        pass


                try:
                    coords = [float(x) for x in parts[:3]]
                    element = _clean_element_name(parts[3])
                    if element and _is_valid_element(element):
                        coordinates.append(coords)
                        site_species.append(element)
                        continue
                except ValueError:
                    pass

        if len(coordinates) < 2:
            return None


        max_coord = max(max(abs(c) for c in coord) for coord in coordinates)
        coords_are_cartesian = max_coord > 2.0


        if coords_are_cartesian:

            coords_array = np.array(coordinates)
            mins = np.min(coords_array, axis=0)
            maxs = np.max(coords_array, axis=0)
            ranges = maxs - mins
            size = max(ranges) + 5
            lattice = [[size, 0.0, 0.0], [0.0, size, 0.0], [0.0, 0.0, size]]
        else:

            lattice = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]

        return Structure(
            lattice=lattice,
            species=site_species,
            coords=coordinates,
            coords_are_cartesian=coords_are_cartesian
        )

    except Exception as e:
        return None


def _clean_json_string(text: str) -> str:
    """Clean JSON string for parsing."""

    text = re.sub(r'```.*?\n|```|\[|\]', '', text)

    text = re.sub(r'["""]', '"', text)

    text = re.sub(r'(\{)(\d+)(:)', r'\1"\2"\3', text)


    def fix_newlines(match):
        content = match.group(1)
        escaped_content = content.replace('\n', '\\n')
        return f'"{escaped_content}"'

    text = re.sub(r'"([^"]*)"', fix_newlines, text)
    return text.strip()


def _parse_poscar_structure(poscar_str: str) -> Optional[Structure]:
    """Parse POSCAR format string."""
    poscar_str = poscar_str.strip('poscar')
    try:
        lines = [line.strip() for line in poscar_str.split('\n') if line.strip()]

        if len(lines) < 8:
            return None


        scale = float(lines[1])
        lattice = [list(map(float, line.split()[:3])) for line in lines[2:5]]

        if any(len(vec) != 3 for vec in lattice):
            return None


        species = lines[5].split()
        counts = list(map(int, lines[6].split()))

        if len(species) != len(counts):
            return None


        coord_type = lines[7].lower().strip()
        if not coord_type.startswith(('direct', 'cart')):
            return None


        coordinates = []
        site_species = []


        species_expected = {}
        for sp, count in zip(species, counts):
            species_expected[sp] = count

        species_actual = {sp: 0 for sp in species}

        for line in lines[8:]:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 3:
                continue

            try:
                coords = [float(x) for x in parts[:3]]


                if len(parts) >= 4:
                    element = _clean_element_name(parts[3])
                    if element in species:
                        coordinates.append(coords)
                        site_species.append(element)
                        species_actual[element] += 1
                else:

                    assigned = False
                    for sp in species:
                        if species_actual[sp] < species_expected[sp]:
                            coordinates.append(coords)
                            site_species.append(sp)
                            species_actual[sp] += 1
                            assigned = True
                            break
                    if not assigned:
                        coordinates.append(coords)
                        site_species.append(species[0])

            except (ValueError, IndexError):
                continue

        if not coordinates:
            return None


        return Structure(
            lattice=lattice,
            species=site_species,
            coords=coordinates,
            coords_are_cartesian=coord_type.startswith('cart')
        )

    except (ValueError, IndexError):
        return None


def _parse_mixed_coordinates(input_str: str) -> Optional[Structure]:
    """Parse mixed coordinate format like 'Element X Y Z'."""
    try:
        lines = [line.strip() for line in input_str.split('\n') if line.strip()]

        coordinates = []
        site_species = []
        coord_type = 'direct'


        for line in lines:
            if line.lower().strip() in ['direct', 'cartesian']:
                coord_type = line.lower().strip()
                break


        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                element = _clean_element_name(parts[0])
                if element and _is_valid_element(element):
                    try:
                        coords = [float(x) for x in parts[1:4]]
                        coordinates.append(coords)
                        site_species.append(element)
                    except ValueError:
                        continue


        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    coords = [float(x) for x in parts[:3]]
                    element = _clean_element_name(parts[3])
                    if element and _is_valid_element(element):
                        coordinates.append(coords)
                        site_species.append(element)
                except ValueError:
                    continue

        if len(coordinates) < 2:
            return None


        if coord_type == 'direct':
            lattice = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        else:

            coords_array = np.array(coordinates)
            ranges = np.ptp(coords_array, axis=0)
            size = max(ranges) + 5
            lattice = [[size, 0.0, 0.0], [0.0, size, 0.0], [0.0, 0.0, size]]

        return Structure(
            lattice=lattice,
            species=site_species,
            coords=coordinates,
            coords_are_cartesian=(coord_type == 'cartesian')
        )

    except Exception:
        return None


def _clean_element_name(element_str: str) -> str:
    """Clean element name by removing numbers and extra characters."""

    cleaned = re.sub(r'[^a-zA-Z]', '', element_str)
    if not cleaned:
        return ""


    return cleaned[0].upper() + cleaned[1:].lower()


def _is_valid_element(element: str) -> bool:
    """Check if string is a valid element symbol."""
    valid_elements = {
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
        'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm'
    }
    return element in valid_elements


def _parse_cif_structure(cif_str: str) -> Optional[Structure]:
    """Parse CIF format string."""
    try:

        if not ("data_" in cif_str or "_cell_length_a" in cif_str):
            return None


        cleaned_cif = _clean_cif_string(cif_str)


        try:
            return Structure.from_str(cleaned_cif, fmt="cif")
        except Exception:

            return _parse_cif_cell_parameters(cleaned_cif)

    except Exception:
        return None


def _clean_cif_string(cif_str: str) -> str:
    """Clean CIF string for parsing."""

    cif_str = re.sub(r'```.*?\n|```', '', cif_str)


    cif_str = cif_str.replace('\r\n', '\n').replace('\r', '\n')


    cif_str = cif_str.strip()


    if not cif_str.startswith('data_'):
        cif_str = 'data_structure\n' + cif_str

    return cif_str


def _parse_cif_cell_parameters(cif_str: str) -> Optional[Structure]:
    """Parse CIF cell parameters and create structure."""
    try:

        a = _extract_cif_parameter(cif_str, '_cell_length_a')
        b = _extract_cif_parameter(cif_str, '_cell_length_b')
        c = _extract_cif_parameter(cif_str, '_cell_length_c')
        alpha = _extract_cif_parameter(cif_str, '_cell_angle_alpha')
        beta = _extract_cif_parameter(cif_str, '_cell_angle_beta')
        gamma = _extract_cif_parameter(cif_str, '_cell_angle_gamma')

        if not all(x is not None for x in [a, b, c, alpha, beta, gamma]):
            return None


        positions = []
        species = []


        for line in cif_str.split('\n'):
            if '_atom_site_' in line:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        element = _clean_element_name(parts[0])
                        if element and _is_valid_element(element):

                            coords = []
                            for x in parts[1:4]:
                                try:
                                    val = float(x)
                                    if np.isnan(val) or np.isinf(val):
                                        print(f"Warning: Invalid coordinate value {x} for atom {element}")
                                        val = 0.0
                                    coords.append(val)
                                except ValueError:
                                    print(f"Warning: Could not convert coordinate {x} to float for atom {element}")
                                    coords.append(0.0)


                            if len(coords) == 3:
                                positions.append(coords)
                                species.append(element)
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Error processing atom site: {str(e)}")
                        continue

        if not positions:
            print("Warning: No valid atomic positions found in CIF")
            return None


        from math import cos, sin, radians


        alpha_rad = radians(alpha)
        beta_rad = radians(beta)
        gamma_rad = radians(gamma)


        lattice = np.array([
            [a, 0, 0],
            [b * cos(gamma_rad), b * sin(gamma_rad), 0],
            [c * cos(beta_rad),
             c * (cos(alpha_rad) - cos(beta_rad) * cos(gamma_rad)) / sin(gamma_rad),
             c * np.sqrt(1 - cos(alpha_rad) ** 2 - cos(beta_rad) ** 2 - cos(gamma_rad) ** 2 +
                         2 * cos(alpha_rad) * cos(beta_rad) * cos(gamma_rad)) / sin(gamma_rad)]
        ])


        try:
            structure = Structure(
                lattice=lattice,
                species=species,
                coords=positions,
                coords_are_cartesian=True
            )


            for i, site in enumerate(structure):
                if any(np.isnan(x) for x in site.coords):
                    print(f"Warning: NaN coordinates found in final structure for atom {i}")
                    return None
                if any(np.isinf(x) for x in site.coords):
                    print(f"Warning: Infinite coordinates found in final structure for atom {i}")
                    return None

            return structure

        except Exception as e:
            print(f"Error creating structure: {str(e)}")
            return None

    except Exception as e:
        print(f"Error parsing CIF cell parameters: {str(e)}")
        return None


def _extract_cif_parameter(cif_str: str, param_name: str) -> Optional[float]:
    """Extract a parameter value from CIF string."""
    pattern = f"{param_name}\\s+([-+]?\\d*\\.?\\d+)"
    match = re.search(pattern, cif_str)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _validate_lattice(structure: Structure) -> bool:
    """Validate lattice parameters to prevent singular matrices and invalid angles.

    Args:
        structure: Structure to validate

    Returns:
        True if lattice is valid, False otherwise
    """
    try:

        lengths = structure.lattice.abc
        if any(l <= 0 for l in lengths):
            print("Warning: Invalid lattice lengths (must be positive)")
            return False


        angles = structure.lattice.angles
        if any(a <= 0 or a >= 180 for a in angles):
            print("Warning: Invalid lattice angles (must be between 0 and 180 degrees)")
            return False


        try:
            inv_matrix = structure.lattice.inv_matrix
            if np.isnan(inv_matrix).any() or np.isinf(inv_matrix).any():
                print("Warning: Singular lattice matrix (contains NaN or Inf)")
                print("Inverse matrix:")
                print(inv_matrix)
                return False
        except np.linalg.LinAlgError:
            print("Warning: Singular lattice matrix (cannot compute inverse)")
            return False

        return True
    except Exception as e:
        print(f"Warning: Lattice validation failed: {e}")
        return False
