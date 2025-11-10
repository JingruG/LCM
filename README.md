## Latent Crystallographic Microscope: Probing the Emergent Crystallographic Knowledge in Large Language Models

### Experiments

#### 1. Format Understanding and Property Extraction

Evaluates LLM capabilities for:
- Format recognition (CIF vs POSCAR)
- Property extraction across multiple complexity tiers

#### 2. Onset Analysis

Uses activation patching to identify where crystallographic reasoning emerges:
- Coordinate patching: Spatial coordinate correction
- Stability judge: Thermodynamic stability assessment
- Valence verifier: Charge neutrality reasoning

#### 3. Onset Layer Intervention

Tests injecting stability vectors during generation to improve crystal structure stability.

### Setup

1. Install dependencies:
```bash
pip install torch transformers vllm pandas numpy pymatgen
```

2. Store crystal stuctures to be tested in the `../data/` directory

3. Run experiments from each directories

### Citation

```
@inproceedings{
    gan2025latent,
    title={Latent Crystallographic Microscope: Probing the Emergent Crystallographic Knowledge in Large Language Models},
    author={Jingru Gan and Yanqiao Zhu and Wei Wang},
    booktitle={Mechanistic Interpretability Workshop at NeurIPS 2025},
    year={2025},
    url={https://openreview.net/forum?id=28HkQ6mKxn}
}
```
