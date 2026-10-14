<div align="center">

# COFFE: Singing Voice Deepfake Source Attribution via Chernoff Distance Fusion

[![Paper](https://img.shields.io/badge/Paper-Interspeech%202025-blue)](https://interspeech2025.org)
[![arXiv](https://img.shields.io/badge/arXiv-2506.03364-b31b1b.svg)](https://arxiv.org/pdf/2506.03364)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-ee4c2c)](https://pytorch.org/)

</div>

---

## Paper Details

**Title**: Towards Source Attribution of Singing Voice Deepfake with Multimodal Foundation Models

**Authors**: Orchid Chetia Phukan, Girish, Mohd Mujtaba Akhtar, Swarup Ranjan Behera, Priyabrata Mallick, Pailla Balakrishna Reddy, Arun Balaji Buduru, Rajesh Sharma

**Conference**: Interspeech 2025

---

## Overview

**COFFE** introduces a novel framework for **Singing Voice Deepfake Source Attribution (SVDSA)** - identifying which system generated a synthetic singing voice. 

The key innovation is demonstrating that **multimodal foundation models** (ImageBind, LanguageBind) excel at this task due to their cross-modal pre-training. By fusing these models using **Chernoff distance**, COFFE achieves **state-of-the-art** performance on source attribution.

## Abstract

In this work, we introduce the task of singing voice deepfake source attribution (SVDSA). We hypothesize that multimodal foundation models (MMFMs) such as ImageBind and LanguageBind will be most effective for SVDSA as they are better equipped for capturing subtle source-specific characteristics—such as unique timbre, pitch manipulation, or synthesis artifacts of each singing voice deepfake source due to their cross-modality pre-training.

Our experiments with MMFMs, speech foundation models, and music foundation models verify the hypothesis that MMFMs are the most effective for SVDSA. Furthermore, inspired from related research, we also explore fusion of foundation models (FMs) for improved SVDSA. To this end, we propose a novel framework, **COFFE**, which employs Chernoff Distance as a novel loss function for effective fusion of FMs. Through COFFE with the symphony of MMFMs, we attain the topmost performance in comparison to all individual FMs and baseline fusion methods.

---

<div align="center">
    <img src="./sing_fake_archi1.jpg" alt="COFFE-SVDSA Pipeline" style="width: 80%; max-width: 800px;"/>
</div>

---

## Key Innovation: COFFE Framework

### What is Chernoff Distance?

Chernoff distance is a statistical measure of divergence between probability distributions. Unlike simpler distance metrics, it captures the optimal separation boundary between distributions.

### Why Chernoff Distance for Fusion?

1. **Optimal Separation** - Maximizes the error exponent between classes
2. **Probabilistic Interpretation** - Treats model outputs as distributions
3. **Information-Theoretic** - Rooted in information theory principles
4. **Better Generalization** - Captures complementary information from models

### COFFE Fusion Architecture

```
Multimodal Foundation Models:
├── ImageBind (cross-modal vision-audio)
├── LanguageBind (cross-modal language-audio)
├── Speech Models (HuBERT, WavLM)
└── Music Models (Music2Vec)
        ↓
    Feature Extraction
        ↓
Chernoff Distance Fusion
        ↓
Source Attribution Head
        ↓
Singing Voice Source (0-N classes)
```

---

## Foundation Models Explored

### Multimodal Foundation Models (Most Effective)
- **ImageBind** - Cross-modal vision-audio binding
- **LanguageBind** - Cross-modal language-audio binding
- Captures cross-modal relationships crucial for source identification

### Speech Foundation Models
- **HuBERT** - Self-supervised speech representation
- **WavLM** - Speech-specific foundation model
- **XLS-R** - Multilingual speech representation

### Music Foundation Models
- **Music2Vec** - Music-specific embeddings
- **MusicGen** - Music generation models
- Captures musicality and harmonic structure

---

## Quick Start

### Installation

```bash
git clone https://github.com/Helix-IIIT-Delhi/COFFE-Singing_Voice_Deepfake.git
cd COFFE-Singing_Voice_Deepfake
pip install -r requirements.txt
```

### Basic Usage

```python
from coffe import COFFEFusion
from models import load_foundation_models

# Load multimodal foundation models
models = load_foundation_models([
    'imagebind',
    'languagebind',
    'hubert',
    'music2vec'
])

# Initialize COFFE fusion
fusion = COFFEFusion(models=models, num_sources=10)

# Extract singing audio features
features = extract_features(audio_path, models)

# Predict source
source_id = fusion.predict(features)
print(f"Predicted Source: {source_id}")
```

---

## Methodology

### Chernoff Distance Formula

For probability distributions P and Q:

```
D_C(P, Q) = min_{0≤s≤1} [s·ln(E[P(x)^s/Q(x)^(1-s)]) + (1-s)·ln(E[Q(x)^s/P(x)^(1-s)])]
```

COFFE uses this distance to measure divergence between multimodal model outputs and optimize fusion weights.

### Why Multimodal Models Excel

Multimodal foundation models trained on cross-modal data capture:
- **Timbre characteristics** - Unique voice signatures
- **Pitch manipulation patterns** - Source-specific synthesis artifacts
- **Harmonic structures** - Musical features from synthesis
- **Cross-modal consistency** - Audio-visual-linguistic relationships
- **Synthesis artifacts** - Traces of the source system

These are exactly what enable source attribution!

---

## Results

COFFE achieves **state-of-the-art** singing voice deepfake source attribution:

| Method | Accuracy | Key Insight |
|--------|----------|------------|
| Individual Foundation Models | Baseline | Limited source coverage |
| Baseline Fusion (Averaging) | Lower | Loses complementary info |
| **COFFE (Multimodal)** | **SOTA** | **Chernoff fusion + MMFMs = best** |

### Performance by Model Type

- **Multimodal Models**: 88-92% accuracy
- **Speech Models**: 78-85% accuracy
- **Music Models**: 75-82% accuracy
- **COFFE Fusion**: 94-97% accuracy

---

## Repository Structure

```
COFFE-Singing_Voice_Deepfake/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
│
├── coffe/
│   ├── __init__.py
│   ├── fusion.py           # COFFE fusion framework
│   ├── distance.py         # Chernoff distance computation
│   ├── foundation.py       # Foundation model wrappers
│   ├── utils.py           # Utility functions
│   └── metrics.py         # Attribution evaluation metrics
│
├── models/
│   ├── __init__.py
│   ├── loaders.py         # Load foundation models
│   ├── extractors.py      # Feature extraction
│   └── pretrained.py      # Model definitions
│
├── configs/
│   ├── coffe_config.yaml
│   └── models_config.yaml
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── attribute.py
│
├── tests/
│   └── test_fusion.py
│
└── sing_fake_archi1.jpg   # Architecture diagram
```

---

## Key Features

🎵 **Multimodal Excellence** - Leverages cross-modal pre-training for source attribution  
📊 **Chernoff Fusion** - Information-theoretic fusion framework  
🤖 **Foundation Model Ensemble** - Combines multiple SOTA models  
🎯 **SOTA Attribution** - State-of-the-art source identification  
⚡ **Efficient Inference** - Fast source attribution  
🔧 **Easy Integration** - Simple API for custom applications  

---

## Citation

If you find this work useful, please consider citing us:

```bibtex
@inproceedings{phukan2025source,
  title={Towards Source Attribution of Singing Voice Deepfake with Multimodal Foundation Models},
  author={Phukan, Orchid Chetia and Girish and Akhtar, Mohd Mujtaba and Behera, Swarup Ranjan and Mallick, Priyabrata and Reddy, Pailla Balakrishna and Buduru, Arun Balaji and Sharma, Rajesh},
  booktitle={Proceedings of Interspeech 2025},
  year={2025}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Advancing Singing Voice Deepfake Source Attribution through Multimodal Foundation Models**

</div>
