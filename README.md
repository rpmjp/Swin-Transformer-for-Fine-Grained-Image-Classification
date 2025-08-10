# Swin Transformer for Fine-Grained Image Classification

**A Comprehensive Evaluation on Oxford-IIIT Pet Dataset**

This repository contains the complete implementation and evaluation of Swin Transformer architecture for fine-grained image classification, conducted as part of CS 634 - Deep Learning (Fall 2025) at NJIT.

## Project Overview

This research investigates whether the architectural advantages of Swin Transformers, originally demonstrated on large-scale datasets (ImageNet-1K, COCO, ADE20K), translate to resource-constrained, small-dataset scenarios typical of real-world deployments. We systematically compare Swin Transformer variants against established baselines using the Oxford-IIIT Pet Dataset.

### Key Research Questions
- Do Swin Transformer's innovations work on small datasets?
- How do different vision architectures perform under resource constraints?
- What are the practical deployment trade-offs between accuracy and efficiency?

## Key Findings

- **Swin Transformers achieve 93.8%-96.35% accuracy** (11-16% improvement over CNNs)
- **Vision Transformers fail catastrophically** (7.17% vs. 84% reported in original paper)
- **EfficientNet exhibits inverse scaling** (smaller models outperform larger ones)
- **Architecture design matters more than parameter count** for small datasets

## Repository Structure

```
├── notebooks/
│   ├── Swin-T_classifier.ipynb          # Swin Tiny implementation
│   ├── Swin-S_classifier.ipynb          # Swin Small implementation  
│   ├── Swin-B_224_classifier.ipynb      # Swin Base (224x224)
│   ├── Swin-B_384_classifier.ipynb      # Swin Base (384x384)
│   ├── Resnet-models.ipynb              # RegNet implementations
│   ├── EfficientNet-models.ipynb        # EfficientNet B3-B7 evaluation
│   └── Vit-model.ipynb                  # Vision Transformer evaluation
├── data/
│   ├── images/                          # Oxford-IIIT Pet images
│   └── annotations/                     # Dataset labels and metadata
├── results/
│   ├── training_curves/                 # Loss and accuracy plots
│   ├── performance_tables/              # Numerical results
│   └── comparison_charts/               # Architecture comparisons
├── docs/
│   ├── Final_Report.pdf                 # Complete research report
│   ├── Presentation_Slides.pdf          # Video presentation slides
│   └── Video_Presentation_Link.txt      # Link to presentation video
└── README.md
```

##  Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA-capable GPU (24GB+ recommended)
- Required packages: `timm`, `torchvision`, `pandas`, `matplotlib`, `tqdm`

### Installation
```bash
git clone https://github.com/yourusername/swin-transformer-evaluation
cd swin-transformer-evaluation
pip install -r requirements.txt
```

### Dataset Setup
1. Download Oxford-IIIT Pet Dataset from [official source](https://www.robots.ox.ac.uk/~vgg/data/pets/)
2. Extract to `data/` directory
3. Ensure structure matches:
   ```
   data/
   ├── images/
   │   ├── Abyssinian_1.jpg
   │   ├── Abyssinian_2.jpg
   │   └── ...
   └── annotations/
   ```

### Running Experiments

#### Swin Transformer Evaluation
```bash
# Run Swin-T (most efficient)
jupyter notebook notebooks/Swin-T_classifier.ipynb

# Run Swin-B for maximum accuracy
jupyter notebook notebooks/Swin-B_224_classifier.ipynb
```

#### Baseline Comparisons
```bash
# RegNet CNNs
jupyter notebook notebooks/Resnet-models.ipynb

# EfficientNet family
jupyter notebook notebooks/EfficientNet-models.ipynb

# Vision Transformers
jupyter notebook notebooks/Vit-model.ipynb
```

##  Results Summary

### Performance Hierarchy

| Architecture | Best Model | Accuracy | Parameters | Throughput |
|--------------|------------|----------|------------|------------|
| **Swin Transformers** | Swin-B | **96.35%** | 86.8M | 61.75 img/s |
| **RegNet CNNs** | RegNetY-16G | **85.25%** | 10M | 281.2 img/s |
| **EfficientNet** | EffNet-B3 | **80.58%** | 11M | 142.7 img/s |
| **Vision Transformers** | ViT-B/16 | **7.17%** | 86M | 18.2 img/s |

### Key Insights
- **Swin-T provides best efficiency-accuracy balance** (94.93% at 112.92 img/s)
- **ViT completely fails on small datasets** (requires massive pretraining data)
- **EfficientNet scaling laws break down** (B3 outperforms B7 significantly)
- **Hardware constraints significantly impact results** (memory limitations, batch size reductions)

##  Technical Implementation Details

### Model Configurations
- **Swin Transformers**: 4×4 patches, 7×7 windows, hierarchical stages {2,2,6,2} to {2,2,18,2}
- **Training**: 5 epochs, AdamW optimizer, CrossEntropyLoss
- **Data**: 80/20 train/val split, ImageNet normalization
- **Hardware**: NVIDIA RTX 4090 (24GB), batch sizes 16-64

### Architecture Highlights
- **Shifted Window Attention**: Linear O(n) complexity vs. quadratic O(n²) in ViT
- **Hierarchical Design**: Multi-scale features like CNNs
- **Relative Position Bias**: Better transferability than absolute positioning

##  Hardware Limitations

**Important**: This research was conducted on consumer hardware (RTX 4090, 24GB), which introduced several constraints:
- Reduced batch sizes may have affected training stability
- ViT-L/16 could not be evaluated due to memory limitations
- Training times were extended, potentially impacting convergence
- Results may improve with enterprise-grade hardware (48GB+ multi-GPU setups)

##  Practical Deployment Guidance

### Recommendations by Use Case

**Maximum Accuracy (>95% required)**
- Use **Swin-B (224²)**: 96.35% accuracy, reasonable efficiency

**Balanced Performance (90-95% accuracy)**
- Use **Swin-T**: 94.93% accuracy, best efficiency (112.92 img/s)

**Maximum Throughput (real-time applications)**
- Use **RegNetY-8G**: 285.0 img/s, 84.10% accuracy

**Avoid for Small Datasets**
- Any ViT variant (catastrophic failure)
- EfficientNet B6/B7 (poor accuracy despite high cost)
- High-resolution training (diminishing returns)

##  Future Work

### Architectural Variations
- **Adaptive window sizing**: Dynamic attention based on content complexity
- **Hybrid architectures**: Combine Swin with other attention mechanisms
- **Small-dataset-aware NAS**: Optimize for transfer learning effectiveness

### Novel Applications
- **Edge computing**: Real-time species identification
- **Medical imaging**: Resource-constrained diagnostics  
- **Mobile deployment**: Wildlife monitoring, quality control
- **Federated learning**: Distributed training on limited devices

## 📚 Publications & References

### Course Information
- **Course**: CS 634 - Deep Learning, Fall 2025
- **Institution**: New Jersey Institute of Technology (NJIT)
- **Student**: Robert Jean Pierre (UCID: RJ447)

### Key References
- Liu, Z., et al. (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." *ICCV 2021*.
- Dosovitskiy, A., et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR 2021*.
- Tan, M., & Le, Q. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *ICML 2019*.

##  Contributing

This is an academic research project completed as part of coursework. While direct contributions are not expected, feedback and discussions about the methodology, results, or future directions are welcome.

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- **NJIT CS Department** for providing computational resources
- **Original Swin Transformer authors** for their groundbreaking work
- **timm library maintainers** for excellent model implementations
- **Oxford VGG Group** for the Pet Dataset

##  Contact

**Robert Jean Pierre**
- Course: CS 634 - Deep Learning, Fall 2025
- Institution: New Jersey Institute of Technology

---

*"This research demonstrates that architectural innovation remains crucial for advancing computer vision capabilities under practical deployment constraints."*
