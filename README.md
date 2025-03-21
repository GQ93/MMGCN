# Ensemble Manifold Regularized Multi-Modal Graph Convolutional Network for Cognitive Ability Prediction

This repository contains the implementation of the ensemble manifold regularized multi-modal graph convolutional network (MGCN) for predicting cognitive ability using multi-modal fMRI data. The project introduces an interpretable model that leverages both functional connectivity (FC) and fMRI time series data to predict individual cognitive and behavioral traits.

---
## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)
---

## Project Overview

The goal of this project is to develop a multi-modal graph convolutional network (MGCN) model that integrates fMRI time series data and functional connectivity (FC) between brain regions to predict cognitive abilities. The model applies a manifold-based regularization term to exploit the relationships between subjects across different modalities and introduces interpretable techniques like gradient-weighted regression activation mapping (Grad-RAM) and edge mask learning to highlight cognition-related brain regions and edges.

Due to data privacy and protection policies, the datasets used in this project cannot be publicly shared without explicit permission. If you are interested in accessing the dataset for research purposes, please submit a formal request.
### Project Structure
```commandline
MMGCN
├── config/                          # Model configuration files
├── datasets/                        # Custom dataset loader
├── [models]/                        # Model implementations (e.g, MLP, MTL, MVGCN)
├── utlis/                           # Helper functions
├── environment.yaml                 # List of Python dependencies required for the project
└── PNCdata_preprocess_v.ipynb       # Data preprocessing
```
---

## Installation
To set up the environment for this project, follow these steps:
1. **Clone the repository:**
```bash
git clone https://github.com/GQ93/MMGCN.git
cd MMGCN
```
2. **Create and activate a conda environment:**
```bash
conda env create -f environment.yaml
conda activate MMGCN
```

The environment.yaml file contains all the necessary dependencies for this project.

---
## Usage
Usage
The repository contains separate directories for each model variant. To run a specific model, first edit the configuration of the hyper-parameters in config folder for each model and then navigate to its directory and execute the training script. For example:
```bash
cd UniGCNBrainNetwork
python main_UniGCNBrainNetwork.py
```

## Citation
```commandline
@ARTICLE{9428628,
  author={Qu, Gang and Xiao, Li and Hu, Wenxing and Wang, Junqi and Zhang, Kun and Calhoun, Vince D. and Wang, Yu-Ping},
  journal={IEEE Transactions on Biomedical Engineering}, 
  title={Ensemble Manifold Regularized Multi-Modal Graph Convolutional Network for Cognitive Ability Prediction}, 
  year={2021},
  volume={68},
  number={12},
  pages={3564-3573},
  keywords={Functional magnetic resonance imaging;Biological system modeling;Predictive models;Biomarkers;Deep learning;Brain modeling;Time series analysis;fMRI;functional connectivity;graph convolutional networks;interpretable deep learning;multi-modal deep learning},
  doi={10.1109/TBME.2021.3077875}}
```



## License
This project is licensed under the MIT License. See the LICENSE file for details.