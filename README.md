# Ensemble Manifold Regularized Multi-Modal Graph Convolutional Network for Cognitive Ability Prediction

This repository contains the implementation of the ensemble manifold regularized multi-modal graph convolutional network (MGCN) for predicting cognitive ability using multi-modal fMRI data. The project introduces an interpretable model that leverages both functional connectivity (FC) and fMRI time series data to predict individual cognitive and behavioral traits.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Citation](#citation)
- [License](#license)

## Project Overview

The goal of this project is to develop a multi-modal graph convolutional network (MGCN) model that integrates fMRI time series data and functional connectivity (FC) between brain regions to predict cognitive abilities. The model applies a manifold-based regularization term to exploit the relationships between subjects across different modalities and introduces interpretable techniques like gradient-weighted regression activation mapping (Grad-RAM) and edge mask learning to highlight cognition-related brain regions and edges.

Due to data privacy and protection policies, the datasets used in this project cannot be publicly shared without explicit permission. If you are interested in accessing the dataset for research purposes, please submit a formal request.



## Installation

Since this repository contains sensitive data preprocessing code, it has been hidden for privacy reasons. However, the model and evaluation code can still be run with your own pre-processed data. Check the "environment.yml" for the enviroment configuration

## Citation
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


## License
This project is licensed under the MIT License. See the LICENSE file for details.