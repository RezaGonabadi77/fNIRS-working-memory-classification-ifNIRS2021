# fNIRS Working Memory Classification
Classifying fNIRS dataset based working memory using differents machine learning tools
## Introduction

Welcome to the fNIRS Working Memory Classification project! This repository contains MATLAB scripts and resources for the classification of fNIRS (functional Near-Infrared Spectroscopy) data related to working memory tasks using various machine learning techniques.

## Dataset

### Description

This project utilizes a specific fNIRS working memory dataset. The dataset consists of fNIRS recordings collected during working memory tasks.
### Preprocessing

Before classification, it is essential to preprocess the fNIRS data to ensure its quality and readiness for analysis. The preprocessing steps may include:

- Noise reduction
- Motion artifact removal
- Data alignment and synchronization
- Other data cleaning processes as necessary

### Workflow

The classification workflow comprises the following key steps:

#### 1. Preprocessing

In this step, we clean and prepare the fNIRS data for feature extraction and classification.

#### 2. Feature Extraction

Feature extraction is a crucial part of our classification pipeline. We extract both time-domain and frequency-domain features from the preprocessed data. These features may include:

- Variance
- Median
- Entropy
- Frequency domain features (e.g., power spectral density)

#### 3. Feature Selection

To improve the efficiency and effectiveness of our classification models, we perform feature selection. Feature selection methods such as forward selection, backward elimination, and Principal Component Analysis (PCA) for dimensionality reduction are employed.

#### 4. Classification

The heart of our project is the classification of fNIRS data. We apply various machine learning classifiers and strategies, including ensemble learning techniques.


## Contributing

We welcome contributions from the community to enhance and expand this project. If you have ideas for improvements, wish to report issues, or want to contribute code, please review our [Contribution Guidelines](CONTRIBUTING.md).

