# MIS: Mechanistic Interpretability through Superposition

This repository contains the official codebase and supplementary materials for the paper **[Mechanistic Interpretability through Superposition]** by [Your Name and Co-authors]. The work explores the mechanisms underlying superposition in neural networks, providing insights into sparse coding and representation disentanglement.

📄 **Read the paper**: [https://brendel-group.github.io/mis/](https://brendel-group.github.io/mis/)

## Table of Contents
- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Running Experiments](#running-experiments)
  - [Training a Model](#training-a-model)
  - [Visualizing Results](#visualizing-results)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## About

The **MIS** project aims to provide insights into superposition mechanisms and sparse coding in neural networks. The repository includes:
- Code to reproduce experiments presented in the paper.
- Pre-trained models and configuration files.
- Tools for analyzing and visualizing results.

## Features
- Implements superposition analysis and sparse coding techniques.
- Provides reproducible experiments with pre-configured setups.
- Includes tools for visualizing superposition effects in neural networks.
- Designed for easy extension to related research topics.

## Installation

To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/brendel-group/mis.git
cd mis
pip install -r requirements.txt
```

## Usage
First copy `MIS.py` and `metric.py` into your working directory:
```bash
cp <local-path-to-repo>/MIS.py ./
cp <local-path-to-repo>/metric.py ./
```
Then import `MIS.py`:
```
import MIS
```
To run psychophysics experiment and calculate the Mechanistic Interpretability Score of neuron units, we load in images and the associate neuron units activations into a custom object class `task_config`, then specify the follow psychophysics-specific parametres:
- `seed`: ensures reproducibility
- `device`: either `'cpu'` or `'cuda'`
- `metric_type`: the type of image metric function to use. Currently supports `'dreamsim'` and `'lpips'`
- `K`: the number of explanation images in an explanation set in a single psychophysics task
- `N`: the number of psychophysics tasks for a neuron unit
- `quantile`: Quantile threshold for selecting images
- `alpha`: Optional threshold parameter for MIS calculation
```
from MIS import task_config, run_psychophysics

# Assume we have the following raw data:
# image_list: a python list of PIL Image objects (length: N_images)
# activations: a torch tensor containing neuron unit activations in response to the images in image_list, shape: N_images x N_units

device = 'cuda'
task_data = task_config(device=device, image_set=image_list, activations=activations)

seed = 117
metric_type = "dreamsim"
K = 9
N = 100
quantile = 0.25
alpha = None

# Now we run psychophysics experiment
MIS = run_psychophysics(seed, task_data, metric_type=metric_type, K=K, N=N, quantile=quantile, alpha=alpha)
```
