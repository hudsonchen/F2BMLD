# Nested Kernel Quadrature

This repository contains the implementation of the code for the paper "Towards a Unified Analysis of Neural Networks in Nonparametric Instrumental Variable Regression: Optimization and Generalization". 

## Installation

To install the required packages, run the following command:
```
pip install -r requirements.txt
```

## Reproducing Results

### 1. Synthetic Experiment

To reproduce the results for the synthetic experiment (Figure 2 (Left)), run the following command:

`python main/run_f2bmld.py --policy_noise_level 0.0 --noise_level 0.2 --lagrange_reg 0.3 --seed 0 --max_steps 50_000 --batch_size 32`

You can vary the Lagrangian multiplier lambda by altering the argument '--lagrange_reg 0.3', vary the environment noise level by altering the argument '--noise_level 0.2'. 



