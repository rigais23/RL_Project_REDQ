# REDQ: Randomized Ensembled Double Q-Learning 


Implementation of the **REDQ** algorithm and a **Soft Actor-Critic (SAC)** baseline for continuous control environments. 

This repository contains the source code, experimental setups, and results for the Reinforcement Learning course project (Master in Artificial Intelligence at UPC).

## Agents in Action

*Here are the trained REDQ agents ($G=20$) successfully solving the benchmark environments:*

<p align="center">
  <img src="gifs/bipedal_walker_redq.gif" 
  alt="BipedalWalker REDQ" width="45%">
  &nbsp; &nbsp; &nbsp; &nbsp;
  <img src="gifs/hopper_redq.gif" alt="Hopper REDQ" width="45%">
</p>
<p align="center">
  <em>Left: BipedalWalker-v3. Right: Hopper-v5.</em>
</p>

## Project Overview

Model-free Deep Reinforcement Learning often suffers from high sample inefficiency. The **REDQ**  algorithm bridges the gap between model-free and model-based sample efficiency by using a high **Update-To-Data (UTD) ratio**. 

To solve the overestimation bias caused by multiple backpropagation updates per environment step, REDQ uses an ensemble of $N$ Q-networks and a randomized in-target minimization subset $M$.

### Features of this Implementation
* **Unified Algorithmic Framework:** SAC is dynamically instantiated by restricting the REDQ class parameters ($N=2$, $M=2$, $G=1$), highlighting the theoretical continuity between the algorithms.
* **Highly Modular Architecture:** Separation of Replay Buffer, Neural Networks, and Agent Logic.
* **PyTorch Optimization:** Critic gradients are frozen during Actor updates to significantly reduce computational overhead (wall-clock time) during high-UTD training loops.

## Repository Structure
```text
├── src/
│   ├── redq.py         # Unified REDQ and SAC agent logic
│   ├── networks.py     # Actor and Ensemble Critic neural architectures
│   └── buffer.py       # Replay Buffer implementation
├── notebooks/
  ├── config.py           # Hyperparameters and ablation study configurations
  └── training_notebook.ipynb  # Main training loop and evaluation script
├── helpers/            # Utils for plotting and saving raw logs
├── gifs/               # Converted gifs for the README
└── README.md
````

## Author & Acknowledgments

[**Ricard Garcia Isern**](www.ricardgarcia.com) 