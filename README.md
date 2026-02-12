# AuxDPO

Official implementation of **Auxiliary-Enhanced Direct Preference Optimization (AuxDPO)**.

> 🎤 Accepted as **ICLR 2026 Oral**  
> 📄 Paper: https://arxiv.org/abs/2510.20413  

---

## Overview

AuxDPO extends Direct Preference Optimization (DPO) by incorporating auxiliary objectives to improve alignment stability and performance.

This repository contains:

- 📓 Training notebook  
- 💾 Model checkpoints  
- 📊 Evaluation scripts  
- 🗂 Dataset preprocessing utilities  

> ⚠️ **Note:** All experiments were run on **AMD MI300X clusters**.  
Some components are optimized for AMD ROCm environments.

---

## Repository Structure

AuxDPO/
│
├── aux-dpo.ipynb # Main training & experiment notebook
│
├── checkpoints/ # Saved model checkpoints
│
├── evaluation/ # Evaluation scripts and metrics
│
├── prepare_datasets/ # Dataset preprocessing utilities
│
└── README.md


---

## Environment Setup

Experiments were conducted on:

- AMD MI300X GPUs  
- ROCm-based PyTorch builds  
- Distributed multi-GPU setup  

If you are using NVIDIA GPUs, minor modifications may be required (especially related to device placement and backend configuration).

Example installation:

```bash
pip install torch torchvision torchaudio
pip install transformers datasets accelerate trl

(Adjust according to your GPU stack.)

Running Training

The primary implementation is inside:

aux-dpo.ipynb


This notebook contains:

Model initialization

DPO + auxiliary objective training

Logging and checkpointing

Validation

Dataset Preparation

Dataset preprocessing scripts are located in:

prepare_datasets/


These scripts:

Clean raw preference datasets

Format them for DPO-style training

Perform tokenization

Generate train/validation splits

Evaluation

Evaluation scripts are available inside:

evaluation/


These scripts support:

Preference accuracy

Reward modeling comparison

Benchmark-based evaluation

Hardware Notes

Because experiments were run on AMD MI300X:

Some ROCm-specific configurations may appear

FSDP / distributed training code may assume certain topology

Kernel-level optimizations may differ from CUDA

Users running on NVIDIA GPUs may need to adjust:

torch.cuda vs torch.device

Backend configuration

Flash attention compatibility
