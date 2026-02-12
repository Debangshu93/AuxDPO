# AuxDPO

Official implementation of **Auxiliary-Enhanced Direct Preference Optimization (AuxDPO)**.

> 🎤 **Accepted as ICLR 2026 Oral** > 📄 **Paper:** [https://arxiv.org/abs/2510.20413](https://arxiv.org/abs/2510.20413)

---

## Overview

AuxDPO extends Direct Preference Optimization (DPO) by incorporating auxiliary objectives to improve alignment stability and performance.

This repository contains:

- 📓 **Training notebook** - 📊 **Evaluation scripts** - 🗂 **Dataset preprocessing utilities** > ⚠️ **Note:** All experiments were run on **AMD MI300X clusters**. Some components are optimized for AMD ROCm environments.

---

## Repository Structure

```text
AuxDPO/
├── aux-dpo.ipynb          # Main training & experiment notebook
├── evaluation/            # Evaluation scripts and metrics
├── prepare_datasets/      # Dataset preprocessing utilities
└── README.md
```

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
```
(Adjust according to your GPU stack.)

---

## 🚀 Running Training

The primary implementation is contained within: `aux-dpo.ipynb`

This notebook handles the end-to-end pipeline:
* **Model Initialization:** Loading base and reference models.
* **Core Training:** DPO combined with our auxiliary objective.
* **Logging:** Integrated checkpointing and metric tracking.
* **Validation:** Real-time performance monitoring.

---

## 📊 Dataset Preparation

Preprocessing scripts are located in `prepare_datasets/`. These utilities:
* **Clean** raw preference datasets for quality control.
* **Format** data specifically for DPO-style pair training.
* **Tokenize** and generate optimized train/validation splits.

---

## 📈 Evaluation

Evaluation scripts are available in the `evaluation/` directory, supporting:
* **Preference Accuracy:** Measuring model alignment with chosen labels.
* **Reward Comparison:** Contrastive analysis against standard DPO.
* **Benchmarks:** Scripts for standard LLM evaluation frameworks.

---

## ⚙️ Hardware Notes

Since experiments were optimized for **AMD MI300X**, please note:
* **ROCm Configurations:** You may find specific flags for ROCm-based PyTorch.
* **Distributed Training:** FSDP code assumes the topology of an MI300X cluster.
* **NVIDIA Compatibility:** If running on CUDA, you may need to adjust:
    * `torch.cuda` vs `torch.device` calls.
    * Distributed backend configurations.
    * Flash Attention versioning.
