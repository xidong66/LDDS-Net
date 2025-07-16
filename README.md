# LDGDN: Lightweight Dynamic Granularity Denoised Network
> **Official Tensorflow implementation**  
> Paper: *Lightweight Dynamic Granularity Denoised Network for Medical Time-Series Classification*  
> https://github.com/xidong66/LDDS-Net

---

## 📌 TL;DR
A **parameter-efficient** yet **noise-robust** architecture for medical time-series classification.  
Achieves **SOTA on 6/7 public datasets** with **< 0.5 M parameters** and **real-time** inference on **Raspberry Pi 5**.

---

## 🔑 Key Components

| Module | Purpose | Core Idea |
|---|---|---|
| **LCA** – Lightweight Channel Aggregation | Cross-channel fusion | Channel-mixing + Hadamard product → 3–5× param reduction |
| **RSM** – Residual Shrinkage Module | Noise suppression | Adaptive soft-thresholding → 0 dB SNR still works |
| **DGC** – Dynamic Granularity Controller | Scale selection | Auto-pick temporal granularity → 40 % FLOP drop w/o loss |

---

## 🚀 Benchmarks (single 6-second ECG lead)

| Dataset | Accuracy | Params | FLOPs | Noise-Robust (0 dB) |
|---|---|---|---|---|
| **CinC2017** | **94.1 %** | **0.47 M** | **0.82 G** | **88.3 %** |
| **Ninapro DB1** | **92.7 %** | *same* | *same* | **85.9 %** |
| **xxx** | **95.3 %** | *same* | *same* | **87.2 %** |
| PTB-XL, Chapman, Georgia, Shaoxing | **Top-1** on 6/7 | *same* | *same* | ≥ 84 % |

---


## 📦 Quick Start

```bash
git clone https://github.com/xidong66/LDDS-Net.git
cd LDDS-Net
pip install -r  # tensorflow>=2.6, keras>=2.6