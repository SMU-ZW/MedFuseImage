## Code for [MedFuse](https://arxiv.org/abs/2207.07027): Multi-modal fusion and benchmarking with clinical time-series and chest X-ray images


Table of contents
=================
# MedFuse image only branch

CXR only part from MedFuse for MIMIC-IV CXR.

## 1. Environment

Install from `requirements.txt`:

## 2. Processing

1. Change directory in resize and run resize on CXR image.
2. Under datasets/cxr_dataset.py, change directory of mortality & readmission (split could be adjusted as fusion applied).
3. Run sh ./scripts/mortality/train/MedFuse_mort_superpod.sh && sh ./scripts/mortality/train/MedFuse_read_superpod.sh.