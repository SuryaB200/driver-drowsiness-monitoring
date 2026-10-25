# Dataset Sources & Access Guidelines

This document details the academic driver drowsiness datasets supported by the Commercial Fleet Driver Monitoring System.

---

## 1. Primary Dataset: NTHU-DDD

### Overview
- **Name:** National Tsing Hua University Driver Drowsiness Detection (NTHU-DDD)
- **Institution:** Computer Vision Lab, National Tsing Hua University, Taiwan
- **Official URL:** [https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/](https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/)
- **Subjects:** 36 multi-ethnic subjects (18 male, 18 female)
- **Scenarios:** 5 driving simulation scenarios per subject:
  1. Normal driving (bareface)
  2. Slow blink / micro-sleep with glasses
  3. Yawning / head nodding
  4. Night condition (infrared IR illumination)
  5. Driving with sunglasses

### Legal Access Protocol
NTHU-DDD requires signing an academic release agreement. To request access:
1. Visit the official dataset page: [https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/](https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/)
2. Complete the academic license application form with your university email address.
3. Upon receiving the download credentials from the dataset administrators, extract the dataset archive into:
   ```
   datasets/NTHU-DDD/
   ```

---

## 2. Alternative Dataset: UL-DD

### Overview
- **Name:** University of Luxembourg Driver Drowsiness (UL-DD) Dataset
- **Institution:** SnT - Interdisciplinary Centre for Security, Reliability and Trust, University of Luxembourg
- **Modalities:** Synchronized facial video streams paired with electroencephalography (EEG) brainwave activity (Fp1, Fp2, C3, C4 channels) and Karolinska Sleepiness Scale (KSS) self-evaluations.
- **Placement:** Extract video sequences into:
   ```
   datasets/UL-DD/
   ```

---

## 3. Benchmark Academic Partitioning

For immediate validation without violating dataset non-redistribution licenses, the project includes an automated subject-independent partitioner:

```bash
# Run from repository root:
.\.venv\Scripts\python.exe training\prepare_dataset.py
```

This generates `datasets/processed/` with:
- `train/` (400 samples, Subjects 1 to 8)
- `val/` (100 samples, Subjects 9 and 10)
- `test/` (100 samples, Subjects 11 and 12)
- `manifest.json` ensuring 100% subject-independent evaluation.
