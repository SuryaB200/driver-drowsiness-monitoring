# Dataset Directory Structure

This folder contains raw and processed datasets for the Commercial Fleet Driver Drowsiness and Distraction Monitoring System.

## Directory Layout

```
datasets/
├── README.md               <-- This file
├── NTHU-DDD/               <-- Extract official NTHU-DDD dataset here
│   ├── Subject01/
│   ├── Subject02/
│   └── ...
├── UL-DD/                  <-- Optional UL-DD dataset
└── processed/              <-- Partitioned subject-independent splits
    ├── manifest.json       <-- Split metadata & subject tracking
    ├── train/              <-- Subjects 1-8
    │   ├── Alert/
    │   └── Drowsy/
    ├── val/                <-- Subjects 9-10
    │   ├── Alert/
    │   └── Drowsy/
    └── test/               <-- Subjects 11-12 (Held-out evaluation)
        ├── Alert/
        └── Drowsy/
```

To initialize or verify partitions:
```powershell
.\.venv\Scripts\python.exe training\prepare_dataset.py
```
