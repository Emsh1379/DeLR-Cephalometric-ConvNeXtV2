# R1 revision runs - progress

Last update: 2026-10-19 10:52 (server UTC 07:22). Updated automatically every 20 minutes while the runs are going.

| Run | Status | Epochs | Best validation MRE (mm) | Last validation MRE (mm) |
|---|---|---|---|---|
| ConvNeXt V2, fixed-scale L1 loss (8 px), ISBI 2015 | training | 34/150 | 40.78 | 41.333 |
| ConvNeXt V2, fixed-scale L1 loss (32 px), ISBI 2015 | queued | 0/150 |  |  |
| ConvNeXt V2, Aariz all 29 landmarks | training | 14/200 | 117.493 | 117.493 |
| ResNet-34 dual encoder, ISBI 2015 | queued | 0/150 |  |  |

Validation set: ISBI 2015 Test1 (ISBI runs) and the Aariz validation split (Aariz run). Test results appear in the run folders once a run has finished.
Analyses (CPU / inference only) are in `results/revision/`. Training logs and predictions are in `outputs/<run>/`.
