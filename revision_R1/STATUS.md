# R1 revision runs - progress

Last update: 2026-10-19 15:37 (server UTC 12:08). Updated automatically every 20 minutes while the runs are going.

| Run | Status | Epochs | Best validation MRE (mm) | Last validation MRE (mm) |
|---|---|---|---|---|
| ConvNeXt V2, fixed-scale L1 loss (8 px), ISBI 2015 | finished | 67/150 | 31.369 | 35.219 |
| ConvNeXt V2, fixed-scale L1 loss (32 px), ISBI 2015 | training | 47/150 | 40.939 | 42.282 |
| ConvNeXt V2, Aariz all 29 landmarks | training | 52/200 | 1.258 | 1.287 |
| ResNet-34 dual encoder, ISBI 2015 | queued | 0/150 |  |  |

Validation set: ISBI 2015 Test1 (ISBI runs) and the Aariz validation split (Aariz run). Test results appear in the run folders once a run has finished.
Analyses (CPU / inference only) are in `results/revision/`. Training logs and predictions are in `outputs/<run>/`.
