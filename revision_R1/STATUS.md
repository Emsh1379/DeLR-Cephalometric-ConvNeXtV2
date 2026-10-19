# R1 revision runs - progress

Last update: 2026-10-19 18:36 (server UTC 15:06). Updated automatically every 20 minutes while the runs are going.

| Run | Status | Epochs | Best validation MRE (mm) | Last validation MRE (mm) |
|---|---|---|---|---|
| ConvNeXt V2, fixed-scale L1 loss (8 px), ISBI 2015 | finished | 67/150 | 31.369 | 35.219 |
| ConvNeXt V2, fixed-scale L1 loss (32 px), ISBI 2015 | finished | 100/150 | 40.823 | 40.907 |
| ConvNeXt V2, fixed-scale L1 loss (32 px), peak lr 5e-5, ISBI 2015 | training | 42/150 | 103.961 | 103.961 |
| ConvNeXt V2, Aariz all 29 landmarks | training | 97/200 | 1.092 | 1.097 |
| ResNet-34 dual encoder, ISBI 2015 | queued | 0/150 |  |  |

Validation set: ISBI 2015 Test1 (ISBI runs) and the Aariz validation split (Aariz run). Test results appear in the run folders once a run has finished.
Analyses (CPU / inference only) are in `results/revision/`. Training logs and predictions are in `outputs/<run>/`.
