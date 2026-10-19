# Cross-dataset evaluation

In-domain = model trained on the evaluation dataset, scored on the same images and the same shared landmarks. Millimetres follow the convention of the evaluation dataset.

| Trained on | Evaluated on | Shared landmarks | In-domain MRE (95% CI) | In-domain SDR 2 mm (%) | Cross-dataset MRE (95% CI) | Cross-dataset SDR 2 mm (%) | Cross-dataset SDR 4 mm (%) | Change in MRE |
|---|---|---|---|---|---|---|---|---|
| ISBI 2015 | Aariz test (n=150) | 17 | 0.96 (0.88-1.05) | 90.3 | 3.59 (2.97-4.27) | 67.5 | 86.1 | +274% |
| CephAdoAdu | Aariz test (n=150) | 9 | 1.04 (0.96-1.12) | 87.6 | 2.75 (2.56-2.94) | 60.1 | 83.4 | +165% |
| Aariz | ISBI 2015 Test2 (n=100) | 17 | 1.31 (1.26-1.36) | 77.8 | 1.75 (1.68-1.81) | 68.0 | 91.4 | +34% |
| CephAdoAdu | ISBI 2015 Test2 (n=100) | 8 | 1.25 (1.18-1.34) | 82.5 | 2.63 (2.55-2.71) | 69.2 | 85.1 | +110% |
| Aariz | CephAdoAdu validation part (n=300) | 9 | 1.06 (1.02-1.10) | 88.6 | 2.96 (2.69-3.27) | 62.6 | 81.6 | +179% |
| ISBI 2015 | CephAdoAdu validation part (n=300) | 8 | 0.93 (0.89-0.97) | 91.5 | 5.05 (4.64-5.49) | 57.5 | 74.1 | +444% |

## Per-landmark MRE (mm): in-domain -> cross-dataset

**isbi->aariz**: a_point 0.87 -> 3.57; ans 1.21 -> 2.27; b_point 0.83 -> 1.97; menton 0.49 -> 0.82; nasion 0.86 -> 1.86; orbitale 1.29 -> 3.12; pogonion 0.55 -> 1.23; pns 1.41 -> 2.67; sella 0.63 -> 2.72; articulare 1.58 -> 10.44; gnathion 0.41 -> 0.76; gonion 1.47 -> 8.58; porion 1.55 -> 14.72; lower_incisor_tip 0.73 -> 1.16; upper_incisor_tip 0.58 -> 1.07; lower_lip 1.10 -> 2.89; upper_lip 0.74 -> 1.18
**ceph->aariz**: a_point 0.87 -> 1.65; ans 1.21 -> 2.23; orbitale 1.29 -> 1.39; pogonion 0.55 -> 11.16; porion 1.55 -> 2.27; lower_incisor_tip 0.73 -> 0.97; upper_incisor_apex 1.15 -> 2.01; upper_incisor_tip 0.58 -> 0.75; lower_incisor_apex 1.39 -> 2.29
**aariz->isbi**: sella 0.48 -> 0.67; nasion 0.91 -> 2.68; orbitale 2.26 -> 1.60; porion 1.95 -> 1.94; a_point 1.25 -> 2.74; b_point 2.52 -> 1.20; pogonion 0.67 -> 1.08; menton 0.66 -> 0.90; gnathion 0.53 -> 0.77; gonion 1.22 -> 1.60; lower_incisor_tip 0.96 -> 1.36; upper_incisor_tip 0.66 -> 0.84; upper_lip 2.59 -> 3.60; lower_lip 2.04 -> 4.22; pns 1.03 -> 2.01; ans 1.30 -> 1.28; articulare 1.18 -> 1.24
**ceph->isbi**: orbitale 2.26 -> 1.38; porion 1.95 -> 1.74; a_point 1.25 -> 2.14; pogonion 0.67 -> 11.40; lower_incisor_tip 0.96 -> 0.93; upper_incisor_tip 0.66 -> 0.66; subnasale 0.96 -> 1.42; ans 1.30 -> 1.37
**aariz->ceph**: a_point 1.05 -> 2.01; ans 0.89 -> 2.07; upper_incisor_tip 0.50 -> 1.13; upper_incisor_apex 1.28 -> 3.05; orbitale 1.17 -> 2.05; porion 1.53 -> 1.78; lower_incisor_tip 0.60 -> 1.06; lower_incisor_apex 1.39 -> 2.75; pogonion 1.11 -> 10.72
**isbi->ceph**: a_point 1.05 -> 3.50; ans 0.89 -> 1.94; upper_incisor_tip 0.50 -> 1.52; orbitale 1.17 -> 2.59; porion 1.53 -> 16.15; lower_incisor_tip 0.60 -> 1.94; subnasale 0.57 -> 1.78; pogonion 1.11 -> 11.00