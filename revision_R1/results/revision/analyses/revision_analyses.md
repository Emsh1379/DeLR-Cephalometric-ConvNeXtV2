# Revision analyses

## A. CephAdoAdu: adolescents vs adults, equivalence (TOST)

Unit of analysis: per-image mean radial error (n = 150 adolescents, 150 adults). Difference adolescent - adult = -0.077 mm; 90% CI -0.144 to -0.010 (Welch), -0.143 to -0.010 (bootstrap); 95% CI -0.157 to +0.002 (Welch), -0.1546 to +0.0041 (bootstrap).

| Equivalence margin | p (H0: diff <= -margin) | p (H0: diff >= +margin) | TOST p | Equivalent at alpha 0.05 |
|---|---|---|---|---|
| ±0.25 mm | 1.331e-05 | 1.03e-14 | 1.331e-05 | yes |
| ±0.50 mm | 0 | 4.177e-35 | 4.177e-35 | yes |

SDR at 2 mm: difference +3.20 percentage points, 90% CI +1.07 to +5.33.

| Margin | TOST p | Equivalent |
|---|---|---|
| ±5 percentage points | 0.08256 | no |
| ±10 percentage points | 1.403e-07 | yes |

## C. Aariz: search for interchanged or implausible labels (all 1000 images)

Lip check: labrale superius lying below labrale inferius (larger y) in an annotator's labels.

| Split | Images | Junior: lips interchanged | Senior: lips interchanged |
|---|---|---|---|
| train | 700 | 1 | 1 |
| valid | 150 | 0 | 0 |
| test | 150 | 1 | 1 |

| Split | Image | Device | Junior | Senior |
|---|---|---|---|---|
| train | cks2ip8g52b530yufc8z6edv8 | ART Plus | yes | yes |
| test | cl5lg05un01hm074k3dwy9l5q | ProMax with ProTouch | yes | yes |

Junior-senior distance above 5 mm (any landmark): 251 landmark annotations.

| Landmark | Annotations with junior-senior distance > 5 mm |
|---|---|
| Gonion | 39 |
| Porion | 36 |
| Orbitale | 24 |
| Ramus | 24 |
| Upper Incisor Apex | 20 |
| Condylion | 19 |
| Anterior Nasal Spine | 12 |
| Articulare | 12 |
| Soft Tissue Nasion | 11 |
| Nasion | 11 |
| B-point | 6 |
| Lower Incisor Apex | 5 |
| Upper Incisor Tip | 5 |
| Posterior Nasal Spine | 5 |
| A-point | 3 |
| Lower Incisor Tip | 3 |
| Soft Tissue Pogonion | 3 |
| Menton | 3 |
| Lower 2nd PM Cusp Tip | 2 |
| Lower Molar Cusp Tip | 2 |
| Upper Molar Cusp Tip | 2 |
| Upper 2nd PM Cusp Tip | 1 |
| Labrale inferius | 1 |
| Pronasale | 1 |
| Pogonion | 1 |

Full list in revision_analyses.json (C_label_check.junior_senior_gt_mm).

## B. Aariz test set by device (26 landmarks, lip labels of flagged test images corrected)

| Device | Images | Pixel size (mm) | MRE (mm) | MRE 95% CI | SDR 2 mm % (95% CI) |
|---|---|---|---|---|---|
| ART Plus | 55 | 0.1 | 1.411 | 1.294-1.549 | 80.77 (78.3-83.2) |
| Veraviewepocs 2D | 26 | 0.144 | 0.687 | 0.647-0.730 | 96.75 (95.6-97.8) |
| Hyperion X5 | 21 | 0.089 | 0.422 | 0.371-0.490 | 99.63 (99.1-100.0) |
| ProMax with ProTouch | 20 | 0.139 | 1.191 | 1.106-1.286 | 86.54 (84.0-89.0) |
| Rotograph EVO | 13 | 0.135 | 1.993 | 1.795-2.244 | 64.79 (60.1-68.6) |
| Smart3D | 9 | 0.1 | 0.428 | 0.393-0.473 | 100.00 (100.0-100.0) |
| ProMax 2D | 6 | 0.139 | 1.102 | 0.959-1.266 | 85.26 (76.9-92.3) |
| All | 150 |  | 1.097 | 1.004-1.191 | 86.90 (84.8-88.9) |

Kruskal-Wallis on per-image MRE: H = 119.31, p = 2.28e-23. Overall MRE uncorrected 1.105 mm, corrected 1.097 mm.

## E. Aariz 29 landmarks

Skipped: /kaggle/working/DeLR-Cephalometric-ConvNeXtV2/checkpoints/Aariz_29/test_predictions.json not found.

## D. ISBI 2015: clinical measurements and classification

Measurements and class limits as defined for the ISBI 2015 challenge (Wang et al. 2016; Lindner et al. 2016). Reference = measurement from the mean of the two clinicians' landmarks; prediction = from the model's landmarks. Sign/orientation conventions of ODI, APDI and FMA were fixed on the 150 TRAINING images by maximising agreement with the senior clinician's classification labels supplied with the dataset (no test image used).

Agreement of the reference-landmark classes with the senior clinician's labels on the training images (check that the measurement formulas reproduce the official definitions; values near 100% expected):

| Measurement | Agreement on training images (%) |
|---|---|
| ANB | 85.3 |
| SNB | 92.7 |
| SNA | 83.3 |
| ODI | 88.0 |
| APDI | 84.7 |
| FHI | 94.0 |
| FMA | 62.7 |
| MW | 78.7 |

Chosen conventions: {'odi_main': 1, 'odi_pp': 0, 'apdi_fa': 1, 'apdi_ab': 1, 'apdi_pp': 0, 'fma': 0}

| Split | Measurement | Mean absolute difference | Classification agreement % (95% CI) | Prediction vs senior label (%) | Reference landmarks vs senior label (%) |
|---|---|---|---|---|---|
| Test1 | ANB | 1.18° | 80.0 (73.3-86.0) | 76.0 | 88.0 |
| Test1 | SNB | 1.38° | 89.3 (84.0-94.0) | 83.3 | 90.0 |
| Test1 | SNA | 1.90° | 72.7 (65.3-80.0) | 71.3 | 82.7 |
| Test1 | ODI | 2.43° | 86.0 (80.0-91.3) | 79.3 | 88.0 |
| Test1 | APDI | 2.41° | 84.0 (78.0-89.3) | 82.7 | 91.3 |
| Test1 | FHI | 0.014 | 85.3 (79.3-90.7) | 82.0 | 92.7 |
| Test1 | FMA | 1.46° | 93.3 (89.3-97.3) | 75.3 | 73.3 |
| Test1 | MW | 0.94 mm | 83.3 (77.3-89.3) | 68.7 | 78.0 |
| Test2 | ANB | 0.86° | 83.0 (75.0-90.0) | 84.0 | 97.0 |
| Test2 | SNB | 1.10° | 88.0 (81.0-94.0) | 84.0 | 92.0 |
| Test2 | SNA | 1.18° | 75.0 (66.0-83.0) | 77.0 | 90.0 |
| Test2 | ODI | 3.16° | 76.0 (67.0-84.0) | 68.0 | 90.0 |
| Test2 | APDI | 1.84° | 89.0 (82.0-95.0) | 85.0 | 94.0 |
| Test2 | FHI | 0.009 | 91.0 (85.0-96.0) | 93.0 | 96.0 |
| Test2 | FMA | 0.97° | 97.0 (93.0-100.0) | 68.0 | 70.0 |
| Test2 | MW | 0.79 mm | 82.0 (74.0-89.0) | 71.0 | 82.0 |

Mean classification agreement: Test1 84.2%, Test2 85.1%.

## F. Aspect ratio (height/width) of the original image vs per-image MRE

Images were resized to 1024 x 1024 without preserving aspect ratio; |log(height/width)| measures how anisotropic the resize was. ISBI 2015 images all have the same size (1935 x 2400 px, height/width 1.24).

| Evaluation set | Height/width range | Spearman rho, abs(log aspect) vs MRE | MRE (mm) by aspect-ratio tertile |
|---|---|---|---|
| Aariz test | 0.76-1.30 (median 1.13) | -0.196 (p = 0.0161) | 0.76-1.13: 1.153; 1.13-1.24: 1.322; 1.24-1.30: 0.840 |
| CephAdoAdu validation part | 0.43-1.54 (median 0.89) | -0.307 (p = 5.82e-08) | 0.43-0.84: 0.867; 0.84-1.10: 1.082; 1.10-1.54: 1.082 |
