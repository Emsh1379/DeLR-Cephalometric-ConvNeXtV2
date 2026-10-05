# Per-device accuracy, failure cases and rank correlations

Recomputed from the saved prediction files. A failure is a radial error above 4 mm.

## Aariz - accuracy by imaging device (test set)

| Device | N images | Pixel size (mm) | MRE +/- SD (mm) | MRE 95% CI | SDR@2.0 (%) | SDR@2.5 (%) | SDR@3.0 (%) | SDR@4.0 (%) |
|---|---|---|---|---|---|---|---|---|
| ART Plus | 55 | 0.1 | 1.411 +/- 1.579 | 1.290-1.548 | 80.77 | 86.50 | 90.42 | 94.83 |
| Veraviewepocs 2D | 26 | 0.144 | 0.687 +/- 0.560 | 0.647-0.729 | 96.75 | 98.22 | 99.41 | 99.85 |
| Hyperion X5 | 21 | 0.089 | 0.422 +/- 0.374 | 0.370-0.489 | 99.63 | 99.82 | 99.82 | 100.00 |
| ProMax with ProTouch | 20 | 0.139 | 1.253 +/- 1.371 | 1.125-1.411 | 86.15 | 92.50 | 95.19 | 97.69 |
| Rotograph EVO | 13 | 0.135 | 1.993 +/- 1.629 | 1.789-2.241 | 64.79 | 71.30 | 79.59 | 88.17 |
| Smart3D | 9 | 0.1 | 0.428 +/- 0.315 | 0.392-0.473 | 100.00 | 100.00 | 100.00 | 100.00 |
| ProMax 2D | 6 | 0.139 | 1.102 +/- 0.748 | 0.958-1.265 | 85.26 | 92.95 | 98.72 | 100.00 |
| **All** | 150 |  | 1.105 +/- 1.313 |  | 86.85 | 90.95 | 93.90 | 96.74 |

Check: 7 devices, 150 images; image-weighted mean of the device MREs = 1.105 mm (overall 1.105 mm). Kruskal-Wallis on per-image mean error across devices: H = 118.28, p = 3.741e-23.

Aariz inter-observer (junior vs senior) distance on the same landmarks: 0.520 +/- 1.113 mm (model vs their mean: 1.105 mm).

## Failure cases (named by landmark)

### Aariz (test, 26 landmarks)

127 of 3900 landmark predictions (3.26 %) exceed 4 mm; 58 of 150 images contain at least one.

Failures (> 4 mm) by landmark, most frequent first:

| # | Landmark | n > 4 mm | % of images | MRE (mm) | Max error (mm) |
|---|---|---|---|---|---|
| 10 | Ramus | 21 | 14.0 | 2.094 | 10.47 |
| 13 | Condylion | 16 | 10.7 | 2.000 | 16.52 |
| 16 | Porion | 12 | 8.0 | 1.552 | 7.89 |
| 12 | Articulare | 11 | 7.3 | 1.577 | 20.29 |
| 15 | Gonion | 11 | 7.3 | 1.473 | 7.15 |
| 20 | Upper 2nd PM Cusp Tip | 9 | 6.0 | 1.336 | 7.14 |
| 2 | Anterior Nasal Spine | 7 | 4.7 | 1.214 | 5.38 |
| 8 | Posterior Nasal Spine | 7 | 4.7 | 1.412 | 11.58 |
| 23 | Upper Molar Cusp Tip | 7 | 4.7 | 1.321 | 11.11 |
| 17 | Lower 2nd PM Cusp Tip | 6 | 4.0 | 1.277 | 12.30 |
| 19 | Lower Molar Cusp Tip | 5 | 3.3 | 1.058 | 10.08 |
| 25 | Labrale inferius | 4 | 2.7 | 1.097 | 16.61 |
| 6 | Orbitale | 3 | 2.0 | 1.292 | 14.48 |
| 5 | Nasion | 2 | 1.3 | 0.859 | 6.81 |
| 21 | Upper Incisor Apex | 2 | 1.3 | 1.154 | 4.45 |
| 24 | Lower Incisor Apex | 2 | 1.3 | 1.390 | 4.89 |
| 9 | Pronasale | 1 | 0.7 | 0.791 | 8.85 |
| 26 | Labrale superius | 1 | 0.7 | 0.742 | 16.63 |
| 1 | A-point | 0 | 0.0 | 0.874 | 3.24 |
| 3 | B-point | 0 | 0.0 | 0.834 | 3.50 |
| 4 | Menton | 0 | 0.0 | 0.486 | 2.38 |
| 7 | Pogonion | 0 | 0.0 | 0.553 | 3.27 |
| 11 | Sella | 0 | 0.0 | 0.631 | 3.00 |
| 14 | Gnathion | 0 | 0.0 | 0.414 | 2.26 |
| 18 | Lower Incisor Tip | 0 | 0.0 | 0.726 | 2.51 |
| 22 | Upper Incisor Tip | 0 | 0.0 | 0.577 | 3.77 |

The 15 largest individual errors:

| Rank | Image | # | Landmark | Error (mm) | Group / device |
|---|---|---|---|---|---|
| 1 | cks2ip8g62b7b0yuffrlb7xa1 | 12 | Articulare | 20.29 | ART Plus |
| 2 | cl5lg05un01hm074k3dwy9l5q | 26 | Labrale superius | 16.63 | ProMax with ProTouch |
| 3 | cl5lg05un01hm074k3dwy9l5q | 25 | Labrale inferius | 16.61 | ProMax with ProTouch |
| 4 | cks2ip8g62b7b0yuffrlb7xa1 | 13 | Condylion | 16.52 | ART Plus |
| 5 | cks2ip8g62b7b0yuffrlb7xa1 | 6 | Orbitale | 14.48 | ART Plus |
| 6 | cks2ip8g02asq0yuf59owfht0 | 17 | Lower 2nd PM Cusp Tip | 12.30 | ART Plus |
| 7 | cl777aklx03ua076whaie06kb | 8 | Posterior Nasal Spine | 11.58 | Rotograph EVO |
| 8 | cks2ip8fq29zp0yuf7cxd3bz2 | 17 | Lower 2nd PM Cusp Tip | 11.53 | ART Plus |
| 9 | cks2ip8g02asq0yuf59owfht0 | 23 | Upper Molar Cusp Tip | 11.11 | ART Plus |
| 10 | cks2ip8g62b7b0yuffrlb7xa1 | 10 | Ramus | 10.47 | ART Plus |
| 11 | cks2ip8g02asq0yuf59owfht0 | 19 | Lower Molar Cusp Tip | 10.08 | ART Plus |
| 12 | cl777aklx03ua076whaie06kb | 10 | Ramus | 9.71 | Rotograph EVO |
| 13 | cks2ip8fz2aq80yuf2pz61jh3 | 12 | Articulare | 9.08 | ART Plus |
| 14 | cl5lg05ul01gq074kb4xm8mml | 9 | Pronasale | 8.85 | ProMax with ProTouch |
| 15 | cl7ww1jcp9cfk084fgkua8l9o | 10 | Ramus | 8.79 | Rotograph EVO |

The 5 images with the highest mean error:

| Image | Image MRE (mm) | Landmarks > 4 mm | Worst landmark | Group / device |
|---|---|---|---|---|
| cks2ip8g62b7b0yuffrlb7xa1 | 3.42 | 4 | Articulare (20.29 mm) | ART Plus |
| cl777aklx03ua076whaie06kb | 2.99 | 7 | Posterior Nasal Spine (11.58 mm) | Rotograph EVO |
| cl76jfqyd0e3l07613b4o2677 | 2.87 | 10 | Nasion (6.33 mm) | Rotograph EVO |
| cks2ip8g02asq0yuf59owfht0 | 2.72 | 4 | Lower 2nd PM Cusp Tip (12.30 mm) | ART Plus |
| cks2ip8fq29zz0yuf15jsfhlu | 2.65 | 6 | Ramus (8.29 mm) | ART Plus |

### ISBI 2015 Test1 (19 landmarks)

46 of 2850 landmark predictions (1.61 %) exceed 4 mm; 38 of 150 images contain at least one.

Failures (> 4 mm) by landmark, most frequent first:

| # | Landmark | n > 4 mm | % of images | MRE (mm) | Max error (mm) |
|---|---|---|---|---|---|
| 4 | Porion | 9 | 6.0 | 1.820 | 5.19 |
| 2 | Nasion | 7 | 4.7 | 1.232 | 5.54 |
| 19 | Articulare | 7 | 4.7 | 1.582 | 6.50 |
| 10 | Gonion | 6 | 4.0 | 1.568 | 4.63 |
| 18 | Anterior nasal spine | 4 | 2.7 | 1.319 | 6.94 |
| 5 | Subspinale (A point) | 3 | 2.0 | 1.661 | 6.11 |
| 1 | Sella | 2 | 1.3 | 0.779 | 31.90 |
| 3 | Orbitale | 2 | 1.3 | 1.284 | 4.85 |
| 15 | Subnasale | 2 | 1.3 | 0.963 | 4.35 |
| 8 | Menton | 1 | 0.7 | 0.790 | 4.89 |
| 12 | Upper incisal incision | 1 | 0.7 | 0.753 | 6.36 |
| 16 | Soft tissue pogonion | 1 | 0.7 | 1.122 | 4.10 |
| 17 | Posterior nasal spine | 1 | 0.7 | 0.861 | 4.47 |
| 6 | Supramentale (B point) | 0 | 0.0 | 1.121 | 3.45 |
| 7 | Pogonion | 0 | 0.0 | 0.762 | 3.44 |
| 9 | Gnathion | 0 | 0.0 | 0.672 | 3.16 |
| 11 | Lower incisal incision | 0 | 0.0 | 0.968 | 3.57 |
| 13 | Upper lip | 0 | 0.0 | 1.221 | 2.88 |
| 14 | Lower lip | 0 | 0.0 | 0.875 | 2.39 |

The 15 largest individual errors:

| Rank | Image | # | Landmark | Error (mm) |
|---|---|---|---|---|
| 1 | 181 | 1 | Sella | 31.90 |
| 2 | 194 | 1 | Sella | 14.36 |
| 3 | 169 | 18 | Anterior nasal spine | 6.94 |
| 4 | 267 | 19 | Articulare | 6.50 |
| 5 | 208 | 12 | Upper incisal incision | 6.36 |
| 6 | 295 | 5 | Subspinale (A point) | 6.11 |
| 7 | 162 | 2 | Nasion | 5.54 |
| 8 | 269 | 18 | Anterior nasal spine | 5.39 |
| 9 | 243 | 19 | Articulare | 5.30 |
| 10 | 215 | 4 | Porion | 5.19 |
| 11 | 293 | 4 | Porion | 5.16 |
| 12 | 289 | 4 | Porion | 4.90 |
| 13 | 208 | 8 | Menton | 4.89 |
| 14 | 160 | 3 | Orbitale | 4.85 |
| 15 | 281 | 4 | Porion | 4.78 |

The 5 images with the highest mean error:

| Image | Image MRE (mm) | Landmarks > 4 mm | Worst landmark |
|---|---|---|---|
| 181 | 2.59 | 1 | Sella (31.90 mm) |
| 208 | 2.47 | 4 | Upper incisal incision (6.36 mm) |
| 194 | 2.34 | 1 | Sella (14.36 mm) |
| 267 | 1.68 | 2 | Articulare (6.50 mm) |
| 256 | 1.64 | 1 | Nasion (4.33 mm) |

Inter-observer (junior vs senior) distance: 2.363 +/- 2.164 mm.

### ISBI 2015 Test2 (19 landmarks)

102 of 1900 landmark predictions (5.37 %) exceed 4 mm; 77 of 100 images contain at least one.

Failures (> 4 mm) by landmark, most frequent first:

| # | Landmark | n > 4 mm | % of images | MRE (mm) | Max error (mm) |
|---|---|---|---|---|---|
| 16 | Soft tissue pogonion | 69 | 69.0 | 4.625 | 8.13 |
| 6 | Supramentale (B point) | 11 | 11.0 | 2.519 | 5.92 |
| 4 | Porion | 8 | 8.0 | 1.952 | 24.19 |
| 14 | Lower lip | 4 | 4.0 | 2.035 | 5.05 |
| 13 | Upper lip | 3 | 3.0 | 2.591 | 4.07 |
| 3 | Orbitale | 2 | 2.0 | 2.264 | 5.76 |
| 2 | Nasion | 1 | 1.0 | 0.907 | 4.81 |
| 10 | Gonion | 1 | 1.0 | 1.221 | 4.93 |
| 11 | Lower incisal incision | 1 | 1.0 | 0.960 | 6.03 |
| 18 | Anterior nasal spine | 1 | 1.0 | 1.304 | 4.18 |
| 19 | Articulare | 1 | 1.0 | 1.177 | 4.12 |
| 1 | Sella | 0 | 0.0 | 0.479 | 1.68 |
| 5 | Subspinale (A point) | 0 | 0.0 | 1.255 | 3.73 |
| 7 | Pogonion | 0 | 0.0 | 0.670 | 2.21 |
| 8 | Menton | 0 | 0.0 | 0.661 | 2.07 |
| 9 | Gnathion | 0 | 0.0 | 0.531 | 1.49 |
| 12 | Upper incisal incision | 0 | 0.0 | 0.657 | 3.02 |
| 15 | Subnasale | 0 | 0.0 | 0.962 | 3.10 |
| 17 | Posterior nasal spine | 0 | 0.0 | 1.030 | 2.86 |

The 15 largest individual errors:

| Rank | Image | # | Landmark | Error (mm) |
|---|---|---|---|---|
| 1 | 302 | 4 | Porion | 24.19 |
| 2 | 393 | 4 | Porion | 12.10 |
| 3 | 347 | 4 | Porion | 8.61 |
| 4 | 361 | 16 | Soft tissue pogonion | 8.13 |
| 5 | 389 | 16 | Soft tissue pogonion | 7.40 |
| 6 | 303 | 16 | Soft tissue pogonion | 7.22 |
| 7 | 346 | 16 | Soft tissue pogonion | 7.03 |
| 8 | 331 | 16 | Soft tissue pogonion | 6.93 |
| 9 | 347 | 16 | Soft tissue pogonion | 6.92 |
| 10 | 397 | 16 | Soft tissue pogonion | 6.80 |
| 11 | 355 | 16 | Soft tissue pogonion | 6.75 |
| 12 | 376 | 16 | Soft tissue pogonion | 6.67 |
| 13 | 371 | 16 | Soft tissue pogonion | 6.64 |
| 14 | 370 | 16 | Soft tissue pogonion | 6.42 |
| 15 | 360 | 16 | Soft tissue pogonion | 6.35 |

The 5 images with the highest mean error:

| Image | Image MRE (mm) | Landmarks > 4 mm | Worst landmark |
|---|---|---|---|
| 302 | 2.49 | 1 | Porion (24.19 mm) |
| 393 | 1.99 | 3 | Porion (12.10 mm) |
| 331 | 1.98 | 2 | Soft tissue pogonion (6.93 mm) |
| 376 | 1.87 | 1 | Soft tissue pogonion (6.67 mm) |
| 349 | 1.82 | 2 | Lower incisal incision (6.03 mm) |

Inter-observer (junior vs senior) distance: 1.514 +/- 1.405 mm.

### CephAdoAdu (official validation part, 10 landmarks)

44 of 3000 landmark predictions (1.47 %) exceed 4 mm; 40 of 300 images contain at least one.

Failures (> 4 mm) by landmark, most frequent first:

| # | Landmark | n > 4 mm | % of images | MRE (mm) | Max error (mm) |
|---|---|---|---|---|---|
| 6 | P | 13 | 4.3 | 1.534 | 7.72 |
| 8 | LIA | 10 | 3.3 | 1.394 | 7.22 |
| 10 | Pog | 6 | 2.0 | 1.114 | 9.87 |
| 4 | UIA | 5 | 1.7 | 1.283 | 7.06 |
| 5 | Or | 4 | 1.3 | 1.168 | 12.07 |
| 2 | ANS | 2 | 0.7 | 0.890 | 6.59 |
| 3 | UI | 2 | 0.7 | 0.502 | 6.52 |
| 7 | LI | 1 | 0.3 | 0.598 | 5.14 |
| 9 | Sn | 1 | 0.3 | 0.575 | 5.04 |
| 1 | A | 0 | 0.0 | 1.047 | 3.93 |

The 15 largest individual errors:

| Rank | Image | # | Landmark | Error (mm) | Group / device |
|---|---|---|---|---|---|
| 1 | 4cfb147bcb3c4f44bf50133869e08a67 | 5 | Or | 12.07 | adolescent |
| 2 | 3fddd4ae901f451198ae41553394cf62 | 10 | Pog | 9.87 | adolescent |
| 3 | 6e3c7652b2f34db6a5072d5d7ec116ef | 10 | Pog | 8.08 | adolescent |
| 4 | 09a848d9656348afbb908d433c382009 | 6 | P | 7.72 | adolescent |
| 5 | 4ab47eff1f794f45b011bc6f3812b782 | 8 | LIA | 7.22 | adolescent |
| 6 | 00e416838e72477d9287288698fad079 | 4 | UIA | 7.06 | adolescent |
| 7 | TC807 | 10 | Pog | 6.82 | adult |
| 8 | 0e4665273c4748c28b6f31626db294d8 | 10 | Pog | 6.73 | adolescent |
| 9 | d2293f54d58d4f1689d6739deaed1ef9 | 2 | ANS | 6.59 | adult |
| 10 | 3f42983e87dc42d6965586f6f735a8ed | 10 | Pog | 6.54 | adolescent |
| 11 | 645563ad05de4f3e8bd25983e7c0e8ba | 3 | UI | 6.52 | adolescent |
| 12 | TC1152 | 6 | P | 6.51 | adult |
| 13 | 1f2bd805f205454ab184a7f07618e57a | 8 | LIA | 6.44 | adolescent |
| 14 | 3c89b11adc0e479395278564e0cbef9a | 5 | Or | 6.19 | adolescent |
| 15 | 5e8b639e27364576ae7c094147e1fb7d | 4 | UIA | 6.11 | adolescent |

The 5 images with the highest mean error:

| Image | Image MRE (mm) | Landmarks > 4 mm | Worst landmark | Group / device |
|---|---|---|---|---|
| 6e3c7652b2f34db6a5072d5d7ec116ef | 2.51 | 1 | Pog (8.08 mm) | adolescent |
| 00e416838e72477d9287288698fad079 | 2.47 | 3 | UIA (7.06 mm) | adolescent |
| TC727 | 2.46 | 2 | UIA (5.36 mm) | adult |
| 4cfb147bcb3c4f44bf50133869e08a67 | 2.21 | 1 | Or (12.07 mm) | adolescent |
| 4ab47eff1f794f45b011bc6f3812b782 | 2.16 | 1 | LIA (7.22 mm) | adolescent |

Failures by age group: adolescent 24 of 1500, adult 20 of 1500.

## Rank correlations

| Quantity | n | Spearman rho | p | Pearson r | p |
|---|---|---|---|---|---|
| Aariz: per-landmark model MRE vs junior-senior distance (26 landmarks) | 26 | +0.306 | 0.1279 | +0.387 | 0.05096 |
| Aariz: per-image model MRE vs junior-senior distance (150 images) | 150 | +0.647 | 3.509e-19 | +0.585 | 3.858e-15 |
| Aariz: all landmark instances, model error vs junior-senior distance | 3900 | +0.116 | 3.251e-13 | +0.215 | 4.481e-42 |
| Aariz: per-image model MRE vs pixel size | 150 | +0.122 | 0.1368 | +0.103 | 0.2095 |
| ISBI 2015 Test1: per-landmark model MRE vs junior-senior distance (19 landmarks) | 19 | +0.705 | 0.000744 | +0.473 | 0.04066 |
| ISBI 2015 Test1: per-image model MRE vs junior-senior distance | 150 | +0.348 | 1.309e-05 | +0.445 | 1.125e-08 |
| ISBI 2015 Test2: per-landmark model MRE vs junior-senior distance (19 landmarks) | 19 | +0.837 | 7.946e-06 | +0.769 | 0.0001189 |
| ISBI 2015 Test2: per-image model MRE vs junior-senior distance | 100 | +0.319 | 0.001197 | +0.252 | 0.01143 |
| CephAdoAdu: per-image model MRE vs CeLDA mm-per-pixel factor | 300 | -0.033 | 0.5721 | +0.072 | 0.2126 |

### Aariz - model error and inter-observer distance by landmark

| # | Landmark | Model MRE (mm) | Junior-senior distance (mm) |
|---|---|---|---|
| 1 | A-point | 0.874 | 0.496 |
| 2 | Anterior Nasal Spine | 1.214 | 0.609 |
| 3 | B-point | 0.834 | 0.706 |
| 4 | Menton | 0.486 | 0.823 |
| 5 | Nasion | 0.859 | 0.568 |
| 6 | Orbitale | 1.292 | 1.023 |
| 7 | Pogonion | 0.553 | 0.271 |
| 8 | Posterior Nasal Spine | 1.412 | 0.227 |
| 9 | Pronasale | 0.791 | 0.230 |
| 10 | Ramus | 2.094 | 0.877 |
| 11 | Sella | 0.631 | 0.034 |
| 12 | Articulare | 1.577 | 0.695 |
| 13 | Condylion | 2.000 | 0.807 |
| 14 | Gnathion | 0.414 | 0.452 |
| 15 | Gonion | 1.473 | 1.310 |
| 16 | Porion | 1.552 | 1.408 |
| 17 | Lower 2nd PM Cusp Tip | 1.277 | 0.000 |
| 18 | Lower Incisor Tip | 0.726 | 0.349 |
| 19 | Lower Molar Cusp Tip | 1.058 | 0.040 |
| 20 | Upper 2nd PM Cusp Tip | 1.336 | 0.000 |
| 21 | Upper Incisor Apex | 1.154 | 1.083 |
| 22 | Upper Incisor Tip | 0.577 | 0.349 |
| 23 | Upper Molar Cusp Tip | 1.321 | 0.010 |
| 24 | Lower Incisor Apex | 1.390 | 0.810 |
| 25 | Labrale inferius | 1.097 | 0.228 |
| 26 | Labrale superius | 0.742 | 0.124 |
