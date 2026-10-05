# Predicted uncertainty and failure cases (Aariz test set)

Inference re-run with `checkpoints/Aariz_26/best_model.pt`. Largest difference between re-run and saved predictions: 0.035 original pixels; MRE re-run 1.1051 mm, saved predictions 1.1051 mm. All statistics below use the errors of the saved prediction files.

Uncertainty = exp(predicted log-scale) of the last refinement update, converted from network pixels to millimetres with the image's pixel size and resize factors.

## Correlation over all landmark predictions

| Uncertainty unit | n | Spearman rho | Pearson r |
|---|---:|---:|---:|
| millimetres | 3900 | 0.605 | 0.429 |
| network pixels (raw output) | 3900 | 0.261 | 0.306 |

Uncertainty (mm): mean 0.208, median 0.180. Error (mm): mean 1.105, median 0.732. Ratio mean error / mean uncertainty = 5.31; median of per-prediction ratio = 4.11; least-squares slope through origin = 4.49.

## Deciles of predicted uncertainty (mm)

| Decile | n | Uncertainty range (mm) | Mean uncertainty (mm) | MRE (mm) | SDR 2 mm (%) | Error / uncertainty |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 390 | 0.038-0.085 | 0.069 | 0.330 | 100.0 | 4.77 |
| 2 | 390 | 0.085-0.118 | 0.102 | 0.462 | 99.7 | 4.52 |
| 3 | 390 | 0.118-0.141 | 0.130 | 0.526 | 99.2 | 4.04 |
| 4 | 390 | 0.141-0.160 | 0.150 | 0.711 | 96.9 | 4.73 |
| 5 | 390 | 0.160-0.180 | 0.170 | 0.904 | 93.8 | 5.32 |
| 6 | 390 | 0.180-0.202 | 0.191 | 1.148 | 88.2 | 6.02 |
| 7 | 390 | 0.202-0.229 | 0.215 | 1.370 | 79.2 | 6.37 |
| 8 | 390 | 0.229-0.266 | 0.247 | 1.469 | 81.5 | 5.96 |
| 9 | 390 | 0.266-0.330 | 0.294 | 1.614 | 74.6 | 5.49 |
| 10 | 390 | 0.330-4.809 | 0.515 | 2.517 | 55.1 | 4.89 |

## The 20 largest errors with their predicted uncertainty

| Rank | Image | Device | Landmark | Error (mm) | Predicted uncertainty (mm) | Uncertainty percentile |
|---:|---|---|---|---:|---:|---:|
| 1 | cks2ip8g62b7b0yuffrlb7xa1 | ART Plus | Articulare | 20.29 | 0.759 | 99 |
| 2 | cl5lg05un01hm074k3dwy9l5q | ProMax with ProTouch | Labrale superius | 16.63 | 0.293 | 85 |
| 3 | cl5lg05un01hm074k3dwy9l5q | ProMax with ProTouch | Labrale inferius | 16.61 | 0.263 | 79 |
| 4 | cks2ip8g62b7b0yuffrlb7xa1 | ART Plus | Condylion | 16.52 | 0.600 | 98 |
| 5 | cks2ip8g62b7b0yuffrlb7xa1 | ART Plus | Orbitale | 14.48 | 1.097 | 100 |
| 6 | cks2ip8g02asq0yuf59owfht0 | ART Plus | Lower 2nd PM Cusp Tip | 12.30 | 0.238 | 73 |
| 7 | cl777aklx03ua076whaie06kb | Rotograph EVO | Posterior Nasal Spine | 11.58 | 0.719 | 99 |
| 8 | cks2ip8fq29zp0yuf7cxd3bz2 | ART Plus | Lower 2nd PM Cusp Tip | 11.53 | 0.407 | 95 |
| 9 | cks2ip8g02asq0yuf59owfht0 | ART Plus | Upper Molar Cusp Tip | 11.11 | 0.291 | 85 |
| 10 | cks2ip8g62b7b0yuffrlb7xa1 | ART Plus | Ramus | 10.47 | 1.939 | 100 |
| 11 | cks2ip8g02asq0yuf59owfht0 | ART Plus | Lower Molar Cusp Tip | 10.08 | 0.283 | 83 |
| 12 | cl777aklx03ua076whaie06kb | Rotograph EVO | Ramus | 9.71 | 0.611 | 98 |
| 13 | cks2ip8fz2aq80yuf2pz61jh3 | ART Plus | Articulare | 9.08 | 0.323 | 89 |
| 14 | cl5lg05ul01gq074kb4xm8mml | ProMax with ProTouch | Pronasale | 8.85 | 2.551 | 100 |
| 15 | cl7ww1jcp9cfk084fgkua8l9o | Rotograph EVO | Ramus | 8.79 | 0.328 | 90 |
| 16 | cks2ip8g42b0s0yuf23y8917i | ART Plus | Articulare | 8.42 | 0.257 | 78 |
| 17 | cks2ip8fq29zz0yuf15jsfhlu | ART Plus | Ramus | 8.29 | 0.510 | 97 |
| 18 | cks2ip8g32av80yuf73500e1b | ART Plus | Labrale inferius | 8.06 | 4.809 | 100 |
| 19 | cks2ip8fq29zp0yuf7cxd3bz2 | ART Plus | Upper Molar Cusp Tip | 8.00 | 0.387 | 94 |
| 20 | cl5lg05ul01gm074k2rztekih | ProMax with ProTouch | Upper Molar Cusp Tip | 7.90 | 0.767 | 99 |

Of the 127 errors above 4 mm, 69 are in the top uncertainty decile and 123 are above the median uncertainty; median uncertainty of these failures 0.358 mm versus 0.177 mm for the other predictions.

## Overlays

- `overlay_aariz_1_cks2ip8g62b7b0yuffrlb7xa1.jpg` (ART Plus): 12 Articulare 20.3 mm (u 0.76), 13 Condylion 16.5 mm (u 0.60), 6 Orbitale 14.5 mm (u 1.10), 10 Ramus 10.5 mm (u 1.94)
- `overlay_aariz_2_cl5lg05un01hm074k3dwy9l5q.jpg` (ProMax with ProTouch): 26 Labrale superius 16.6 mm (u 0.29), 25 Labrale inferius 16.6 mm (u 0.26)
- `overlay_aariz_3_cks2ip8g02asq0yuf59owfht0.jpg` (ART Plus): 17 Lower 2nd PM Cusp Tip 12.3 mm (u 0.24), 23 Upper Molar Cusp Tip 11.1 mm (u 0.29), 19 Lower Molar Cusp Tip 10.1 mm (u 0.28), 20 Upper 2nd PM Cusp Tip 7.1 mm (u 0.27)
- `overlay_aariz_4_cl777aklx03ua076whaie06kb.jpg` (Rotograph EVO): 8 Posterior Nasal Spine 11.6 mm (u 0.72), 10 Ramus 9.7 mm (u 0.61), 12 Articulare 6.2 mm (u 0.52), 13 Condylion 5.7 mm (u 0.58), 16 Porion 4.9 mm (u 0.47), 21 Upper Incisor Apex 4.4 mm (u 0.39), 15 Gonion 4.4 mm (u 0.81)
- `overlay_aariz_5_cks2ip8fq29zp0yuf7cxd3bz2.jpg` (ART Plus): 17 Lower 2nd PM Cusp Tip 11.5 mm (u 0.41), 23 Upper Molar Cusp Tip 8.0 mm (u 0.39), 20 Upper 2nd PM Cusp Tip 7.1 mm (u 0.43), 19 Lower Molar Cusp Tip 6.1 mm (u 0.33), 10 Ramus 5.7 mm (u 0.35), 12 Articulare 5.1 mm (u 0.40)
- `overlay_aariz_6_cks2ip8fz2aq80yuf2pz61jh3.jpg` (ART Plus): 12 Articulare 9.1 mm (u 0.32), 19 Lower Molar Cusp Tip 5.3 mm (u 0.20)

Green circle: reference; filled dot: prediction (red and numbered if error > 4 mm); yellow line joins the two.
