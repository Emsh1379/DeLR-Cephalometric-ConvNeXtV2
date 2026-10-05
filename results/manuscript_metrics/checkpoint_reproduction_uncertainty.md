# Released checkpoints: reproduction of the prediction files and predicted uncertainty

Inference re-run with the released weights. 'Max diff' is the largest coordinate difference (original pixels) between the re-run and the saved prediction file. Uncertainty = exp(predicted log-scale) of the last refinement update, converted to millimetres. Correlations and deciles are over all landmark predictions and use the errors of the saved prediction files.

| Evaluation set | Images | Max diff (px) | MRE re-run (mm) | MRE saved (mm) | Spearman rho | Pearson r | Most confident decile: MRE (SDR 2 mm) | Least confident decile: MRE (SDR 2 mm) | Mean error / mean uncertainty | Errors > 4 mm with above-median uncertainty | Errors > 4 mm in least confident decile |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Aariz test | 150 | 0.035 | 1.1051 | 1.1051 | 0.605 | 0.429 | 0.33 (100.0%) | 2.52 (55.1%) | 5.31 | 123/127 | 69/127 |
| ISBI 2015 Test1 | 150 | 0.001 | 1.1238 | 1.1238 | 0.136 | 0.550 | 0.62 (99.6%) | 1.52 (81.4%) | 4.55 | 29/46 | 12/46 |
| ISBI 2015 Test2 | 100 | 0.000 | 1.4632 | 1.4632 | 0.192 | 0.391 | 0.75 (94.2%) | 2.54 (56.8%) | 6.43 | 90/102 | 47/102 |
| CephAdoAdu validation part | 300 | 0.000 | 1.0105 | 1.0105 | 0.387 | 0.383 | 0.46 (99.7%) | 1.63 (73.7%) | 5.09 | 38/44 | 17/44 |

## Deciles of predicted uncertainty

| Evaluation set | Decile | Mean uncertainty (mm) | MRE (mm) | SDR 2 mm (%) | Error / uncertainty |
|---|---|---|---|---|---|
| Aariz test | 1 | 0.069 | 0.330 | 100.0 | 4.77 |
| Aariz test | 2 | 0.102 | 0.462 | 99.7 | 4.52 |
| Aariz test | 3 | 0.130 | 0.526 | 99.2 | 4.04 |
| Aariz test | 4 | 0.150 | 0.711 | 96.9 | 4.73 |
| Aariz test | 5 | 0.170 | 0.904 | 93.8 | 5.32 |
| Aariz test | 6 | 0.191 | 1.148 | 88.2 | 6.02 |
| Aariz test | 7 | 0.215 | 1.370 | 79.2 | 6.37 |
| Aariz test | 8 | 0.247 | 1.469 | 81.5 | 5.96 |
| Aariz test | 9 | 0.294 | 1.614 | 74.6 | 5.49 |
| Aariz test | 10 | 0.515 | 2.517 | 55.1 | 4.89 |
| ISBI 2015 Test1 | 1 | 0.134 | 0.616 | 99.6 | 4.62 |
| ISBI 2015 Test1 | 2 | 0.156 | 1.159 | 87.4 | 7.42 |
| ISBI 2015 Test1 | 3 | 0.182 | 1.303 | 80.4 | 7.15 |
| ISBI 2015 Test1 | 4 | 0.200 | 1.137 | 85.6 | 5.68 |
| ISBI 2015 Test1 | 5 | 0.211 | 1.102 | 87.7 | 5.23 |
| ISBI 2015 Test1 | 6 | 0.223 | 0.925 | 93.0 | 4.15 |
| ISBI 2015 Test1 | 7 | 0.237 | 1.051 | 87.4 | 4.43 |
| ISBI 2015 Test1 | 8 | 0.255 | 1.160 | 87.4 | 4.55 |
| ISBI 2015 Test1 | 9 | 0.278 | 1.266 | 81.4 | 4.56 |
| ISBI 2015 Test1 | 10 | 0.595 | 1.520 | 81.4 | 2.56 |
| ISBI 2015 Test2 | 1 | 0.133 | 0.751 | 94.2 | 5.66 |
| ISBI 2015 Test2 | 2 | 0.156 | 1.504 | 70.0 | 9.64 |
| ISBI 2015 Test2 | 3 | 0.181 | 1.482 | 74.2 | 8.19 |
| ISBI 2015 Test2 | 4 | 0.198 | 1.339 | 75.3 | 6.76 |
| ISBI 2015 Test2 | 5 | 0.209 | 1.207 | 76.8 | 5.79 |
| ISBI 2015 Test2 | 6 | 0.219 | 1.278 | 76.3 | 5.83 |
| ISBI 2015 Test2 | 7 | 0.232 | 1.203 | 83.2 | 5.19 |
| ISBI 2015 Test2 | 8 | 0.248 | 1.430 | 75.3 | 5.76 |
| ISBI 2015 Test2 | 9 | 0.268 | 1.900 | 65.3 | 7.09 |
| ISBI 2015 Test2 | 10 | 0.431 | 2.537 | 56.8 | 5.88 |
| CephAdoAdu validation part | 1 | 0.130 | 0.462 | 99.7 | 3.55 |
| CephAdoAdu validation part | 2 | 0.147 | 0.642 | 97.0 | 4.35 |
| CephAdoAdu validation part | 3 | 0.159 | 0.792 | 94.7 | 4.99 |
| CephAdoAdu validation part | 4 | 0.169 | 0.855 | 94.0 | 5.05 |
| CephAdoAdu validation part | 5 | 0.181 | 1.053 | 89.3 | 5.82 |
| CephAdoAdu validation part | 6 | 0.193 | 0.976 | 91.7 | 5.07 |
| CephAdoAdu validation part | 7 | 0.206 | 1.161 | 88.3 | 5.65 |
| CephAdoAdu validation part | 8 | 0.221 | 1.277 | 85.7 | 5.78 |
| CephAdoAdu validation part | 9 | 0.244 | 1.259 | 82.7 | 5.16 |
| CephAdoAdu validation part | 10 | 0.337 | 1.629 | 73.7 | 4.83 |
