| Model | Test1 MRE (95% CI) | Test1 SDR 2/2.5/3/4 mm (%) | Test2 MRE (95% CI) | Test2 SDR 2/2.5/3/4 mm (%) |
|---|---|---|---|---|
| ConvNeXt V2-Tiny, Laplace loss (reference) | 1.180 (1.135-1.229) | 85.79 / 91.72 / 95.26 / 98.14 | 1.571 (1.469-1.700) | 74.16 / 82.16 / 87.84 / 95.16 |
| ResNet-34 backbone | 1.143 (1.099-1.192) | 86.46 / 90.98 / 94.91 / 98.21 | 1.478 (1.430-1.528) | 74.84 / 82.68 / 88.26 / 94.53 |
| Fixed-scale L1 loss, 8 px | 31.369 (30.957-31.801) | 0.39 / 0.49 / 0.74 / 1.51 | 30.755 (30.305-31.227) | 0.21 / 0.42 / 0.63 / 1.63 |
| Fixed-scale L1 loss, 32 px | 40.823 (40.339-41.302) | 0.35 / 0.53 / 0.70 / 1.19 | 40.312 (39.806-40.819) | 0.21 / 0.26 / 0.32 / 0.79 |
| Fixed-scale L1 loss, 32 px, peak lr 5e-5 | 40.313 (39.863-40.771) | 0.00 / 0.00 / 0.00 / 0.07 | 39.955 (39.481-40.414) | 0.00 / 0.00 / 0.00 / 0.00 |

Paired difference in per-image MRE versus 'ConvNeXt V2-Tiny, Laplace loss (reference)' (positive = worse than the reference):

| Model | Test1 difference, mm (95% CI) | Test2 difference, mm (95% CI) |
|---|---|---|
| ResNet-34 backbone | -0.037 (-0.071 to -0.007) | -0.094 (-0.214 to -0.009) |
| Fixed-scale L1 loss, 8 px | +30.189 (+29.786 to +30.601) | +29.184 (+28.765 to +29.605) |
| Fixed-scale L1 loss, 32 px | +39.643 (+39.160 to +40.119) | +38.741 (+38.229 to +39.263) |
| Fixed-scale L1 loss, 32 px, peak lr 5e-5 | +39.133 (+38.681 to +39.582) | +38.383 (+37.917 to +38.860) |
