# ISBI 2015: clinical measurements and classification (official challenge formulas)

## Self-check (all 400 images)

Official formulas (ported from EvaluationCode/angle.py, including its integer truncation of coordinates) applied to each clinician's own landmarks, compared with the class labels stored in that clinician's file. 100% means the port is exact.

| Measurement | Junior landmarks vs junior labels (%) | Senior landmarks vs senior labels (%) | Junior vs senior labels (%) |
|---|---|---|---|
| ANB | 100.0 | 100.0 | 78.0 |
| SNB | 100.0 | 100.0 | 82.0 |
| SNA | 100.0 | 100.0 | 67.8 |
| ODI | 100.0 | 100.0 | 77.0 |
| APDI | 100.0 | 100.0 | 82.2 |
| FHI | 100.0 | 100.0 | 84.2 |
| FMA | 100.0 | 100.0 | 82.0 |
| MW | 100.0 | 100.0 | 87.5 |

## Released model: Test1 (150 images)

Reference = measurement from the mean of the two clinicians' landmarks (the standard used for the landmark errors). Clinician labels = the class labels stored in each clinician's file.

| Measurement | Mean absolute difference (predicted vs reference) | Class agreement with reference % (95% CI) | Cohen's kappa | Agreement with senior labels (%) | Agreement with junior labels (%) | Senior vs junior labels (%) | Agreement with senior where clinicians agree (%) |
|---|---|---|---|---|---|---|---|
| ANB | 1.18° | 80.0 (73.3-86.0) | 0.67 | 76.0 | 80.0 | 73.3 | 89.1 (n=110) |
| SNB | 1.38° | 89.3 (84.0-94.0) | 0.80 | 83.3 | 82.0 | 78.0 | 94.0 (n=117) |
| SNA | 1.90° | 72.7 (65.3-79.3) | 0.57 | 71.3 | 78.7 | 66.7 | 89.0 (n=100) |
| ODI | 2.43° | 86.0 (80.0-91.3) | 0.76 | 79.3 | 88.7 | 80.0 | 92.5 (n=120) |
| APDI | 2.41° | 84.0 (78.0-90.0) | 0.75 | 82.7 | 82.0 | 82.7 | 89.5 (n=124) |
| FHI | 0.014 | 85.3 (79.3-90.7) | 0.72 | 82.0 | 86.0 | 80.0 | 92.5 (n=120) |
| FMA | 1.57° | 85.3 (79.3-90.7) | 0.73 | 82.7 | 84.0 | 81.3 | 91.8 (n=122) |
| MW | 0.94 mm | 88.7 (83.3-93.3) | 0.83 | 83.3 | 88.7 | 89.3 | 91.0 (n=134) |

Mean over the 8 measurements: agreement with reference 83.9%, with senior labels 80.1%; senior vs junior clinician 78.9%.

## Released model: Test2 (100 images)

Reference = measurement from the mean of the two clinicians' landmarks (the standard used for the landmark errors). Clinician labels = the class labels stored in each clinician's file.

| Measurement | Mean absolute difference (predicted vs reference) | Class agreement with reference % (95% CI) | Cohen's kappa | Agreement with senior labels (%) | Agreement with junior labels (%) | Senior vs junior labels (%) | Agreement with senior where clinicians agree (%) |
|---|---|---|---|---|---|---|---|
| ANB | 0.86° | 83.0 (75.0-90.0) | 0.74 | 84.0 | 83.0 | 90.0 | 87.8 (n=90) |
| SNB | 1.10° | 88.0 (81.0-94.0) | 0.78 | 84.0 | 83.0 | 82.0 | 91.5 (n=82) |
| SNA | 1.18° | 75.0 (66.0-83.0) | 0.60 | 77.0 | 75.0 | 72.0 | 86.1 (n=72) |
| ODI | 3.16° | 76.0 (67.0-84.0) | 0.59 | 68.0 | 83.0 | 69.0 | 87.0 (n=69) |
| APDI | 1.84° | 89.0 (83.0-95.0) | 0.83 | 85.0 | 85.0 | 88.0 | 89.8 (n=88) |
| FHI | 0.009 | 91.0 (85.0-96.0) | 0.83 | 93.0 | 88.0 | 91.0 | 94.5 (n=91) |
| FMA | 1.01° | 89.0 (83.0-95.0) | 0.77 | 86.0 | 88.0 | 79.0 | 97.5 (n=79) |
| MW | 0.79 mm | 85.0 (78.0-92.0) | 0.77 | 84.0 | 84.0 | 90.0 | 87.8 (n=90) |

Mean over the 8 measurements: agreement with reference 84.5%, with senior labels 82.6%; senior vs junior clinician 82.6%.

## Seed replicate 42: Test1 (150 images)

Reference = measurement from the mean of the two clinicians' landmarks (the standard used for the landmark errors). Clinician labels = the class labels stored in each clinician's file.

| Measurement | Mean absolute difference (predicted vs reference) | Class agreement with reference % (95% CI) | Cohen's kappa | Agreement with senior labels (%) | Agreement with junior labels (%) | Senior vs junior labels (%) | Agreement with senior where clinicians agree (%) |
|---|---|---|---|---|---|---|---|
| ANB | 1.21° | 75.3 (68.7-82.0) | 0.59 | 72.0 | 81.3 | 73.3 | 87.3 (n=110) |
| SNB | 1.23° | 90.7 (86.0-94.7) | 0.82 | 84.7 | 84.7 | 78.0 | 96.6 (n=117) |
| SNA | 1.83° | 69.3 (62.0-76.7) | 0.52 | 68.0 | 74.7 | 66.7 | 84.0 (n=100) |
| ODI | 2.46° | 88.7 (83.3-93.3) | 0.81 | 80.7 | 90.0 | 80.0 | 94.2 (n=120) |
| APDI | 2.34° | 85.3 (79.3-90.7) | 0.77 | 84.0 | 82.0 | 82.7 | 90.3 (n=124) |
| FHI | 0.014 | 84.0 (78.0-89.3) | 0.69 | 79.3 | 87.3 | 80.0 | 91.7 (n=120) |
| FMA | 1.48° | 85.3 (79.3-90.7) | 0.72 | 80.7 | 86.0 | 81.3 | 91.0 (n=122) |
| MW | 0.94 mm | 90.7 (86.0-94.7) | 0.86 | 86.7 | 89.3 | 89.3 | 93.3 (n=134) |

Mean over the 8 measurements: agreement with reference 83.7%, with senior labels 79.5%; senior vs junior clinician 78.9%.

## Seed replicate 42: Test2 (100 images)

Reference = measurement from the mean of the two clinicians' landmarks (the standard used for the landmark errors). Clinician labels = the class labels stored in each clinician's file.

| Measurement | Mean absolute difference (predicted vs reference) | Class agreement with reference % (95% CI) | Cohen's kappa | Agreement with senior labels (%) | Agreement with junior labels (%) | Senior vs junior labels (%) | Agreement with senior where clinicians agree (%) |
|---|---|---|---|---|---|---|---|
| ANB | 0.96° | 79.0 (71.0-87.0) | 0.67 | 80.0 | 81.0 | 90.0 | 84.4 (n=90) |
| SNB | 1.05° | 90.0 (84.0-96.0) | 0.82 | 86.0 | 85.0 | 82.0 | 93.9 (n=82) |
| SNA | 1.23° | 74.0 (65.0-82.0) | 0.59 | 74.0 | 74.0 | 72.0 | 83.3 (n=72) |
| ODI | 3.15° | 75.0 (66.0-83.0) | 0.57 | 67.0 | 80.0 | 69.0 | 84.1 (n=69) |
| APDI | 1.71° | 90.0 (84.0-95.0) | 0.85 | 86.0 | 84.0 | 88.0 | 89.8 (n=88) |
| FHI | 0.009 | 91.0 (85.0-96.0) | 0.83 | 93.0 | 88.0 | 91.0 | 94.5 (n=91) |
| FMA | 0.98° | 90.0 (84.0-95.0) | 0.79 | 85.0 | 89.0 | 79.0 | 97.5 (n=79) |
| MW | 0.80 mm | 84.0 (76.0-91.0) | 0.76 | 83.0 | 85.0 | 90.0 | 87.8 (n=90) |

Mean over the 8 measurements: agreement with reference 84.1%, with senior labels 81.8%; senior vs junior clinician 82.6%.

## Seed replicate 2: Test1 (150 images)

Reference = measurement from the mean of the two clinicians' landmarks (the standard used for the landmark errors). Clinician labels = the class labels stored in each clinician's file.

| Measurement | Mean absolute difference (predicted vs reference) | Class agreement with reference % (95% CI) | Cohen's kappa | Agreement with senior labels (%) | Agreement with junior labels (%) | Senior vs junior labels (%) | Agreement with senior where clinicians agree (%) |
|---|---|---|---|---|---|---|---|
| ANB | 1.15° | 80.7 (74.0-86.7) | 0.68 | 76.7 | 82.0 | 73.3 | 90.9 (n=110) |
| SNB | 1.20° | 91.3 (86.7-95.3) | 0.84 | 84.7 | 84.0 | 78.0 | 96.6 (n=117) |
| SNA | 1.75° | 70.0 (62.7-77.3) | 0.53 | 68.7 | 75.3 | 66.7 | 85.0 (n=100) |
| ODI | 2.34° | 88.0 (82.7-92.7) | 0.79 | 81.3 | 88.0 | 80.0 | 93.3 (n=120) |
| APDI | 2.17° | 87.3 (82.0-92.7) | 0.80 | 82.0 | 84.0 | 82.7 | 90.3 (n=124) |
| FHI | 0.014 | 86.0 (80.0-91.3) | 0.73 | 82.7 | 86.7 | 80.0 | 93.3 (n=120) |
| FMA | 1.46° | 81.3 (75.3-87.3) | 0.66 | 82.7 | 83.3 | 81.3 | 91.0 (n=122) |
| MW | 0.82 mm | 90.7 (86.0-95.3) | 0.86 | 85.3 | 90.7 | 89.3 | 93.3 (n=134) |

Mean over the 8 measurements: agreement with reference 84.4%, with senior labels 80.5%; senior vs junior clinician 78.9%.

## Seed replicate 2: Test2 (100 images)

Reference = measurement from the mean of the two clinicians' landmarks (the standard used for the landmark errors). Clinician labels = the class labels stored in each clinician's file.

| Measurement | Mean absolute difference (predicted vs reference) | Class agreement with reference % (95% CI) | Cohen's kappa | Agreement with senior labels (%) | Agreement with junior labels (%) | Senior vs junior labels (%) | Agreement with senior where clinicians agree (%) |
|---|---|---|---|---|---|---|---|
| ANB | 0.86° | 81.0 (73.0-89.0) | 0.70 | 80.0 | 81.0 | 90.0 | 84.4 (n=90) |
| SNB | 1.11° | 91.0 (85.0-96.0) | 0.83 | 87.0 | 86.0 | 82.0 | 95.1 (n=82) |
| SNA | 1.22° | 75.0 (67.0-83.0) | 0.61 | 75.0 | 75.0 | 72.0 | 84.7 (n=72) |
| ODI | 3.13° | 75.0 (66.0-83.0) | 0.57 | 67.0 | 80.0 | 69.0 | 84.1 (n=69) |
| APDI | 1.63° | 93.0 (88.0-98.0) | 0.89 | 87.0 | 89.0 | 88.0 | 93.2 (n=88) |
| FHI | 0.010 | 92.0 (86.0-97.0) | 0.84 | 94.0 | 89.0 | 91.0 | 95.6 (n=91) |
| FMA | 1.06° | 91.0 (85.0-96.0) | 0.81 | 88.0 | 86.0 | 79.0 | 97.5 (n=79) |
| MW | 0.81 mm | 83.0 (76.0-90.0) | 0.74 | 82.0 | 82.0 | 90.0 | 85.6 (n=90) |

Mean over the 8 measurements: agreement with reference 85.1%, with senior labels 82.5%; senior vs junior clinician 82.6%.

Class definitions: ANB: 1 = skeletal class I (3.2-5.7), 2 = class II (>5.7), 3 = class III (<3.2); SNB: 1 = normal mandible (74.6-78.7), 2 = retrognathic (<74.6), 3 = prognathic (>78.7); SNA: 1 = normal maxilla (79.4-83.2), 2 = prognathic (>83.2), 3 = retrognathic (<79.4); ODI: 1 = normal (68.4-80.5), 2 = deep bite tendency (>80.5), 3 = open bite tendency (<68.4); APDI: 1 = normal (77.6-85.2), 2 = class II tendency (<77.6), 3 = class III tendency (>85.2); FHI: 1 = normal (0.65-0.75), 2 = short face (>0.75), 3 = long face (<0.65); FMA: 1 = normal (26.8-31.4), 2 = high angle (>31.4), 3 = low angle (<26.8); MW: 1 = normal (2-4.5 mm), 2 = edge to edge (0 mm), 3 = anterior open bite / crossbite (<2 mm), 4 = large overjet (>4.5 mm)
