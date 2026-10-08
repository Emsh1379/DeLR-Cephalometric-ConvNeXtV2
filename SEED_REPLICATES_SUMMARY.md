# Three-phase seed replicates on ISBI 2015 - complete (2026-10-08)

Both runs finished all three phases (200 + 200 + 128 epochs). The complete recipe was repeated twice with
mixed-precision arithmetic in the backbone: once with the published seed (42) and once with a new seed (2).

## Final results (three phases)

| Model | Seed | Precision | Test1 MRE +/- SD (mm) | Test1 MRE 95% CI | Test1 SDR 2.0 / 2.5 / 3.0 / 4.0 (%) | Test2 MRE +/- SD (mm) | Test2 MRE 95% CI | Test2 SDR 2.0 / 2.5 / 3.0 / 4.0 (%) |
|---|---|---|---|---|---|---|---|---|
| Published model | 42 | full | 1.124 +/- 1.076 | 1.082-1.169 | 87.12 / 92.53 / 96.11 / 98.39 | 1.463 +/- 1.385 | 1.419-1.508 | 74.74 / 83.47 / 88.84 / 94.63 |
| Replicate | 42 | mixed (backbone) | 1.102 +/- 0.942 | 1.063-1.145 | 87.40 / 92.21 / 95.68 / 98.35 | 1.426 +/- 1.280 | 1.386-1.467 | 76.74 / 83.21 / 88.89 / 95.00 |
| Replicate | 2 | mixed (backbone) | 1.122 +/- 0.943 | 1.081-1.165 | 87.05 / 92.53 / 95.61 / 98.32 | 1.462 +/- 1.467 | 1.406-1.532 | 75.42 / 83.00 / 88.74 / 94.95 |

Across the three complete runs: Test1 MRE 1.102-1.124 mm and SDR at 2 mm 87.05-87.40%; Test2 MRE 1.426-1.463 mm and
SDR at 2 mm 74.74-76.74%. 95% CIs: percentile bootstrap, 10,000 image-level resamples.

## By phase

| Seed | Phase | Epochs | Best validation MRE (mm) | Test1 MRE (mm) / SDR 2 mm (%) | Test2 MRE (mm) / SDR 2 mm (%) |
|---|---|---|---|---|---|
| 42 | 1 | 200 | 1.171 | 1.171 / 85.75 | 1.532 / 75.11 |
| 42 | 2 | 200 | 1.102 | 1.102 / 87.40 | 1.426 / 76.74 |
| 42 | 3 | 128 | 1.102 (no improvement; best phase-3 epoch 1.121) | 1.102 / 87.40 | 1.426 / 76.74 |
| 2 | 1 | 200 | 1.171 | 1.171 / 85.47 | 1.556 / 74.47 |
| 2 | 2 | 200 | 1.122 | 1.122 / 86.91 | 1.482 / 75.63 |
| 2 | 3 | 128 | 1.122 | 1.122 / 87.05 | 1.462 / 75.42 |

For seed 42 the third phase never beat the phase-2 checkpoint, so its final model is the phase-2 one (the predictions
are identical). For seed 2 the third phase produced a marginally better checkpoint. This matches the manuscript's
remark about diminishing returns from further fine-tuning.

## Paired differences in per-image error (same images)

| Comparison | Test1: mean difference in mm (95% CI) | Test2: mean difference in mm (95% CI) |
|---|---|---|
| Published - seed 42 replicate | +0.022 (+0.005 to +0.042) | +0.037 (+0.014 to +0.066) |
| Published - seed 2 replicate | +0.002 (-0.016 to +0.024) | +0.001 (-0.055 to +0.036) |
| Seed 42 replicate - seed 2 replicate | -0.019 (-0.033 to -0.007) | -0.037 (-0.094 to +0.001) |

## What this shows

- **The published three-phase result is reproducible.** The new seed gives 1.122 mm / 1.462 mm against the published
  1.124 mm / 1.463 mm: differences of 0.002 and 0.001 mm.
- **Run-to-run spread of the complete recipe is about 0.02 mm on Test1 and 0.04 mm on Test2**, i.e. well inside the
  bootstrap confidence intervals of each run.
- **The same-seed replicate is not identical to the published model** (1.102 vs 1.124 mm). It differs in precision
  mode and hardware, and it was resumed from saved state several times; it is slightly better, not worse, so mixed
  precision did not harm accuracy.
- **The unstable phase-1 run with seed 1 (1.360 mm) was not typical.** Phase 1 gave 1.171 mm for both seeds here and
  1.188 mm originally. Whether phases 2 and 3 would have repaired the seed-1 run was not tested.

## Practical notes on how the runs were made

- Mixed precision (16-bit) in the convolutional backbone only; transformer encoders, coordinate heads, losses and
  metrics in 32-bit. About 63 s per epoch instead of about 110 s.
- The Kaggle sessions ended early four times. Training state was saved every epoch and uploaded to Hugging Face every
  30 minutes, and each run continued from its last completed epoch on a new session (phase 1 at epochs 53/55, phase 2
  at epochs 138/158). A resumed run is not bit-identical to an uninterrupted one.

## Proposed additions for the manuscript (not applied - you edit these files)

Table S7 currently lists first-phase runs only. Suggested: retitle it "Additional training runs of the full model on
ISBI 2015" and add these rows (and drop the footnote sentence "The final ISBI 2015 model (three phases, 528 epochs) was
not repeated with another seed"):

| Run | Epochs | Seed | Test1 MRE (mm) | Test1 SDR 2.0 / 2.5 / 3.0 / 4.0 (%) | Test2 MRE (mm) | Test2 SDR 2.0 / 2.5 / 3.0 / 4.0 (%) |
|---|---|---|---|---|---|---|
| Final model (three phases), as reported | 528 | 42 | 1.124 | 87.12 / 92.53 / 96.11 / 98.39 | 1.463 | 74.74 / 83.47 / 88.84 / 94.63 |
| Three-phase replicate* | 528 | 42 | 1.102 | 87.40 / 92.21 / 95.68 / 98.35 | 1.426 | 76.74 / 83.21 / 88.89 / 95.00 |
| Three-phase replicate* | 528 | 2 | 1.122 | 87.05 / 92.53 / 95.61 / 98.32 | 1.462 | 75.42 / 83.00 / 88.74 / 94.95 |
| Phase 1 of the replicate* | 200 | 42 | 1.171 | 85.75 / 91.72 / 95.05 / 97.93 | 1.532 | 75.11 / 82.47 / 88.21 / 94.95 |
| Phase 1 of the replicate* | 200 | 2 | 1.171 | 85.47 / 91.75 / 95.51 / 98.14 | 1.556 | 74.47 / 82.47 / 88.05 / 94.84 |

*Trained with mixed-precision arithmetic in the backbone.

Replacement for the limitations sentences (from "Fifth, each configuration was trained once ..." up to "... should not
be interpreted."):

> Fifth, the Aariz and CephAdoAdu models and the ablation variants were each trained once, so their confidence
> intervals reflect sampling of the evaluation images only, and the ablation variants were trained for fewer epochs
> than the final models. For ISBI 2015, the complete three-phase training was repeated twice, with the original and
> with a different random seed (Table S7): the Test1 MRE was 1.102 and 1.122 mm, against 1.124 mm for the reported
> model, and the Test2 MRE 1.426 and 1.462 mm against 1.463 mm, so run-to-run variation of the full recipe was about
> 0.02-0.04 mm. Single 200-epoch runs varied more: one repetition of the first phase became temporarily unstable near
> the peak learning rate and ended at 1.360 mm instead of 1.17-1.19 mm, so differences of a few hundredths of a
> millimetre between ablation variants, such as that between the full model and the variant without the heatmap head,
> should not be interpreted.

If you adopt this, add one clause to section 2.5 or to the Table S7 footnote: "the replicate runs used mixed-precision
arithmetic in the convolutional backbone to shorten training".

## Where everything is

- Hugging Face, `main` branch of `emad2001/DeLR-Cephalometric-ConvNeXtV2`: `checkpoints/ISBI2015_seed_replicates/`
  (logs, predictions for every phase, the two final checkpoints, `train_fast.py`, `job3p.sh`).
- Local: `results/training_runs_3phase/` (logs, metrics, predictions, `final_metrics_with_ci.json`).
- The work branch `seedrep-wip-27` on Hugging Face holds the last rolling training state and is no longer needed.
- The earlier interrupted attempt is in `results/training_runs_3phase_interrupted_2026-10-06/`.
