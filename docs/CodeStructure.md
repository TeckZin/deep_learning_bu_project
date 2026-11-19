# Code structure (Project A)

Start here. This page shows how the code fits together and which file to run or edit.
Design rationale and decisions: [ModularRefactorPlan.md](ModularRefactorPlan.md).
Pipeline stages it maps to: [ModulesAndDesignDecisions.md](ModulesAndDesignDecisions.md).
Team space on the SCC (shared runs, logs, env, permissions): [SCC.md](SCC.md).
How to change each pipeline stage (file, config knob, rules): [ChangingTheArchitecture.md](ChangingTheArchitecture.md).
All paths are relative to `project_a/cli/`.

## Entry points

| I want to... | Run | Writes |
|---|---|---|
| Make the CV folds (once, team-wide) | `python splits.py make` | `splits/folds.csv` (commit it) |
| Train H0 heads for a fold | `python train_h0.py --val-fold F` | `weights/h0/<encoder>_<scheme>/foldF/h0_*.safetensors`, `plan.json` |
| Train the refiner on a fold | `python train.py --config experiments/<exp>.json --val-fold F` | `weights/<run>/foldF/refiner*.safetensors`, `run.json` (jobs: run = `<login>-<exp>`) |
| Both, as SCC jobs | `bash jobs/cv.sh <exp> [FOLDS]` | same, logs in `logs/` |
| Build the leaderboard zip | `python export_submission.py --refiner CKPT --zip --check IMAGES` | `dist/submission/`, `dist/submission.zip` |
| Run the leaderboard inference | `python infer.py --input IMAGES --output OUT` | `OUT/k<K>/<id>.png`, `OUT_intermediate/` |
| Fold manifest for staff tools | `python splits.py manifest --fold F --out val_foldF --stress` | `val_foldF/`, `val_foldF_stress/` |
| Check everything still works | `python -m unittest discover -s tests` | (35 tests) |
| See presets and their sizes | `python -m submission.model` | (prints) |

`$ROADSEG_DATA` points at the dataset folder (`.../RoadSeg/dataset`, containing `Train/` and `Val/`).

## The whole pipeline

```mermaid
flowchart LR
    data[("Train + Val tiles<br/>$ROADSEG_DATA")] --> splits["splits.py make<br/>5 folds, stratified<br/>urban / rural"]
    splits --> folds[/"splits/folds.csv"/]
    folds --> h0["train_h0.py<br/>H0 heads on frozen DINOv2"]
    h0 --> h0w[/"weights/h0/encoder_scheme/foldF/<br/>h0_*.safetensors + plan.json"/]
    folds --> tr["train.py<br/>refiner"]
    exp[/"experiments/exp.json<br/>+ --arch / --set"/] --> tr
    h0w --> tr
    tr --> rw[/"weights/run/foldF/<br/>refiner_best.safetensors"/]
    rw --> ex["export_submission.py"]
    h0w --> ex
    ex --> zip[/"dist/submission.zip<br/>infer.py, frozen_encoder.py,<br/>submission/, 1 weights file"/]
    zip --> inf["infer.py<br/>(leaderboard)"]
    inf --> masks[/"OUT/kK/id.png<br/>1 = road, 0 = not"/]
```

## Inside one run: data → frozen encoder → refiner

```mermaid
flowchart TB
    subgraph pipeline ["Pipeline (not shipped, except frozen_encoder.py)"]
        tile["data.py<br/>TileDataset, flips / rotations,<br/>StressAugment"]
        enc["frozen_encoder.py: encoder<br/>frozen DINOv2 → f at 1/16"]
        h0["frozen_encoder.py: H0 head<br/>f → y0 (full resolution)"]
        crop["data.py<br/>random_crops, degrade_logits"]
        tile --> enc --> h0 --> crop
    end
    subgraph refiner ["Refiner: submission/ (shipped)"]
        init["memory.py<br/>initialize(f, y0) → z0"]
        step["model.py: step(f, z_t, y_t)<br/>run K times, same weights"]
        init --> step
        step -- "z_t+1, y_t+1" --> step
    end
    crop --> init
    crop --> step
    step --> loss["losses.py<br/>BCE + Dice + clDice,<br/>every step"]
    step --> ev["evaluation.py<br/>validate at K = 0, 1, 2, 4"]
```

## One refinement step (`submission/model.py`)

```mermaid
flowchart LR
    f["features f"] --> tok
    y["answer y_t"] --> tok
    tok["tokenizers.py<br/>one fused token<br/>per 16x16 patch<br/>+ pos.py codes"] --> cat(("concat"))
    z["memory z_t<br/>(32 tokens)"] --> cat
    cat --> core["core.py<br/>ONE shared transformer<br/>called once per step"]
    core -- "grid tokens" --> head["output_heads.py<br/>pixel-shuffle + dilated CNN<br/>→ correction dy"]
    core -- "memory tokens" --> z2["z_t+1"]
    head --> add(("y_t + dy"))
    y --> add
    add --> y2["y_t+1"]
    y --> head
```

Every box is chosen by `submission/config.py`: `tokenizer`, `memory`, `core` and `head` sections,
each with a `kind` that picks a class from that file's registry (`TOKENIZERS`, `MEMORIES`, `HEADS`).

## Which H0 gives the refiner its y0 (fold 0 validating)

```mermaid
flowchart LR
    subgraph train ["training folds 1-4"]
        f1["fold 1"]
        f2["fold 2"]
        f3["fold 3"]
        f4["fold 4"]
    end
    v["fold 0<br/>validation"]
    oof1["h0_oof1<br/>trained on folds 2, 3, 4"] -- "y0 for fold 1 images" --> f1
    oof2["h0_oof2<br/>trained on folds 1, 3, 4"] -- "y0 for fold 2 images" --> f2
    oofn["h0_oof3, h0_oof4<br/>same pattern"] -- "y0 for folds 3, 4" --> f3 & f4
    all["h0_all<br/>trained on folds 1-4"] -- "y0 for validation<br/>and leaderboard" --> v
```

Default `--h0-scheme oof`: the refiner trains on all four training folds, always with y0 from an
H0 that never saw the image. Option `halves`: H0 (`h0_a`) on half of the training folds, the
refiner on the other half, `h0_a` everywhere.

## File map

```
project_a/cli/
├── submission/               THE REFINER (shipped in the zip; staff predict.py imports it)
│   ├── config.py             architecture presets (PRESETS, ACTIVE), overrides, metadata round trip
│   ├── model.py              Refiner: initialize() / step(); build_refiner(), load_refiner()
│   ├── tokenizers.py         (f, y_t) → grid tokens          doc stages 5, 7, 8
│   ├── memory.py             grid → z0 (memory initialization)   doc stage 6
│   ├── core.py               the shared transformer (model.core)   doc stages 9-11
│   ├── output_heads.py       grid tokens → dy                 doc stages 12-13
│   └── pos.py                2D sin/cos position code
├── frozen_encoder.py         frozen encoder + H0 heads → (f, y0)   doc stages 2, 4 (shipped)
├── infer.py                  leaderboard CLI: encode stage + refine stage (shipped)
├── data.py                   tile dataset, augmentation, crops, y0 degradation   doc stage 1
├── splits.py                 CV folds, OOF / halves plan, fold manifests        doc stage 3
├── losses.py                 BCE + Dice + clDice
├── evaluation.py             validate at any K; K=0 is the peek-in eval        doc stages 4b, 14
├── train_h0.py               train H0 heads for a fold
├── train.py                  train the refiner for a fold
├── export_submission.py      build dist/submission(.zip), --check runs it from an empty folder
├── experiments/*.json        training recipes (one per experiment; fold_overrides per fold)
├── jobs/                     SCC: cv.sh, train_h0.sh, train.sh, common.sh; past/ = old scripts
├── tests/test_project.py     our tests (presets, config, folds, end to end)
├── make_manifest.py          manifest from a whole LoveDA split (older; splits.py manifest does it per fold)
└── staff kit, do not edit: predict.py baseline.py common.py metrics.py evaluate.py
    score.py make_stress.py make_demo.py smoke_test.py
    tests/test_interface.py tests/test_metrics.py
```

## Adding a variant

1. **Architecture change** (e.g. a new head): add a class next to the existing one in
   `submission/output_heads.py` and register it in `HEADS`; add any new fields to `HeadConfig` with
   defaults that keep old presets unchanged; add a preset in `PRESETS` that uses it.
   Never edit a preset that has trained weights.
2. **Quick ablation of an existing field**: no code. `--set core.layers=6` on the command line or
   `"set": ["core.layers=6"]` in an experiment file.
3. **Training-recipe change**: copy an `experiments/*.json`, change values, give it a new `"run"`.
4. Run `python -m unittest discover -s tests`: every preset is checked against the staff
   runner's rules and the parameter caps.
5. To submit it, set `ACTIVE` in `submission/config.py` to the preset, so the staff
   `predict.py` builds the same model. (`infer.py` reads the architecture from the weights.)
