# Validation record

Prepared August 29, 2026; revised August 30, 2026.

## Handout

- Read the EC523 Fall 2026 syllabus supplied in the project; preserved its
  dates and AI-use requirements, with project-specific team size three.
- Compiled the LaTeX source with TeX Live 2024 / latexmk to a 12-page PDF.
- Final compile has no undefined citations/references, overfull boxes,
  underfull-box warnings, or other LaTeX warnings.
- Rendered all 12 final pages using Poppler and inspected every page.
  Corrected a run-in title, figure crop, overlapping diagram label, long
  code lines, and three unwanted spillover pages before delivery.
- All seven bibliography entries resolve. The PDF contains six numbered
  figures (one real dataset-example figure and five vector diagrams).
- The numerical score example is tested against the evaluator: 63.0.

## August 30 revision checks

- Defined the latent token state z_t separately from the dense logit map y_t;
  used g for ground truth and L for loss. Updated the interface documentation.
- Rebuilt the Section 3 diagram with explicit image, answer, and memory
  token banks, shared processing, and both feedback paths.
- Recompiled and visually inspected all 12 revised PDF pages. Section 3's
  definitions, diagram, and mandatory rules now fit together on one page.
- Repeated all 27 tests and the full four-run CLI smoke workflow successfully.
  Scoring formulas, label encoding, and the callable Python API are unchanged.
- Student materials use neutral course-data terminology; the figure retains
  traceable source attribution. Full provenance is in the instructor package.
- Distribution archives exclude generated Python caches and test outputs.

## CLI and evaluation

- CPU environment: Python 3.12, PyTorch 2.6.0, NumPy 2.3.5,
  SciPy 1.15.2, scikit-image 0.25.2, safetensors 0.5.3, Pillow 12.3.0.
- 27 unit/integration tests cover exact loop counts, K=0 baseline identity,
  K=2/K=4 prefix consistency, K=6/8 execution, feedback interventions,
  missing official-baseline failure, core-call validation, output guards,
  all stress families, deterministic stress generation, report combination,
  invalid inputs, empty/ignored labels, pooled metrics, threshold semantics,
  and the exact three-pixel boundary band.
- Ran the complete command-line smoke workflow as separate processes:
  synthetic data and random weights -> ordinary K=1,2,4 -> stress K=4 ->
  four evaluation reports -> combined score. It passed and remained
  explicitly marked DEMO ONLY. Those scores are not course-data results.

## Not performed / not included

- No training on course data, trained staff baseline, actual course split,
  L40S/CUDA benchmarking, training-memory measurement, or resource pilot.
- No competition website, submission quota service, remote execution
  container, secret test labels, or public/private evaluation deployment.
- No claim that runtime hooks prove honest behavior or establish reasoning.
  Source review, controlled experiments, and isolated execution are required.
- No claim of benchmark state of the art or publication novelty.

The instructor release checklist is in the separate instructor package.
