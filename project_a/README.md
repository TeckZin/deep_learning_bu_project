# EC523 Project A: student materials

- `handout/project_a.tex`: editable LaTeX handout, with vector diagrams,
  equations, bibliography, design choices, and grading expectations.
- `handout/assets/course_imagery.png`: attributed imagery/annotation figure.
- `cli/`: submission runner, untrained example, evaluator, stress generator,
  synthetic demo generator, tests, and usage instructions.

Course staff will supply the versioned data package and official split
manifests. Do not obtain or reconstruct evaluation labels from other sources.
The source credit for the borrowed figure remains in the handout; staff will
provide full research-provenance instructions before any paper submission.

Compile from the `handout` directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error project_a.tex
```

The bibliography is embedded in the TeX source, so no BibTeX step is needed.
The PDF was built with TeX Live and checked by rendering every page. See
`VALIDATION.md` for the actual checks performed and untested items.

The separate instructor package contains release decisions, data provenance,
and hosting notes. Those notes are not part of the student distribution.

This is a review draft: the trained frozen baseline and final course resource
allocation are intentionally not invented. The source `.tex` and the full
bundle should travel with the PDF so figures and code remain reproducible.
