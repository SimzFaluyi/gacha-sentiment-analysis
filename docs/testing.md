# Testing and Validation

## Purpose

Testing is used to verify the functionality of individual components
of the computational artefact.

Testing does not constitute empirical evaluation of the proposed
research models.

---

## Preprocessing Tests

The preprocessing tests cover:

1. missing-value removal;
2. empty-text removal;
3. duplicate removal;
4. short-comment filtering;
5. whitespace normalisation; and
6. execution of the combined preprocessing pipeline.

---

## Evaluation Tests

The evaluation tests cover:

1. macro-precision calculation;
2. macro-recall calculation;
3. macro-F1 calculation;
4. confusion-matrix generation; and
5. sentiment-class distribution.

---

## Running Tests

From the repository root:

```bash
pytest
```
---
## Expected Outcome

All automated tests should pass after the project dependencies have
been installed correctly.

## Empirical Evaluation

The repository does not contain empirical model-performance results.

This is because the planned Reddit research dataset was not collected
without the required institutional ethical approval.

Therefore:

accuracy is not reported as an empirical finding;
precision is not reported as an empirical finding;
recall is not reported as an empirical finding;
macro-F1 is not reported as an empirical finding; and
no model is claimed to be empirically superior.
