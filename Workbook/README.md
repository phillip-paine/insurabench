# Workbook

Four notebooks, each a self-contained walk through a common thing you'd
actually do with insurabench. Start with `01` if you're new to the
library; `02`-`04` can be read in any order after that.

## Running these

From the repo root:

```bash
pip install -e .
pip install jupyter
jupyter notebook Workbook/
```

Each notebook is self-contained (imports everything it needs, builds its
own data) except `01_getting_started.ipynb`, which reads the two CSVs in
`data/` -- run notebooks from *inside* this `Workbook/` folder (as the
command above does) so that relative path resolves.

## The notebooks

| Notebook | What it shows |
|---|---|
| `01_getting_started.ipynb` | Loading your own two-table CSV data, fitting a frequency model, checking how good it is, a lift chart, a one-way rating-factor curve. Start here. |
| `02_model_agnostic_pricing.ipynb` | The headline feature: fit a GLM and a GBM, then run identical curves/evaluation code against both, plus a direct double-lift comparison. |
| `03_pricing_the_whole_book.ipynb` | Getting a full pure-premium price two ways -- a single direct Tweedie model vs. composing separate frequency and severity models (which can be different model types) -- compared on the same footing. |
| `04_from_model_to_deliverables.ipynb` | Turning a fitted model into a one-page model-card summary and an actuarial-format rating-table export. |

## `data/`

Two small, fixed sample CSVs (`sample_policies.csv`/`sample_claims.csv`)
used by notebook 01 -- see `data/README.md` for the column-by-column
breakdown and why the data is split into two tables. Notebooks 02-04 use
a larger, in-memory synthetic book instead (via
`insurabench.data.datasets.make_synthetic_two_table`) -- large enough to
show a GLM and a GBM meaningfully differ, which the small sample data
isn't.

## A note on the numbers

These notebooks use synthetic data with deliberately simple, documented
signal (not a realistic actuarial relationship -- see
`make_synthetic_two_table`'s docstring). The actual metric values you'll
see (D², Gini, etc.) are discussed honestly where they're genuinely
informative (e.g. notebook 02's GLM-vs-GBM comparison) and shouldn't be
read as a benchmark of "how good insurabench's models are" in general --
that depends entirely on your own data.
