# Telecom churn prediction 

Code for the churn article, run on the Cell2Cell and Orange Telecom datasets.
The notebook compares six classifiers with basic vs. enhanced preprocessing and
with/without SMOTE-NC, then runs SHAP on the best tree model for each dataset.

## Getting the data

The datasets are not in this repo, so download them from Kaggle and put the
files in `data/`:

- Cell2Cell: https://www.kaggle.com/code/ngclongon/cell2cell/input
  (`cell2celltrain.csv.zip`)
- Orange Telecom: https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset
  (`bigml_59c28831336c6604c800002a.csv`)

Note: the Cell2Cell archive I downloaded only had the labelled file.
No unlabelled holdout file was included or used.

Dataset licences were not verified. Both datasets are on Kaggle and subject to
Kaggle's Terms of Use (https://www.kaggle.com/terms). Check the dataset pages
and those terms before reuse.

## Running it

Install `requirements.txt` (Python 3.13.16), open the notebook in Colab on a
CPU runtime and run the cells from top to bottom. Tuning results are cached in
`objects/`. With the cache the notebook finishes quickly; delete the folder to
redo the searches, which takes about 5 hours.

## Settings

Seed 42 everywhere. Stratified 80/20 split, 5-fold CV on the training part,
1000 bootstrap resamples, alpha 0.05. Logistic Regression and Decision Tree use
their full grids; the ensembles use random search with 10 configurations on
Orange and 6 on Cell2Cell. A configuration is scored by its best F1 over the
threshold grid on pooled out-of-fold predictions.

## Environment

Google Colab, CPU only: Python 3.13.16, Linux 6.6.122+, Intel Xeon 2.20 GHz,
2 cores, 12.7 GB RAM. Tuning plus final fitting took about 314 min (296 on
Cell2Cell, 18 on Orange), not counting bootstrap, tests and SHAP.

## Output

The generated tables (CSV and LaTeX) and figures (PDF and SVG) are archived as
`tables-20261025T133027Z-1-001.zip` and `figures-20261025T133026Z-1-001.zip`.
Unzip them to get the `tables/` and `figures/` folders.

## Licence

The code is released under the MIT License (see `LICENSE`).
