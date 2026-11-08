# Telecom churn prediction 

Code for the churn article, run on the Cell2Cell and Orange Telecom datasets.
The notebook compares six classifiers with basic vs. enhanced preprocessing and
with/without SMOTE-NC, then runs SHAP on the best tree model for each dataset.

## Data

The two dataset files used in the study are in the repository root:

- Cell2Cell: `cell2celltrain.csv.zip`, downloaded from
  https://www.kaggle.com/code/ngclongon/cell2cell/input
- Orange Telecom: `bigml_59c28831336c6604c800002a.csv`, from
  https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset

The Cell2Cell archive contains only the labelled file. No unlabelled holdout
file was included or used.

Dataset licences were not verified. Both datasets are on Kaggle and subject to
Kaggle's Terms of Use (https://www.kaggle.com/terms). Check the dataset pages
and those terms before reuse.

## Running it

1. In Colab, choose a CPU runtime.
2. Put the two dataset files in `data/` inside the Google Drive folder
   `MyDrive/churn_assignment3/` (the notebook mounts Drive and reads from there).
3. Install the pinned versions: `pip install -r requirements.txt` (Python 3.13.16).
4. Open `Telecom_churn_research.ipynb` and run all cells from top to bottom.

The notebook's first cell also installs some packages without version pins.
These are not replaced once the pinned versions are installed. This has not
yet been tested in a clean Colab session.

Tuning results are cached in `objects/`. Delete that folder to repeat the
hyperparameter searches, which took about 287 minutes.

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
