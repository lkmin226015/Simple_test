# Overfitting Test

This folder contains a simple regression overfitting experiment.

The main script is [polynomial_ann_cv.py](/abs/path/c:/Users/user/Desktop/PythonCode/overfitting_test/polynomial_ann_cv.py).

## Goal

- Generate about 100 synthetic samples from an arbitrary N-th order polynomial.
- Add Gaussian noise to simulate measurement error.
- Split the data into train and test sets.
- Apply K-fold cross-validation only on the train set.
- Select the best ANN structure from CV results.
- Evaluate the final selected model on the hold-out test set.

Important:
The test set is not used for model selection or training.
The intended flow is:

`train/test split -> K-fold CV on train only -> final evaluation on test`

## Files

- [polynomial_ann_cv.py](/abs/path/c:/Users/user/Desktop/PythonCode/overfitting_test/polynomial_ann_cv.py): synthetic data generation, ANN training, K-fold CV, and final test evaluation

## Default setup

- Polynomial degree: 4
- Number of samples: 100
- Noise standard deviation: `0.15`
- Train/test split: `80/20`
- K-folds: `5`
- Hidden layer candidates: `(8,)`, `(16,)`, `(32,)`, `(64, 32)`

The ANN is a small `numpy`-only MLP regressor.

## Run

Run from the project root:

```powershell
Test_project\.venv\Scripts\python.exe overfitting_test\polynomial_ann_cv.py
```

## Example

```powershell
Test_project\.venv\Scripts\python.exe overfitting_test\polynomial_ann_cv.py `
  --coefficients 1.0 -2.0 0.5 3.0 `
  --samples 120 `
  --noise-std 0.2 `
  --test-size 0.25 `
  --k-folds 4 `
  --hidden 8 16 32 64,32 `
  --epochs 4000 `
  --learning-rate 0.01
```

## Main arguments

- `--coefficients`: polynomial coefficients from highest degree to constant term
- `--samples`: total number of synthetic samples
- `--noise-std`: Gaussian noise level
- `--test-size`: hold-out test ratio
- `--k-folds`: number of folds applied to the train set only
- `--hidden`: hidden layer candidates
- `--epochs`: training epochs per fold
- `--learning-rate`: learning rate
- `--batch-size`: mini-batch size
- `--l2`: L2 regularization strength
- `--seed`: random seed

Examples for `--hidden`:

- `--hidden 8 16 32`
- `--hidden 8 32,16 64,32`

Here `32,16` means a network with two hidden layers.

## Output

The script prints:

- polynomial coefficients and degree
- train/test split sizes
- CV summary for each hidden layer candidate
- `train_rmse`, `val_rmse`, and `rmse_gap`
- selected final model structure
- final train and test metrics
- sample predictions on the hold-out test set

## How to read overfitting

Potential overfitting signals:

- very low `train_rmse` but much higher `val_rmse`
- very high `train_r2` but weaker `val_r2`
- final `test RMSE` consistently worse than CV validation error

Better generalization usually means:

- small gap between train and validation metrics
- similar behavior on the final test set

## Current limits

- Current experiment is `1D input x -> 1D output y`
- No plot export yet
- No CSV export yet
- Uses a custom `numpy` ANN instead of `scikit-learn`

## Possible extensions

- add matplotlib plots
- save results to CSV
- support multi-input polynomial data
- add larger ANN candidates to force stronger overfitting
- compare proper train-only CV vs intentionally incorrect test-set CV
