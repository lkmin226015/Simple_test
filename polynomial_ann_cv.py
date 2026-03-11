from __future__ import annotations

from dataclasses import dataclass
import argparse
import math

import numpy as np


@dataclass(frozen=True)
class DatasetConfig:
    coefficients: tuple[float, ...]
    n_samples: int = 100
    x_min: float = -2.0
    x_max: float = 2.0
    noise_std: float = 0.15
    random_seed: int = 7


@dataclass(frozen=True)
class TrainingConfig:
    test_size: float = 0.2
    k_folds: int = 5
    hidden_layer_candidates: tuple[tuple[int, ...], ...] = ((8,), (16,), (32,), (64, 32))
    learning_rate: float = 0.01
    epochs: int = 3000
    batch_size: int = 16
    l2_lambda: float = 1.0e-4
    random_seed: int = 7


@dataclass(frozen=True)
class StandardScaler1D:
    mean: float
    std: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


@dataclass(frozen=True)
class FoldMetrics:
    train_rmse: float
    val_rmse: float
    train_mae: float
    val_mae: float
    train_r2: float
    val_r2: float


@dataclass(frozen=True)
class CandidateResult:
    hidden_layers: tuple[int, ...]
    folds: tuple[FoldMetrics, ...]

    @property
    def mean_train_rmse(self) -> float:
        return float(np.mean([fold.train_rmse for fold in self.folds]))

    @property
    def mean_val_rmse(self) -> float:
        return float(np.mean([fold.val_rmse for fold in self.folds]))

    @property
    def mean_rmse_gap(self) -> float:
        return self.mean_val_rmse - self.mean_train_rmse

    @property
    def mean_train_r2(self) -> float:
        return float(np.mean([fold.train_r2 for fold in self.folds]))

    @property
    def mean_val_r2(self) -> float:
        return float(np.mean([fold.val_r2 for fold in self.folds]))


def polynomial_value(x: np.ndarray, coefficients: tuple[float, ...]) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    degree = len(coefficients) - 1
    for index, coefficient in enumerate(coefficients):
        power = degree - index
        y += coefficient * (x**power)
    return y


def make_dataset(config: DatasetConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.random_seed)
    x = np.linspace(config.x_min, config.x_max, config.n_samples, dtype=float)
    x += rng.normal(0.0, 0.03 * (config.x_max - config.x_min), size=config.n_samples)
    x = np.clip(x, config.x_min, config.x_max)
    x.sort()

    y_true = polynomial_value(x, config.coefficients)
    noise = rng.normal(0.0, config.noise_std, size=config.n_samples)
    y_noisy = y_true + noise
    return x.reshape(-1, 1), y_noisy.reshape(-1, 1), y_true.reshape(-1, 1)


def train_test_split_indices(n_samples: int, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    test_count = max(1, int(round(n_samples * test_size)))
    test_indices = np.sort(indices[:test_count])
    train_indices = np.sort(indices[test_count:])
    return train_indices, test_indices


def make_kfolds(n_samples: int, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2:
        raise ValueError("k_folds must be at least 2.")
    if n_samples < n_splits:
        raise ValueError("Number of training samples must be at least k_folds.")

    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    current = 0
    for fold_size in fold_sizes:
        start = current
        stop = current + fold_size
        val_indices = np.sort(indices[start:stop])
        train_indices = np.sort(np.concatenate((indices[:start], indices[stop:])))
        folds.append((train_indices, val_indices))
        current = stop
    return folds


def fit_scaler(values: np.ndarray) -> StandardScaler1D:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1.0e-12:
        std = 1.0
    return StandardScaler1D(mean=mean, std=std)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if total <= 1.0e-12:
        return 0.0
    return 1.0 - residual / total


class MLPRegressor:
    def __init__(
        self,
        hidden_layers: tuple[int, ...],
        learning_rate: float,
        epochs: int,
        batch_size: int,
        l2_lambda: float,
        random_seed: int,
    ) -> None:
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2_lambda = l2_lambda
        self.random_seed = random_seed
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

    def _initialize(self, input_dim: int, output_dim: int) -> None:
        rng = np.random.default_rng(self.random_seed)
        layer_sizes = [input_dim, *self.hidden_layers, output_dim]
        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            self.weights.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out), dtype=float))

    @staticmethod
    def _tanh_derivative(activated: np.ndarray) -> np.ndarray:
        return 1.0 - activated**2

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self._initialize(input_dim=x_train.shape[1], output_dim=y_train.shape[1])
        rng = np.random.default_rng(self.random_seed)

        for _ in range(self.epochs):
            indices = np.arange(x_train.shape[0])
            rng.shuffle(indices)

            for start in range(0, x_train.shape[0], self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                batch_x = x_train[batch_indices]
                batch_y = y_train[batch_indices]

                activations = [batch_x]
                pre_activations = []

                current = batch_x
                for layer_index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
                    z = current @ weight + bias
                    pre_activations.append(z)
                    if layer_index == len(self.weights) - 1:
                        current = z
                    else:
                        current = np.tanh(z)
                    activations.append(current)

                prediction = activations[-1]
                batch_size = batch_x.shape[0]
                delta = 2.0 * (prediction - batch_y) / batch_size

                grad_w: list[np.ndarray] = []
                grad_b: list[np.ndarray] = []

                for layer_index in range(len(self.weights) - 1, -1, -1):
                    prev_activation = activations[layer_index]
                    weight = self.weights[layer_index]
                    dw = prev_activation.T @ delta + self.l2_lambda * weight
                    db = np.sum(delta, axis=0, keepdims=True)
                    grad_w.insert(0, dw)
                    grad_b.insert(0, db)

                    if layer_index > 0:
                        delta = (delta @ weight.T) * self._tanh_derivative(activations[layer_index])

                for layer_index in range(len(self.weights)):
                    self.weights[layer_index] -= self.learning_rate * grad_w[layer_index]
                    self.biases[layer_index] -= self.learning_rate * grad_b[layer_index]

    def predict(self, x_input: np.ndarray) -> np.ndarray:
        current = x_input
        for layer_index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            current = current @ weight + bias
            if layer_index < len(self.weights) - 1:
                current = np.tanh(current)
        return current


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    return rmse(y_true, y_pred), mae(y_true, y_pred), r2_score(y_true, y_pred)


def cross_validate_candidates(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: TrainingConfig,
) -> list[CandidateResult]:
    folds = make_kfolds(n_samples=x_train.shape[0], n_splits=config.k_folds, seed=config.random_seed)
    results: list[CandidateResult] = []

    for candidate_index, hidden_layers in enumerate(config.hidden_layer_candidates):
        fold_metrics: list[FoldMetrics] = []

        for fold_index, (fold_train_idx, fold_val_idx) in enumerate(folds):
            fold_x_train = x_train[fold_train_idx]
            fold_y_train = y_train[fold_train_idx]
            fold_x_val = x_train[fold_val_idx]
            fold_y_val = y_train[fold_val_idx]

            x_scaler = fit_scaler(fold_x_train)
            y_scaler = fit_scaler(fold_y_train)

            scaled_x_train = x_scaler.transform(fold_x_train)
            scaled_y_train = y_scaler.transform(fold_y_train)
            scaled_x_val = x_scaler.transform(fold_x_val)

            model = MLPRegressor(
                hidden_layers=hidden_layers,
                learning_rate=config.learning_rate,
                epochs=config.epochs,
                batch_size=config.batch_size,
                l2_lambda=config.l2_lambda,
                random_seed=config.random_seed + 31 * candidate_index + fold_index,
            )
            model.fit(scaled_x_train, scaled_y_train)

            train_pred = y_scaler.inverse_transform(model.predict(scaled_x_train))
            val_pred = y_scaler.inverse_transform(model.predict(scaled_x_val))

            train_rmse, train_mae, train_r2 = evaluate_predictions(fold_y_train, train_pred)
            val_rmse, val_mae, val_r2 = evaluate_predictions(fold_y_val, val_pred)
            fold_metrics.append(
                FoldMetrics(
                    train_rmse=train_rmse,
                    val_rmse=val_rmse,
                    train_mae=train_mae,
                    val_mae=val_mae,
                    train_r2=train_r2,
                    val_r2=val_r2,
                )
            )

        results.append(CandidateResult(hidden_layers=hidden_layers, folds=tuple(fold_metrics)))
    return results


def train_final_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    hidden_layers: tuple[int, ...],
    config: TrainingConfig,
) -> tuple[MLPRegressor, StandardScaler1D, StandardScaler1D]:
    x_scaler = fit_scaler(x_train)
    y_scaler = fit_scaler(y_train)

    scaled_x_train = x_scaler.transform(x_train)
    scaled_y_train = y_scaler.transform(y_train)

    model = MLPRegressor(
        hidden_layers=hidden_layers,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        batch_size=config.batch_size,
        l2_lambda=config.l2_lambda,
        random_seed=config.random_seed,
    )
    model.fit(scaled_x_train, scaled_y_train)
    return model, x_scaler, y_scaler


def format_layers(hidden_layers: tuple[int, ...]) -> str:
    return "-".join(str(width) for width in hidden_layers)


def run_experiment(dataset_config: DatasetConfig, training_config: TrainingConfig) -> None:
    x_all, y_all, y_true = make_dataset(dataset_config)
    train_idx, test_idx = train_test_split_indices(
        n_samples=x_all.shape[0],
        test_size=training_config.test_size,
        seed=training_config.random_seed,
    )

    x_train = x_all[train_idx]
    y_train = y_all[train_idx]
    x_test = x_all[test_idx]
    y_test = y_all[test_idx]
    y_test_true = y_true[test_idx]

    candidate_results = cross_validate_candidates(x_train=x_train, y_train=y_train, config=training_config)
    best_result = min(candidate_results, key=lambda result: result.mean_val_rmse)

    final_model, x_scaler, y_scaler = train_final_model(
        x_train=x_train,
        y_train=y_train,
        hidden_layers=best_result.hidden_layers,
        config=training_config,
    )

    scaled_x_train = x_scaler.transform(x_train)
    scaled_x_test = x_scaler.transform(x_test)
    train_pred = y_scaler.inverse_transform(final_model.predict(scaled_x_train))
    test_pred = y_scaler.inverse_transform(final_model.predict(scaled_x_test))

    train_rmse, train_mae, train_r2 = evaluate_predictions(y_train, train_pred)
    test_rmse, test_mae, test_r2 = evaluate_predictions(y_test, test_pred)
    test_true_rmse, _, test_true_r2 = evaluate_predictions(y_test_true, test_pred)

    print("=== Polynomial synthetic ANN overfitting test ===")
    print(f"Polynomial coefficients: {dataset_config.coefficients}")
    print(f"Polynomial degree      : {len(dataset_config.coefficients) - 1}")
    print(f"Total samples          : {dataset_config.n_samples}")
    print(f"Train/Test split       : {x_train.shape[0]}/{x_test.shape[0]}")
    print(f"K-folds on train set   : {training_config.k_folds}")
    print(f"Noise std              : {dataset_config.noise_std:.4f}")
    print("CV rule                : K-fold is applied only to the train split, then the final score is measured on hold-out test.")

    print("\n=== Cross-validation summary on train set ===")
    print(f"{'hidden':>12} {'train_rmse':>12} {'val_rmse':>12} {'rmse_gap':>12} {'train_r2':>12} {'val_r2':>12}")
    for result in candidate_results:
        print(
            f"{format_layers(result.hidden_layers):>12} "
            f"{result.mean_train_rmse:12.5f} "
            f"{result.mean_val_rmse:12.5f} "
            f"{result.mean_rmse_gap:12.5f} "
            f"{result.mean_train_r2:12.5f} "
            f"{result.mean_val_r2:12.5f}"
        )

    print("\n=== Selected model ===")
    print(f"Best hidden layers     : {best_result.hidden_layers}")

    print("\n=== Final metrics ===")
    print(f"Train RMSE             : {train_rmse:.5f}")
    print(f"Train MAE              : {train_mae:.5f}")
    print(f"Train R2               : {train_r2:.5f}")
    print(f"Test RMSE              : {test_rmse:.5f}")
    print(f"Test MAE               : {test_mae:.5f}")
    print(f"Test R2                : {test_r2:.5f}")
    print(f"Test-Train RMSE gap    : {test_rmse - train_rmse:.5f}")
    print(f"Test RMSE vs true poly : {test_true_rmse:.5f}")
    print(f"Test R2 vs true poly   : {test_true_r2:.5f}")

    print("\n=== Sample predictions on hold-out test set ===")
    print(f"{'x':>10} {'y_noisy':>12} {'y_true':>12} {'y_pred':>12} {'abs_err':>12}")
    for x_value, y_noisy, y_clean, y_pred in zip(x_test[:10], y_test[:10], y_test_true[:10], test_pred[:10]):
        print(
            f"{float(x_value[0]):10.4f} "
            f"{float(y_noisy[0]):12.5f} "
            f"{float(y_clean[0]):12.5f} "
            f"{float(y_pred[0]):12.5f} "
            f"{abs(float(y_noisy[0] - y_pred[0])):12.5f}"
        )


def parse_hidden_layers(text: str) -> tuple[int, ...]:
    values = tuple(int(chunk.strip()) for chunk in text.split(",") if chunk.strip())
    if not values:
        raise argparse.ArgumentTypeError("Hidden layer candidate must contain at least one integer.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfitting experiment with polynomial synthetic data and a small ANN regressor.")
    parser.add_argument(
        "--coefficients",
        type=float,
        nargs="+",
        default=[0.8, -1.5, 0.0, 2.0, -0.5],
        help="Polynomial coefficients from highest degree to constant term.",
    )
    parser.add_argument("--samples", type=int, default=100, help="Total number of synthetic samples.")
    parser.add_argument("--noise-std", type=float, default=0.15, help="Standard deviation of Gaussian noise.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction used as hold-out test set.")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of folds used only on the training set.")
    parser.add_argument(
        "--hidden",
        type=parse_hidden_layers,
        nargs="+",
        default=[(8,), (16,), (32,), (64, 32)],
        help="Candidate hidden layer sizes. Example: --hidden 8 16 64,32",
    )
    parser.add_argument("--epochs", type=int, default=3000, help="Training epochs per fold.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--l2", type=float, default=1.0e-4, help="L2 regularization strength.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_config = DatasetConfig(
        coefficients=tuple(args.coefficients),
        n_samples=args.samples,
        noise_std=args.noise_std,
        random_seed=args.seed,
    )
    training_config = TrainingConfig(
        test_size=args.test_size,
        k_folds=args.k_folds,
        hidden_layer_candidates=tuple(args.hidden),
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        l2_lambda=args.l2,
        random_seed=args.seed,
    )
    run_experiment(dataset_config=dataset_config, training_config=training_config)


if __name__ == "__main__":
    main()
