#!/usr/bin/env python3
"""Train MiniROCKET on any UCR dataset and write the two JSONs the NPU scripts read.

Every series is resampled to 251 samples (linear interpolation) and normalized
per series, so the compiled constants (length 251, 840 features) stay valid.
Transform, scaler and classifier are fit on the TRAIN split only.
"""
import argparse
import json
import os
from itertools import combinations

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_LEN = 251


class MiniRocket:
    """MiniRocket implementation matching sktime reference implementation"""

    def __init__(self, num_kernels=10000, max_dilations_per_kernel=32, random_state=None):
        self.num_kernels = num_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.random_state = random_state
        self.kernels = None
        self.dilations = None
        self.num_features_per_dilation = None
        self.biases = None
        self.parameters = None

    def _quantiles(self, n):
        return np.array(
            [(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32
        )

    def _fit_dilations(self, n_timepoints, num_features, max_dilations_per_kernel):
        num_kernels = 84
        if num_features < 84:
            num_features = 84

        num_features_per_kernel = num_features // num_kernels
        true_max_dilations_per_kernel = min(
            num_features_per_kernel, max_dilations_per_kernel
        )
        multiplier = num_features_per_kernel / true_max_dilations_per_kernel

        max_exponent = np.log2((n_timepoints - 1) / (9 - 1))
        dilations, num_features_per_dilation = np.unique(
            np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(
                np.int32
            ),
            return_counts=True,
        )
        num_features_per_dilation = (num_features_per_dilation * multiplier).astype(
            np.int32
        )

        remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
        i = 0
        while remainder > 0:
            num_features_per_dilation[i] += 1
            remainder -= 1
            i = (i + 1) % len(num_features_per_dilation)

        return dilations, num_features_per_dilation

    def _fit_biases(self, X, dilations, num_features_per_dilation, quantiles, seed):
        if seed is not None:
            np.random.seed(seed)

        n_instances, n_timepoints = X.shape
        indices = np.array(list(combinations(range(9), 3)), dtype=np.int32)
        num_kernels = len(indices)
        num_dilations = len(dilations)
        num_features = num_kernels * np.sum(num_features_per_dilation)
        biases = np.zeros(num_features, dtype=np.float32)
        feature_index_start = 0

        for dilation_index in range(num_dilations):
            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2
            num_features_this_dilation = num_features_per_dilation[dilation_index]

            for kernel_index in range(num_kernels):
                feature_index_end = feature_index_start + num_features_this_dilation
                _X = X[np.random.randint(n_instances)]
                A = -_X
                G = _X + _X + _X
                C_alpha = np.zeros(n_timepoints, dtype=np.float32)
                C_alpha[:] = A
                C_gamma = np.zeros((9, n_timepoints), dtype=np.float32)
                C_gamma[9 // 2] = G
                start = dilation
                end = n_timepoints - padding

                for gamma_index in range(9 // 2):
                    C_alpha[-end:] = C_alpha[-end:] + A[:end]
                    C_gamma[gamma_index, -end:] = G[:end]
                    end += dilation

                for gamma_index in range(9 // 2 + 1, 9):
                    C_alpha[:-start] = C_alpha[:-start] + A[start:]
                    C_gamma[gamma_index, :-start] = G[start:]
                    start += dilation

                index_0, index_1, index_2 = indices[kernel_index]
                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]
                biases[feature_index_start:feature_index_end] = np.quantile(
                    C, quantiles[feature_index_start:feature_index_end]
                )
                feature_index_start = feature_index_end

        return biases

    def fit(self, X):
        X = X.astype(np.float32)
        _, n_timepoints = X.shape
        self.time_series_length = n_timepoints

        if n_timepoints < 9:
            raise ValueError(f"n_timepoints must be >= 9, but found {n_timepoints}")

        if self.num_kernels < 84:
            self.num_kernels_ = 84
        else:
            self.num_kernels_ = (self.num_kernels // 84) * 84

        seed = np.int32(self.random_state) if isinstance(self.random_state, int) else None

        dilations, num_features_per_dilation = self._fit_dilations(
            n_timepoints, self.num_kernels_, self.max_dilations_per_kernel
        )

        num_features_per_kernel = np.sum(num_features_per_dilation)
        quantiles = self._quantiles(84 * num_features_per_kernel)
        biases = self._fit_biases(X, dilations, num_features_per_dilation, quantiles, seed)

        self.parameters = (dilations, num_features_per_dilation, biases)
        self.dilations = dilations
        self.num_features_per_dilation = num_features_per_dilation
        self.biases = biases
        return self

    def transform(self, X):
        X = X.astype(np.float32)
        n_instances, n_timepoints = X.shape
        dilations, num_features_per_dilation, biases = self.parameters
        indices = np.array(list(combinations(range(9), 3)), dtype=np.int32)
        num_kernels = len(indices)
        num_dilations = len(dilations)
        num_features = num_kernels * np.sum(num_features_per_dilation)
        features = np.zeros((n_instances, num_features), dtype=np.float32)

        for example_index in range(n_instances):
            _X = X[example_index]
            A = -_X
            G = _X + _X + _X
            feature_index_start = 0

            for dilation_index in range(num_dilations):
                _padding0 = dilation_index % 2
                dilation = dilations[dilation_index]
                padding = ((9 - 1) * dilation) // 2
                num_features_this_dilation = num_features_per_dilation[dilation_index]

                C_alpha = np.zeros(n_timepoints, dtype=np.float32)
                C_alpha[:] = A
                C_gamma = np.zeros((9, n_timepoints), dtype=np.float32)
                C_gamma[9 // 2] = G
                start = dilation
                end = n_timepoints - padding

                for gamma_index in range(9 // 2):
                    C_alpha[-end:] = C_alpha[-end:] + A[:end]
                    C_gamma[gamma_index, -end:] = G[:end]
                    end += dilation

                for gamma_index in range(9 // 2 + 1, 9):
                    C_alpha[:-start] = C_alpha[:-start] + A[start:]
                    C_gamma[gamma_index, :-start] = G[start:]
                    start += dilation

                for kernel_index in range(num_kernels):
                    feature_index_end = feature_index_start + num_features_this_dilation
                    _padding1 = (_padding0 + kernel_index) % 2
                    index_0, index_1, index_2 = indices[kernel_index]
                    C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                    if _padding1 == 0:
                        for feature_count in range(num_features_this_dilation):
                            features[example_index, feature_index_start + feature_count] = (
                                np.mean(C > biases[feature_index_start + feature_count])
                            )
                    else:
                        for feature_count in range(num_features_this_dilation):
                            features[example_index, feature_index_start + feature_count] = (
                                np.mean(
                                    C[padding:-padding] > biases[feature_index_start + feature_count]
                                )
                            )
                    feature_index_start = feature_index_end
        return features

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def resample_series(x, n=TARGET_LEN):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == n:
        return x
    return np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, len(x)), x)


def normalize_series(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def prep_array(series_list):
    out = [normalize_series(resample_series(s)) for s in series_list]
    return np.array(out, dtype=np.float32)


def load_ucr_split(name, split):
    from sktime.datasets import load_UCR_UEA_dataset
    X, y = load_UCR_UEA_dataset(name, split=split, return_X_y=True)
    series = [np.array(s.values, dtype=np.float64) for s in X.iloc[:, 0]]
    return series, np.array(y)


def save_model_parameters(minirocket, scaler, classifier, filename):
    kernel_indices = np.array(list(combinations(range(9), 3)), dtype=np.int32)
    model_data = {
        "num_kernels": 84,
        "num_dilations": len(minirocket.dilations),
        "num_features": len(minirocket.biases),
        "num_classes": len(classifier.classes_),
        "time_series_length": minirocket.time_series_length,
        "kernel_indices": kernel_indices.tolist(),
        "dilations": minirocket.dilations.tolist(),
        "num_features_per_dilation": minirocket.num_features_per_dilation.tolist(),
        "biases": minirocket.biases.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "classifier_coef": classifier.coef_.tolist(),
        "classifier_intercept": classifier.intercept_.tolist(),
        "classes": classifier.classes_.tolist(),
    }
    with open(filename, "w") as f:
        json.dump(model_data, f, indent=2)
    print(f"Model parameters saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Train MiniRocket on a UCR dataset")
    parser.add_argument("--dataset", type=str, required=True, help="UCR name, e.g. ArrowHead, CBF, Meat")
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument("--official-split", action="store_true",
                        help="use the archive train/test split instead of 70/30 stratified seed 42")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    tr_series, tr_y = load_ucr_split(args.dataset, "train")
    te_series, te_y = load_ucr_split(args.dataset, "test")
    print(f"raw: train {len(tr_series)}, test {len(te_series)}, "
          f"lengths {sorted(set(len(s) for s in tr_series + te_series))}")

    labels = sorted(set(tr_y.tolist()) | set(te_y.tolist()))
    label_to_int = {lab: i for i, lab in enumerate(labels)}
    n_classes = len(labels)
    if n_classes != 3:
        print(f"WARNING: dataset has {n_classes} classes, the NPU runners assume 3")

    if args.official_split:
        X_train = prep_array(tr_series)
        y_train = np.array([label_to_int[l] for l in tr_y])
        X_test = prep_array(te_series)
        y_test = np.array([label_to_int[l] for l in te_y])
    else:
        X = prep_array(tr_series + te_series)
        y = np.array([label_to_int[l] for l in list(tr_y) + list(te_y)])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)

    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    print("Training MiniRocket...")
    minirocket = MiniRocket(num_kernels=840, random_state=42)
    X_train_features = minirocket.fit_transform(X_train)
    X_test_features = minirocket.transform(X_test)
    print(f"Feature shape: {X_train_features.shape}")
    print(f"dilations {minirocket.dilations.tolist()}, "
          f"num_features_per_dilation {minirocket.num_features_per_dilation.tolist()}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)

    print("Training classifier...")
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_scaled, y_train)

    y_pred = classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"PYTHON BASELINE ACCURACY: {accuracy:.4f}")

    model_path = os.path.join(args.outdir, "minirocket_model.json")
    save_model_parameters(minirocket, scaler, classifier, model_path)

    test_data = {
        "dataset_name": args.dataset,
        "X_test": X_test.tolist(),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "test_accuracy": float(accuracy),
        "num_samples": len(X_test),
        "series_length": X_test.shape[1],
        "num_classes": len(np.unique(y_test)),
    }
    test_path = model_path.replace(".json", "_test_data.json")
    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=2)
    print(f"Full test dataset saved to {test_path}")


if __name__ == "__main__":
    main()
