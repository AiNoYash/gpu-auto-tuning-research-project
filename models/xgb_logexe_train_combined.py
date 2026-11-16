"""Train the combined LogExe model using the completed tuning run's artifacts."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent
TUNER_PATH = MODEL_DIR / "hyperparameter-tuning" / "xgb_logexe_optuna_combined.py"
TUNING_DIR = TUNER_PATH.parent / "combined_logexe"
MODEL_PATH = MODEL_DIR / "xgb_logexe_model_combined.json"
REPORT_PATH = MODEL_DIR / "xgb_logexe_combined_holdout_metrics.json"


def load_tuner():
    spec = importlib.util.spec_from_file_location("combined_logexe_tuning", TUNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_training_inputs():
    """Reuse tuning preprocessing and verify the saved dataset/split identity."""
    tuning = load_tuner()
    manifest = json.loads((TUNING_DIR / "manifest.json").read_text())
    params = json.loads((TUNING_DIR / "best_params.json").read_text())
    dev, holdout, _, sources = tuning.load_data()
    if manifest["features"] != tuning.FEATURES:
        raise ValueError("Feature order differs from the completed tuning run.")
    for kernel in tuning.KERNELS:
        for key in ("dataset_sha256", "split_sha256"):
            if sources[kernel][key] != manifest["sources"][kernel][key]:
                raise ValueError(f"{kernel}: {key} differs from the tuning run.")
    # Paths can change when moving the project between a laptop and Colab.
    for label, data in (("development", dev), ("holdout", holdout)):
        saved = pd.read_csv(TUNING_DIR / f"{label}_row_mapping.csv")
        actual = data[["kernel", "source_row"]].reset_index(drop=True)
        if not actual.equals(saved):
            raise ValueError(f"{label} row membership/order differs from tuning.")
    return tuning, manifest, params, dev, holdout


def main():
    import xgboost as xgb

    tuning, manifest, params, dev, holdout = load_training_inputs()
    rounds = params.pop("n_estimators")
    if not isinstance(rounds, int) or rounds < 1:
        raise ValueError("Saved n_estimators must be a positive integer.")
    features = manifest["features"]
    dtrain = xgb.DMatrix(dev[features], label=dev["log_latency"])
    print(f"Training on {len(dev)} development rows for {rounds} boosting rounds.")
    model = xgb.train(
        params, dtrain, num_boost_round=rounds,
        evals=[(dtrain, "train")], verbose_eval=50,
    )
    # Preserve the selected round count exactly; the 1.1 multiplier was applied by tuning.
    model.set_attr(
        kernel_types=json.dumps(list(tuning.KERNELS)),
        preprocessing_function="xgb_logexe_optuna_combined.preprocess",
        invalid_latency=str(tuning.INVALID_LATENCY),
        target="log_latency_ms",
    )
    model.save_model(str(MODEL_PATH))
    print(f"Saved model to {MODEL_PATH}")

    dtest = xgb.DMatrix(holdout[features])
    prediction = model.predict(dtest)
    error = prediction - holdout["log_latency"].to_numpy()
    valid = holdout["is_valid"].to_numpy()
    per_kernel = tuning.selection_metrics(holdout, prediction, tuning.load_baselines())
    for kernel, metrics in per_kernel.items():
        mask = holdout["kernel"].eq(kernel).to_numpy()
        metrics["rows"] = int(mask.sum())
        metrics["log_rmse_including_invalid_penalties"] = float(np.sqrt(np.mean(error[mask] ** 2)))
    report = {
        "model": str(MODEL_PATH), "features": features,
        "boosting_rounds": rounds, "development_rows": len(dev), "holdout_rows": len(holdout),
        "holdout_log_rmse_including_invalid_penalties": float(np.sqrt(np.mean(error ** 2))),
        "holdout_valid_log_rmse": float(np.sqrt(np.mean(error[valid] ** 2))) if valid.any() else None,
        "per_kernel": per_kernel,
        "metric_definitions": manifest["selection_metrics"],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved holdout metrics to {REPORT_PATH}")


if __name__ == "__main__":
    main()
