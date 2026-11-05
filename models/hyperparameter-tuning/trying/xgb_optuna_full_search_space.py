import optuna
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    params = {
        # speed of learning
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),

        # tree complexity
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 20, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5, log=True),

        # randomness
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),

        # regularization
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
    }

    model = XGBClassifier(
        n_estimators=300,          # fixed for now; Part 3 handles this properly
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        **params,
    )
    return cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=60)

print("Best AUC:", study.best_value)
print(study.best_params)