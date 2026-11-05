import optuna
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    model = XGBClassifier(n_estimators=200, random_state=42, **params)
    score = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()
    return score


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best score:", study.best_value)
print("Best params:", study.best_params)


print(study.best_value)
print(study.best_params)
print(study.best_trial)
print(study.trials_dataframe()) 