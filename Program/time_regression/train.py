import xgboost as xgb
import optuna

print("Loading data from disk to DMatrix...")
test_data = xgb.DMatrix("test_data.buffer")
train_data = xgb.DMatrix("train_data.buffer")


def objective(trial: optuna.Trial):
    param = {
        "eta": trial.suggest_float("eta", 0.1, 1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1, 10
        ),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bynode": trial.suggest_float(
            "colsample_bynode", 0.6, 1.0
        ),
        "num_parallel_tree": trial.suggest_int(
            "num_parallel_tree", 1, 10
        ),
        "tree_method": "gpu_hist",
        "objective": "reg:squarederror",
        "eval_metric": ["rmse", "mae"],
    }
    bst = xgb.train(
        param,
        train_data,
        100,
        [(test_data, "eval")],
        early_stopping_rounds=10,
    )
    return bst.best_score


# find best params
print("Training...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)
best_params = study.best_trial.params
best_params["tree_method"] = "gpu_hist"
best_params["objective"] = "reg:squarederror"
best_params["eval_metric"] = ["rmse", "mae"]
print("Best params:", best_params)

# train for best params again to get metrics and save model
bst = xgb.train(
    best_params,
    train_data,
    100,
    [(train_data, "train"), (test_data, "eval")],
    early_stopping_rounds=10,
)
bst.save_model("best_model.model")
