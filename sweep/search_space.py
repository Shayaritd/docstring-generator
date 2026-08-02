"""
Defines the hyperparameter search space for the Optuna sweep.

All six dimensions use suggest_categorical rather than suggest_float/suggest_int
ranges, because the brief specifies fixed candidate lists (not continuous
ranges) for each hyperparameter. Total grid size if searched exhaustively:
3^6 = 729 combinations — Optuna's default TPE sampler explores this space
intelligently rather than requiring exhaustive search.
"""

SEARCH_SPACE = {
    "lora_r": [8, 16, 32],
    "learning_rate": [1e-4, 2e-4, 5e-4],
    "lora_alpha": [16, 32, 64],
    "lora_dropout": [0.0, 0.05, 0.1],
    "num_train_epochs": [2, 3, 5],
    "per_device_train_batch_size": [4, 8, 16],
}


def suggest_hyperparameters(trial) -> dict:
    """Sample one hyperparameter combination for a given Optuna trial."""
    return {
        "lora_r": trial.suggest_categorical("lora_r", SEARCH_SPACE["lora_r"]),
        "learning_rate": trial.suggest_categorical("learning_rate", SEARCH_SPACE["learning_rate"]),
        "lora_alpha": trial.suggest_categorical("lora_alpha", SEARCH_SPACE["lora_alpha"]),
        "lora_dropout": trial.suggest_categorical("lora_dropout", SEARCH_SPACE["lora_dropout"]),
        "num_train_epochs": trial.suggest_categorical("num_train_epochs", SEARCH_SPACE["num_train_epochs"]),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", SEARCH_SPACE["per_device_train_batch_size"]
        ),
    }


def merge_into_config(base_config: dict, sampled_params: dict) -> dict:
    import copy
    config = copy.deepcopy(base_config)

    config["lora"]["r"] = sampled_params["lora_r"]
    config["lora"]["alpha"] = sampled_params["lora_alpha"]
    config["lora"]["dropout"] = sampled_params["lora_dropout"]
    config["training"]["learning_rate"] = sampled_params["learning_rate"]
    config["training"]["num_train_epochs"] = sampled_params["num_train_epochs"]
    config["training"]["per_device_train_batch_size"] = sampled_params["per_device_train_batch_size"]

    return config
