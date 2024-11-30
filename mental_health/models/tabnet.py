"""from collections import namedtuple

import torch
from sklearn.metrics import accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier


def train_tabnet(params: namedtuple):

    def objective(trial):
        # Hyperparameters to tune
        n_d = trial.suggest_int("n_d", 8, 64)  # Dimension of decision layer
        n_a = trial.suggest_int("n_a", 8, 64)  # Dimension of attention layer
        n_steps = trial.suggest_int("n_steps", 3, 10)  # Number of decision steps
        gamma = trial.suggest_float("gamma", 1.0, 2.0)  # Relaxation parameter
        lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3)  # Sparsity regularization
        momentum = trial.suggest_float("momentum", 0.01, 0.4)  # Batch normalization momentum
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True)  # Learning rate
        cat_idxs = [X_train.columns.get_loc(col) for col in schema.catvar_features()]
        cat_dims = [len(X_train[col].unique()) for col  in schema.catvar_features()]
        seed = config.random_state

        # Initialize the TabNet model
        model = TabNetClassifier(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            optimizer_params=dict(lr=learning_rate),
            momentum=momentum,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            verbose=1,
            seed=seed,
        )

        # Fit the model
        model.fit(
            X_train.values, y_train.values.ravel(),
            eval_set=[(X_test.values, y_test.values.ravel())],
            eval_metric=["accuracy"],
            patience=30,
            batch_size=256,
            max_epochs=200,
        )

        # Evaluate the model
        preds = model.predict(X_test.values)
        acc = accuracy_score(y_test.values, preds)
        return acc

    study = optuna.create_study(direction="maximize")  # Aim to maximize accuracy
    study.optimize(objective, n_trials=50)  # Number of trials

    # Print the best hyperparameters
    logger.info("Best trial:")
    logger.info(f"  Accuracy: {study.best_value}")
    logger.info(f"  Params: {study.best_params}")

    best_params = study.best_params

    # Train the model with the best hyperparameters
    final_model = TabNetClassifier(
        n_d=best_params["n_d"],
        n_a=best_params["n_a"],
        n_steps=best_params["n_steps"],
        gamma=best_params["gamma"],
        lambda_sparse=best_params["lambda_sparse"],
        optimizer_params=dict(lr=best_params["learning_rate"]),
        momentum=best_params["momentum"],
        verbose=1
    )

    cat_idxs = [params.X_train.columns.get_loc(col) for col in params.schema.catvar_features()]
    cat_dims = [len(params.X_train[col].unique()) for col in params.schema.catvar_features()]
    seed = params.config.random_state

    model = TabNetClassifier(
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        momentum=0.02,
        lambda_sparse=1e-3,
        seed=42,
        clip_value=1,
        verbose=1,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_fn=None,
        scheduler_params={},
        mask_type="sparsemax",
        input_dim=None,
        output_dim=None,
        device_name="auto",
        n_shared_decoder=1,
        n_indep_decoder=1,
        grouped_features=[],
    )

    model.fit(
        params.X_train.values,
        params.y_train.values.ravel(),
        eval_set=[(params.X_val.values, params.y_val.values.ravel())],
        eval_metric=["accuracy"],
        patience=30,
        batch_size=256,
        max_epochs=200,
    )

    # Evaluate the final model
    final_preds = model.predict(params.X_val.values)
    final_acc = accuracy_score(params.y_val.values, final_preds)
    params.logger.info(f"Final Accuracy: {final_acc}")
    return model
"""
