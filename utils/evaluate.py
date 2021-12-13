import numpy as np
import sklearn.metrics as skm

def metrics(y_true, y_pred):
    """Compute evaluation metrics."""
    mse = skm.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = skm.mean_absolute_error(y_true, y_pred)
    mape = skm.mean_absolute_percentage_error(y_true, y_pred)
    r2 = skm.r2_score(y_true, y_pred)

    return mape, mae, mse, rmse, r2


def cross_validate(X, y, cv, model):
    """Evaluate model through cross-validation."""
    val_train_scores, val_test_scores = [], []
    for train_idxs, test_idxs in cv.split(X, y):
        # Fit model
        model.fit(X[train_idxs], y[train_idxs])
        # Evaluate on train
        y_train_pred = model.predict(X[train_idxs])
        val_train_scores.append(metrics(y[train_idxs], y_train_pred))
        # Evaluate on test
        y_test_pred = model.predict(X[test_idxs])
        val_test_scores.append(metrics(y[test_idxs], y_test_pred))

    val_train_scores = np.asarray(val_train_scores).mean(axis=0)
    val_test_scores = np.asarray(val_test_scores).mean(axis=0)

    return model, val_train_scores, val_test_scores
