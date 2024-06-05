import numpy as np


def error(y_pred, y_truth):
    """
    Calculates the elementwise error for a given prediction and the ground truth labels.

    :param y_pred: prediction
    :type y_pred: np.array (n_samples,n_dim)
    :param y_truth: ground truth labels
    :type y_truth: np.array(n_samples,n_dim)
    :return: error
    :rtype: np.array (n_samples,n_dim)
    """
    if not (isinstance(y_pred, (np.ndarray, np.generic)) and isinstance(y_truth, (np.ndarray, np.generic))):
        raise RuntimeError(
            f"Tried to compute metric for input instances {type(y_pred)} and {type(y_truth)}."
            f" Please only use np.array.")
    return y_truth - y_pred


def ae(y_pred, y_truth):
    """
    Calculates the elementwise absolute error for a given prediction and the ground truth labels.

    :param y_pred: prediction
    :type y_pred: np.array (n_samples,n_dim)
    :param y_truth: ground truth labels
    :type y_truth: np.array(n_samples,n_dim)
    :return: absolute error
    :rtype: np.array (n_samples,n_dim)
    """
    return np.abs(error(y_pred, y_truth))


def mae(y_pred, y_truth, averaging=False):
    """
    Calculates the mean absolute error for a given prediction and the ground truth labels.

    :param y_pred: prediction
    :type y_pred: np.array (n_samples,n_dim)
    :param y_truth: ground truth labels
    :type y_truth: np.array (n_samples,n_dim)
    :return: the mean absolute error
    :rtype: np.array (n_dim)
    """
    if averaging:
        return np.mean(ae(y_pred, y_truth))
    return np.mean(ae(y_pred, y_truth), axis=0)


def mse(y_pred, y_truth, averaging=False):
    """
    Calculates the mean squared error for a given prediction and the ground truth labels.

    :param y_pred: prediction
    :type y_pred: np.array (n_samples,n_dim)
    :param y_truth: ground truth labels
    :type y_truth: np.array (n_samples,n_dim)
    :return: the mean squared error
    :rtype: np.array (n_dim)
    """
    if averaging:
        return np.mean(np.square(error(y_pred, y_truth)))
    return np.mean(np.square(error(y_pred, y_truth)), axis=0)


def rmse(y_pred, y_truth, averaging=False):
    """
    Calculates the root mean squared error for a given prediction and the ground truth labels.

    :param y_pred: prediction
    :type y_pred: np.array (n_samples,n_dim)
    :param y_truth: ground truth labels
    :type y_truth: np.array (n_samples,n_dim)
    :return: the root mean squared error
    :rtype: np.array (n_dim)
    """
    return np.sqrt(mse(y_pred, y_truth, averaging=averaging))


def mape(y_pred, y_truth, averaging=False):
    """
    Calculates the root mean squared error for a given prediction and the ground truth labels.

    :param y_pred: prediction
    :type y_pred: np.array (n_samples,n_dim)
    :param y_truth: ground truth labels
    :type y_truth: np.array (n_samples,n_dim)
    :return: the root mean squared error
    :rtype: np.array (n_dim)
    """
    if averaging:
        return np.mean((ae(y_pred, y_truth) / y_truth) * 100)
    return np.mean((ae(y_pred, y_truth) / y_truth) * 100, axis=0)


def bias(y_pred, y_truth, averaging=False):
    """
    Calculates the bias error, as the sum of all errors, for a given prediction and the ground truth labels.

    :param y_pred: prediction
    :type y_pred: np.array (n_samples,n_dim)
    :param y_truth: ground truth labels
    :type y_truth: np.array (n_samples,n_dim)
    :return: bias error
    :rtype: np.array (n_dim)
    """
    if averaging:
        return np.mean(np.sum(error(y_pred, y_truth), axis=0))
    return np.sum(error(y_pred, y_truth), axis=0)


def drift_correction_error(y_pred, y_truth, local_error=ae, averaging=False):
    """
    Calculates the error for drift correction methods, as the the sum of prediction errors-min(prediction error).
    As this metric tries to evaluate the effectiveness of the drift adaption the best case performance, that is mostly
    dependend on the used prediction model, we only evaluate how much we deviate from the best case/stationary concept
    prediction.

    :param y_pred: prediction
    :type y_pred: np.array (n_samples,n_dim)
    :param y_truth: ground truth labels
    :type y_truth: np.array (n_samples,n_dim)
    :param local_error: callable of sid2re.driftgenerator.utils.metrics
    :return: drift correction error
    :rtype: np.array (n_dim)
    """
    e_t = local_error(y_pred, y_truth)

    if averaging:
        return np.mean(np.sum(e_t - np.amin(e_t, axis=0), axis=0))
    return np.sum(e_t - np.amin(e_t, axis=0), axis=0)


def drift_reference_error(y_pred, y_truth, y_ref, local_error=ae, averaging=False):
    """
    Calculate the cumulative difference in performance of a drift adaption model and a refference model.
    If this error is negative the drift adaption model has better performance on average.

    :param y_pred: prediction
    :type y_pred: np.array (n_samples,n_dim)
    :param y_truth: ground truth labels
    :type y_truth: np.array (n_samples,n_dim)
    :param y_ref: prediction of the refference method, for example the base learner monitored by
    the drift adaption method
    :type y_ref: np.array (n_samples,n_dim)
    :param local_error: callable of sid2re.driftgenerator.utils.metrics
    :return: drift refference error
    :rtype: np.array (n_dim)
    """
    if averaging:
        return np.mean(np.sum(local_error(y_pred, y_truth) - local_error(y_ref, y_truth), axis=0))
    return np.sum(local_error(y_pred, y_truth) - local_error(y_ref, y_truth), axis=0)
