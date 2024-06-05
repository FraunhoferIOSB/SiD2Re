"""Standard and specific error metrics for applications on non-stationary environments."""
from typing import Optional

import numpy as np
from numpy import typing as npt
from sid2re.driftgenerator.utils.type_aliases import Metric, MetrOut, NumberArray


def error(
    y_pred: NumberArray,
    y_truth: NumberArray,
    averaging: Optional[bool] = False,
) -> NumberArray:
    """Calculate the elementwise error for a given prediction and the ground truth labels.

    Parameters
    ----------
    y_pred : NumberArray
        Of shape (n_samples,n_labels).
    y_truth : NumberArray
        Of shape (n_samples,n_labels).
    averaging : Optional[bool]
        Whether to average over the label dimensions.

    Returns
    -------
    NumberArray
        Of shape (n_samples,n_labels) or (n_samples)

    Raises
    ------
    ValueError
        If wrong input types are passed.

    """
    if not (isinstance(y_pred, (np.ndarray, np.generic)) and isinstance(y_truth, (np.ndarray, np.generic))):
        raise ValueError(
            f'Tried to compute metric for input instances {type(y_pred)} and {type(y_truth)}.'
            + ' Please only use np.ndarray.',
        )
    if averaging:
        return np.asarray(np.mean(np.subtract(y_truth, y_pred), axis=1), dtype=np.float64)
    return np.subtract(y_truth, y_pred)


def ae(
    y_pred: NumberArray,
    y_truth: NumberArray,
    averaging: Optional[bool] = False,
) -> npt.NDArray[np.float64]:
    """Calculate the elementwise absolute error for a given prediction and the ground truth labels.

    Parameters
    ----------
    y_pred : NumberArray
        Of shape (n_samples,n_labels).
    y_truth : NumberArray
        Of shape (n_samples,n_labels).
    averaging : Optional[bool]
        Whether to average over the label dimensions.

    Returns
    -------
    npt.NDArray[np.float64]
        Of shape (n_samples,n_labels) or (n_samples)
    """
    if averaging:
        return np.asarray(np.mean(np.abs(error(y_pred, y_truth)), axis=1), dtype=np.float64)
    return np.asarray(np.abs(error(y_pred, y_truth)), dtype=np.float64)


def mae(
    y_pred: NumberArray,
    y_truth: NumberArray,
    averaging: Optional[bool] = False,
) -> MetrOut:
    """Calculate the mean absolute error for a given prediction and the ground truth labels.

    Parameters
    ----------
    y_pred : NumberArray
        Of shape (n_samples,n_labels).
    y_truth : NumberArray
        Of shape (n_samples,n_labels).
    averaging : Optional[bool]
        Whether to average over the label dimensions.

    Returns
    -------
    MetrOut
        Of shape (n_labels) or float
    """
    if averaging:
        return float(np.mean(ae(y_pred, y_truth)))
    return np.asarray(np.mean(ae(y_pred, y_truth), axis=0), dtype=np.float64)


def mse(
    y_pred: NumberArray,
    y_truth: NumberArray,
    averaging: Optional[bool] = False,
) -> MetrOut:
    """Calculate the mean squared error for a given prediction and the ground truth labels.

    Parameters
    ----------
    y_pred : NumberArray
        Of shape (n_samples,n_labels).
    y_truth : NumberArray
        Of shape (n_samples,n_labels).
    averaging : Optional[bool]
        Whether to average over the label dimensions.

    Returns
    -------
    MetrOut
        Of shape (n_labels) or float
    """
    if averaging:
        return float(np.mean(np.square(error(y_pred, y_truth))))
    return np.asarray(np.mean(np.square(error(y_pred, y_truth)), axis=0), dtype=np.float64)


def rmse(
    y_pred: NumberArray,
    y_truth: NumberArray,
    averaging: Optional[bool] = False,
) -> MetrOut:
    """Calculate the root mean squared error for a given prediction and the ground truth labels.

    Parameters
    ----------
    y_pred : NumberArray
        Of shape (n_samples,n_labels).
    y_truth : NumberArray
        Of shape (n_samples,n_labels).
    averaging : Optional[bool]
        Whether to average over the label dimensions.

    Returns
    -------
    MetrOut
        Of shape (n_labels) or float
    """
    if averaging:
        return float(np.mean(np.sqrt(mse(y_pred, y_truth))))
    return np.asarray(np.sqrt(mse(y_pred, y_truth)), dtype=np.float64)


def ape(
    y_pred: NumberArray,
    y_truth: NumberArray,
    averaging: Optional[bool] = False,
) -> npt.NDArray[np.float64]:
    """Calculate the absolute percentage error for a given prediction and the ground truth labels.

    Parameters
    ----------
    y_pred : NumberArray
        Of shape (n_samples,n_labels).
    y_truth : NumberArray
        Of shape (n_samples,n_labels).
    averaging : Optional[bool]
        Whether to average over the label dimensions.

    Returns
    -------
    npt.NDArray[np.float64]
        Of shape (n_samples,n_labels) or (n_samples)
    """
    epsilon = np.finfo(np.float64).eps
    if averaging:
        return np.asarray(np.mean((ae(y_pred, y_truth) / np.maximum(np.abs(y_truth), epsilon)), axis=1))
    return np.asarray(ae(y_pred, y_truth) / np.maximum(np.abs(y_truth), epsilon))


def mape(
    y_pred: NumberArray,
    y_truth: NumberArray,
    averaging: Optional[bool] = False,
) -> MetrOut:
    """Calculate the mean absolute percentage error for a given prediction and the ground truth labels.

    Parameters
    ----------
    y_pred : NumberArray
        Of shape (n_samples,n_labels).
    y_truth : NumberArray
        Of shape (n_samples,n_labels).
    averaging : Optional[bool]
        Whether to average over the label dimensions.

    Returns
    -------
    MetrOut
        Of shape (n_labels) or float
    """
    if averaging:
        return float(np.mean(ape(y_pred, y_truth)))
    return np.asarray(np.mean(ape(y_pred, y_truth), axis=0), dtype=np.float64)


def bias(
    y_pred: NumberArray,
    y_truth: NumberArray,
    averaging: Optional[bool] = False,
) -> MetrOut:
    """Calculate the bias error, as the sum of all errors, for a given prediction and the ground truth labels.

    Parameters
    ----------
    y_pred : NumberArray
        Of shape (n_samples,n_labels).
    y_truth : NumberArray
        Of shape (n_samples,n_labels).
    averaging : Optional[bool]
        Whether to average over the label dimensions.

    Returns
    -------
    MetrOut
        Of shape (n_labels) or float
    """
    if averaging:
        return float(np.mean(np.sum(error(y_pred, y_truth), axis=0)))
    return np.asarray(np.sum(error(y_pred, y_truth), axis=0), dtype=np.float64)


def drift_correction_error(
    y_pred: NumberArray,
    y_truth: NumberArray,
    local_error: Metric = ae,
    averaging: Optional[bool] = False,
) -> MetrOut:
    """Calculate the error for drift correction methods, as the sum of prediction errors-min(prediction error).

    As this metric tries to evaluate the effectiveness of the drift adaption the best case performance, that is mostly
    dependent on the used prediction model, we only evaluate how much we deviate from the best case/stationary concept
    prediction.

    Parameters
    ----------
    y_pred : NumberArray
        Of shape (n_samples,n_labels).
    y_truth : NumberArray
        Of shape (n_samples,n_labels).
    local_error : Metric
        Internal metric on which basis to calculate the prediction error and also minimal achieved prediction error.
    averaging : Optional[bool]
        Whether to average over the label dimensions.

    Returns
    -------
    MetrOut
        Of shape (n_labels) or float
    """
    e_t = local_error(y_pred, y_truth, False)

    if averaging:
        return float(np.mean(np.sum(e_t - np.amin(e_t, axis=0), axis=0)))
    return np.asarray(np.sum(e_t - np.amin(e_t, axis=0), axis=0), dtype=np.float64)


def drift_reference_error(
    y_pred: NumberArray,
    y_truth: NumberArray,
    y_ref: NumberArray,
    local_error: Metric = ae,
    averaging: Optional[bool] = False,
) -> MetrOut:
    """Calculate the cumulative difference in performance of a drift adaption model and a reference model.

    If this error is negative the drift adaption model has better performance on average.

    Parameters
    ----------
    y_pred : NumberArray
        Of shape (n_samples,n_labels). Predictions of the base model.
    y_truth : NumberArray
        Of shape (n_samples,n_labels).
    y_ref : NumberArray
        Of shape (n_samples,n_labels). Prediction of the reference model.
    local_error : Metric
        Internal metric on which basis to calculate the prediction error and also minimal achieved prediction error.
    averaging : Optional[bool]
        Whether to average over the label dimensions.

    Returns
    -------
    MetrOut
        Of shape (n_labels) or float
    """
    if averaging:
        return float(
            np.mean(
                np.sum(np.subtract(local_error(y_pred, y_truth, False), local_error(y_ref, y_truth, False)), axis=0),
            ),
        )
    return np.asarray(
        np.sum(np.subtract(local_error(y_pred, y_truth, False), local_error(y_ref, y_truth, False)), axis=0),
        dtype=np.float64,
    )
