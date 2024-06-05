from sid2re.driftgenerator.utils.metrics import error, ae, mae, mse, rmse, mape, bias, drift_correction_error, \
    drift_reference_error
import numpy as np
import pytest


def test_formatting_no_averaging():
    formats = [(1, 1), (2, 1), (10, 1), (10, 2), (10, 5)]
    for form in formats:
        pred = np.random.rand(*form)
        truth = np.random.rand(*form)
        ref = np.random.rand(*form)
        for met in [error, ae]:
            print(met)
            res = met(pred, truth)
            assert res.shape == form
        for met in [mae, mse, rmse, mape, bias, drift_correction_error]:
            print(met)
            res = met(pred, truth)
            assert len(res.shape) == 1
            assert res.shape[0] == form[1]
        for met in [drift_reference_error]:
            print(met)
            res = met(pred, truth, ref)
            assert len(res.shape) == 1
            assert res.shape[0] == form[1]


@pytest.mark.parametrize('metric', [error, ae, mae, mse, rmse, mape, bias, drift_correction_error])
def test_metrics_enforce_type_error(metric):
    with pytest.raises(ValueError) as excinfo:
        _ = metric([1, 2, 3, 4], [5, 6, 7, 8])
    assert len(str(excinfo.value)) > 0


def test_formatting_averaging():
    formats = [(1, 1), (2, 1), (10, 1), (10, 2), (10, 5)]
    for form in formats:
        pred = np.random.rand(*form)
        truth = np.random.rand(*form)
        ref = np.random.rand(*form)
        for met in [mae, mse, rmse, mape, bias, drift_correction_error]:
            print(met)
            res = met(pred, truth, averaging=True)
            assert isinstance(res, float)
        for met in [drift_reference_error]:
            print(met)
            res = met(pred, truth, ref, averaging=True)
            assert isinstance(res, float)
