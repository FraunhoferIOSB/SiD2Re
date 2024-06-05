import numpy as np
import pandas
import pandas as pd

import multiprocessing
from sklearn.model_selection import ParameterGrid
import tqdm
import os
from .metrics import mae, mape, mse, rmse, bias, drift_correction_error, \
    drift_reference_error


class Benchmarker:
    def __init__(self, file_name, path, data_path):
        self.file_name = file_name
        self.data_path = data_path
        self.path = path

    def performance_test_evaluation(self, params):
        string_params = params.copy()
        string_params['learner'] = params['learner'].__name__
        string_params['det'] = params['det'].__name__

        try:
            with open(self.path + self.data_path + f"/{params['dataset']}/x.csv", "r") as f:
                X = pandas.read_csv(f)
            with open(self.path + self.data_path + f"/{params['dataset']}/y.csv", "r") as f:
                y = pandas.read_csv(f)
        except FileNotFoundError:
            raise RuntimeError(f"Dataset {params['dataset']} was not found")

        _, predictions, _ = params['det'](X, y, params['det_con'], learner=params['learner'],
                                          batches=params['batches'])
        _, ref_predictions, _ = params['ref'](X, y, [None], learner=[None],
                                              batches=params['batches'])

        predictions = np.array(predictions)
        columns = [f"label_{i}" for i in range(predictions.shape[1])]
        time_stamp = X['time_stamp'].values
        predictions = pd.DataFrame(predictions, columns=columns, index=time_stamp)
        predictions.index.name = 'time_stamp'

        ref_predictions = np.array(predictions)
        columns = [f"label_{i}" for i in range(ref_predictions.shape[1])]
        time_stamp = X['time_stamp'].values
        ref_predictions = pd.DataFrame(ref_predictions, columns=columns, index=time_stamp)
        ref_predictions.index.name = 'time_stamp'

        new_row = {'dataset': params['dataset'],
                   'params': params,
                   'detector': string_params['det'],
                   'base_learner': string_params['learner'],
                   'detector configuration': params['det_con'],
                   'batches': params['batches'],
                   'MAE': mae(predictions[columns].values, y[columns].values, averaging=True),
                   'MAPE': mape(predictions[columns].values, y[columns].values, averaging=True),
                   'MSE': mse(predictions[columns].values, y[columns].values, averaging=True),
                   'RMSE': rmse(predictions[columns].values, y[columns].values, averaging=True),
                   'BIAS': bias(predictions[columns].values, y[columns].values, averaging=True),
                   'DCE': drift_correction_error(predictions[columns].values, y[columns].values, averaging=True),
                   'DRE': drift_reference_error(predictions[columns].values, y[columns].values,
                                                ref_predictions[columns].values, averaging=True)
                   }

        return pd.DataFrame([new_row])


def benchmark(file_name, base_learners, dataset_path, detector, reference, detector_params, batches, path):
    n_of_test = len(os.listdir(path + dataset_path))

    param_grid = {'det': [detector],
                  'learner': base_learners,
                  'det_con': detector_params,
                  'batches': batches,
                  'dataset': os.listdir(path + dataset_path),
                  'ref': [reference]
                  }
    lock = multiprocessing.Lock()
    pool_obj = multiprocessing.Pool(initializer=pool_init, initargs=[lock])

    evaluator = Benchmarker(file_name, path, dataset_path)

    answer = tqdm.tqdm(pool_obj.imap_unordered(evaluator.performance_test_evaluation, list(ParameterGrid(param_grid))),
                       total=n_of_test)

    answer = pd.concat(answer)

    os.makedirs(path + '/benchmark_results', exist_ok=True)
    answer.to_csv(path + '/benchmark_results/' + file_name + ".csv", sep='#', index=False)
    return answer


def pool_init(l_lock):
    global lock
    lock = l_lock
