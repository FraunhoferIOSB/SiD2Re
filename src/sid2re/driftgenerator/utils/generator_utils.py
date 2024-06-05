import numpy as np
import pandas
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import multiprocessing
from sklearn.model_selection import ParameterGrid
import tqdm
import time
from ..concept.utils import get_random_model
import os
import typing


def dist_cos(w: np.ndarray, timestamp: float, minimum: float, maximum: float) -> float:
    """
    :param w: Behaviour defining vector of values between 0 and 1
        w[0]:share of the interval used for shifting the cosine along the time axis
        w[1]:defining periodical interval between 5 and 50
        w[2]:share of the [min,max] interval, used for value axis shift
        w[3]:proportional to cosine amplitudes
    :param timestamp: point in time for which the function value of the transformed cosine is asked
    :param minimum: minimal function value that is allowed to be generated
    :param maximum: maximal function value that is allowed to be generated
    :return: function value of the cosine transformed by params set in w
    """
    if not w.any():
        return 0.0
    interval = 45 * w[1] + 5
    shift = w[0] * interval
    average = w[2] * (maximum - minimum) + minimum
    amplitude = min(maximum - average, average - minimum) * w[3]

    timestamp = (timestamp + shift) * math.pi
    timestamp = timestamp % interval
    return float((np.cos(timestamp) * amplitude) + average)


def dist_linear(w: np.ndarray, timestamp: float, minimum: float, maximum: float) -> float:
    """
    :param w: Behaviour defining vector of values between 0 and 1
        w[0]:share of the interval used for shifting the function along the time axis
        w[1]:defining periodical interval between 5 and 50
        w[2]:share of the [min,max] interval, used for value axis shift
        w[3]:proportional to pitch of the linear function
        w[4]:used to determine whether the pitch of the linear function is positive or negative
    :param timestamp: point in time for which the function value of the transformed linear function is asked
    :param minimum: minimal function value that is allowed to be generated
    :param maximum: maximal function value that is allowed to be generated
    :return: function value of the by w defined linear function
    """
    if not w.any():
        return 0.0
    sign = ((math.floor(w[4] - 0.5) * 2) + 1)
    interval = 45 * w[1] + 5
    shift = w[0] * interval
    base = w[2] * (maximum - minimum) + minimum
    amplitude = (maximum - base) / interval * w[3]

    timestamp = (timestamp + shift) % interval
    if sign == -1:
        base = base + (maximum - base)
    return float(timestamp * sign * amplitude + base)


def dist_quadratic(w: np.ndarray, timestamp: float, minimum: float, maximum: float) -> float:
    """
    :param w: Behaviour defining vector of values between 0 and 1
        w[0]:share of the interval used for shifting the function along the time axis
        w[1]:defining periodical interval between 5 and 50
        w[2]:share of the [min,max] interval, used for value axis shift
        w[3]:proportional to pitch of the quadratic function
        w[4]:used to determine whether the pitch of the quadratic function is positive or negative
    :param timestamp: point in time for which the function value of the transformed quadratic function is asked
    :param minimum: minimal function value that is allowed to be generated
    :param maximum: maximal function value that is allowed to be generated
    :return: function value of the by w defined quadratic function
    """
    if not w.any():
        return 0.0
    sign = ((math.floor(w[4] - 0.5) * 2) + 1)
    interval = 45 * w[1] + 5
    shift = w[0] * interval
    base = w[2] * (maximum - minimum) + minimum
    amplitude = (maximum - base) / (interval ** 2) * w[3]

    timestamp = (timestamp + shift) % interval
    if sign == -1:
        base = base + (maximum - base)
    return float((timestamp ** 2) * sign * amplitude + base)


def dist_uniform(w: np.ndarray, timestamp: float, minimum: float, maximum: float) -> float:
    """
    :param w: Behaviour defining vector of values between 0 and 1
        w[0]:centre of the uniform distribution
        w[1]:share of [minimum,maximum] that is used as range for the uniform distribution
    :param timestamp: point in time for which the function value of the function is  asked
        (in this case not used)
    :param minimum: minimal function value that is allowed to be generated
    :param maximum: maximal function value that is allowed to be generated
    :return: random value sampled of the uniform distribution defined by w
    """
    if not w.any():
        return 0
    value = random.random() * w[1] * (maximum - minimum)
    max_val = w[1] * (maximum - minimum)
    centered_val = value - max_val / 2
    offset = (w[0] * ((maximum - max_val / 2) - (minimum + max_val / 2))) + (minimum + max_val / 2)
    return float(centered_val + offset)


def dist_gauss(w: np.ndarray, timestamp: float, minimum: float, maximum: float) -> float:
    """
    :param w: Behaviour defining vector of values between 0 and 1
        w[0]:centre of the uniform distribution
        w[1]:share of [minimum,maximum] that is used as variance for the gaussian distribution
    :param timestamp: point in time for which the function value of the function is  asked
        (in this case not used)
    :param minimum: minimal function value that is allowed to be generated
    :param maximum: maximal function value that is allowed to be generated
    :return: random value sampled of the uniform distribution defined by w
    """
    if not w.any():
        return 0.0
    value = random.gauss(w[0] * (maximum - minimum) + minimum, w[1] * (maximum - minimum) / 10)
    if value > maximum:
        value = maximum
    elif value < minimum:
        value = minimum
    return float(value)


def dist_constant(w: np.ndarray, timestamp: float, minimum: float, maximum: float) -> float:
    """
    :param w: Behaviour defining vector of values between 0 and 1
        w[0]:value in [minimum,maximum] that used as constant function value
    :param timestamp: point in time for which the function value of the function is  asked
        (in this case not used)
    :param minimum: minimal function value that is allowed to be generated
    :param maximum: maximal function value that is allowed to be generated
    :return: random value sampled of the uniform distribution defined by w
    """
    if not w.any():
        return 0.0
    value = (w[0] * (maximum - minimum)) + minimum
    return float(value)


def dist_periodical(w: np.ndarray, timestamp: float, minimum: float, maximum: float) -> float:
    """
    Decorator method to choose one of the 3 implemented periodical functions
    :param w: Behaviour defining vector of values between 0 and 1
        w[0]:deciding what periodical function is chosen
        w[1:9]:propagated to submethods
    :param timestamp: point in time for which the function value of the function is  asked
        (in this case not used)
    :param minimum: minimal function value that is allowed to be generated
    :param maximum: maximal function value that is allowed to be generated
    :return: value determined by one of the submethods
    """
    if not w.any():
        return 0.0
    selection = math.floor(3 * w[0])
    if selection == 0:
        value = dist_cos(w[1:], timestamp, minimum, maximum)
    if selection == 1:
        value = dist_linear(w[1:], timestamp, minimum, maximum)
    if selection == 2 or selection == 3:
        value = dist_quadratic(w[1:], timestamp, minimum, maximum)
    return float(value)


def linear_trans_func(current_time, shift_centre, shift_radius):
    return (current_time - (shift_centre - shift_radius)) / (shift_radius * 2)


def helper_get_feature(index: int, minimum: float, maximum: float, dist: typing.Callable, other_dist,
                       shifts: np.ndarray, shift_info: np.ndarray, time_stamps: np.ndarray,
                       transition_func: typing.Callable = linear_trans_func) -> np.ndarray:
    """
    Method to manage feature value generation and data shift realization
    :param index: which feature is to be generated
    :param minimum: minimum value allowed in feature value generation
    :param maximum: maximum value allowed in feature value generation
    :param dist: what distribution/sampling function to use to generate feature values
    :param other_dist: what other distributions might be available (changing distributions because of data shift)
    :param shifts: centeres of data shifts
    :param shift_info: information of data shifts, like radius, amplitude ...
    :param time_stamps: points in time for that feature values have to be generated
    :param transition_func: functionality used to blend between two concepts. Usually linear transitions are used, but
        this function can be defined by the user if custom transitions are needed.
    :return: ordered array of generated feature values; shape [len(time_stamps)]
    """
    sample = []
    w = np.random.rand(10)
    new_shifts = []
    new_info = []
    for i in range(shifts.size):
        if shift_info[i][0] == index:
            new_shifts.append(shifts[i])
            new_info.append(shift_info[i])
    shifts = np.array(new_shifts)
    shift_info = np.array(new_info, dtype=object)
    if shifts.size > 0:
        for stamp in time_stamps:
            coeff = get_running_coeff_data(stamp, w.copy(), shifts, shift_info, transition_func)
            if not coeff.any():
                sample.append(0)
            else:
                coeff_pos = coeff + np.min(coeff)
                coeff_normalized = coeff_pos / np.max(coeff_pos)
                sample.append(dist(coeff_normalized, stamp, minimum, maximum))
    else:
        for stamp in time_stamps:
            sample.append(dist(w, stamp, minimum, maximum))

    return np.array(sample)


def helper_get_sensor(index: int, input_feat_1, input_feat_2,
                      shifts: np.ndarray, shift_info: np.ndarray, time_stamps: np.ndarray,
                      transition_func=linear_trans_func) -> np.ndarray:
    """
    Method to manage feature value generation and data shift realization
    :param index: which feature is to be generated
    :param minimum: minimum value allowed in feature value generation
    :param maximum: maximum value allowed in feature value generation
    :param dist: what distribution/sampling function to use to generate feature values
    :param other_dist: what other distributions might be available (changing distributions because of data shift)
    :param shifts: centeres of data shifts
    :param shift_info: information of data shifts, like radius, amplitude ...
    :param time_stamps: points in time for that feature values have to be generated
    :param transition_func: functionality used to blend between two concepts. Usually linear transitions are used, but
        this function can be defined by the user if custom transitions are needed.
    :return: ordered array of generated feature values; shape [len(time_stamps)]
    """
    sample = []
    w = np.random.rand(10)
    new_shifts = []
    new_info = []
    input_feat_1 = list(input_feat_1)
    input_feat_2 = list(input_feat_2)
    for i in range(shifts.size):
        if shift_info[i][0] == index:
            new_shifts.append(shifts[i])
            new_info.append(shift_info[i])
    shifts = np.array(new_shifts)
    shift_info = np.array(new_info, dtype=object)
    _, model1 = get_random_model(2, 1)
    _, model2 = get_random_model(2, 1)
    if shifts.size > 0:
        for stamp in time_stamps:
            coeff = get_running_coeff_data(stamp, w.copy(), shifts, shift_info, transition_func)
            if not coeff.any():
                sample.append(0)
            else:
                coeff_pos = coeff + np.min(coeff)
                coeff_normalized = coeff_pos / np.max(coeff_pos)
                input = np.asarray([input_feat_1.pop(0), input_feat_2.pop(0)]).reshape(1, -1)
                sample.append(
                    coeff_normalized[0] * model1.predict(input)[0] + coeff_normalized[1] * model2.predict(input)[0])
    else:
        for stamp in time_stamps:
            input = np.asarray([input_feat_1.pop(0), input_feat_2.pop(0)]).reshape(1, -1)
            sample.append(w[0] * model1.predict(input)[0] + w[1] * model2.predict(input)[0])
    return np.array(sample)


def get_base_coeff_data(stamp, w, shift_stamps, shift_info):
    for (shift, info) in zip(shift_stamps, shift_info):
        if shift + info[1] < stamp and "reoccuring_concept" not in info[3]:
            w += info[2]
    return w


def get_running_coeff_data(stamp, w, shift_stamps, shift_info, transition_func=linear_trans_func):
    coeff = get_base_coeff_data(stamp, w.copy(), shift_stamps, shift_info)
    for (shift, info) in zip(shift_stamps, shift_info):
        if shift - info[1] < stamp <= shift + info[1]:
            if info[3] == "sudden":
                coeff += info[2]
            elif info[3] == "gradual":
                prob = random.random() * transition_func(stamp, shift, info[1])
                flag = round(prob)
                coeff += flag * info[2]
            elif info[3] == "incremental":
                coeff += transition_func(stamp, shift, info[1]) * info[2]
            elif info[3] == "faulty_sensor":
                coeff = info[2] * 0
            elif "reoccuring_concept" in info[3]:
                coeff += info[2]
            else:
                print(
                    f"Error: shift with following informations: shift_mean {shift}, Interval;new Coeff;method : {info}")
    return coeff


def display_data(data, generator=None, time_vis=False):
    """

    :param data:
    :param generator:
    :param time_vis:
    """
    data = pd.concat(data)
    concept_drifts, data_drifts = None, None
    if generator is not None:
        pd.set_option('display.max_columns', None)
        concept_drifts, data_drifts = generator.get_shift_information()
        print("concept drifts:")
        print(concept_drifts)
        print("data_drifts:")
        print(data_drifts)

    fig = plt.figure(figsize=(10, 20))
    n_figures = data.shape[1]
    special = 0
    if data.shape[1] == 4 and "feat_1" in data.columns:
        special += 1
        # special += 2
        # ax = fig.add_subplot(n_figures+special, 1, special-1, projection='3d')
        # X = np.arange(np.min(data["feat_0"]), np.max(data["feat_0"]), 0.25)
        # Y = np.arange(np.min(data["feat_1"]), np.max(data["feat_1"]), 0.25)
        # Z = np.zeros((X.shape[0],Y.shape[0]))
        # for i in range(0,X.shape[0]):
        #    for j in range(0, Y.shape[0]):
        #        Z[i][j] = generator.get_context.label([])
        # X, Y = np.meshgrid(X, Y)
        # plot = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
        #                       linewidth=0, antialiased=False)
        ax = fig.add_subplot(n_figures + special, 1, special, projection='3d', title="regression concept over time")
        if time_vis:
            ax.scatter(data['feat_0'], data['feat_1'], data['label_0'], c=data['time_stamp'], linewidth=0,
                       antialiased=False)
        else:
            ax.scatter(data['feat_0'], data['feat_1'], data['label_0'], linewidth=0,
                       antialiased=False)

    if data.shape[1] == 3:
        special += 1
        ax = fig.add_subplot(n_figures + special, 1, special, title="regression concept over time")
        ax.scatter(data['feat_0'], data['label_0'], c=data['time_stamp'])

    col_namespace = data.columns
    for i in range(1, n_figures):
        ax = fig.add_subplot(n_figures + special, 1, i + special, title=col_namespace[i])
        if 'feat' in col_namespace[i] and data_drifts is not None:
            feature_drifts = data_drifts[data_drifts['affected_feature'] == i - 1]['time_stamp(centre)']
            feature_drifts_radius = data_drifts[data_drifts['affected_feature'] == i - 1]['radius']
            if not feature_drifts.empty:
                ax.vlines(feature_drifts.values, ymin=data[col_namespace[i]].min(), ymax=data[col_namespace[i]].max())
                errorboxes = [Rectangle((x - xe, data[col_namespace[i]].min()), xe * 2,
                                        data[col_namespace[i]].max() - data[col_namespace[i]].min())
                              for x, xe in zip(feature_drifts.values, feature_drifts_radius.values)]

                # Create patch collection with specified colour/alpha
                pc = PatchCollection(errorboxes, facecolor="blue", alpha=0.2)
                ax.add_collection(pc)
        if 'label' in col_namespace[i] and concept_drifts is not None:
            label_drifts = concept_drifts['time_stamp(centre)']
            label_drifts_radius = concept_drifts['radius']
            if not label_drifts.empty:
                ax.vlines(label_drifts.values, ymin=data[col_namespace[i]].min(), ymax=data[col_namespace[i]].max())
                errorboxes = [Rectangle((x - xe, data[col_namespace[i]].min()), xe * 2,
                                        data[col_namespace[i]].max() - data[col_namespace[i]].min())
                              for x, xe in zip(label_drifts.values, label_drifts_radius.values)]

                # Create patch collection with specified colour/alpha
                pc = PatchCollection(errorboxes, facecolor="green", alpha=0.2)
                ax.add_collection(pc)

        ax.scatter(data['time_stamp'], data[col_namespace[i]])

    plt.show()


class Evaluator:
    def __init__(self, file_name, path, data_path, seed):
        self.file_name = file_name
        self.data_path = data_path
        self.path = path
        self.seed = seed

    """def timing_test_evaluation(self, params):

        res_detected = 0
        res_to_detect = 0
        res_responsetime = 0
        res_num_detected = 0
        if self.seed != -1:
            random.seed(self.seed)
            np.random.seed(self.seed)
        for i in range(params['runs']):
            generator = params['gen'](number_of_models=10, number_of_features=params['n_feat'],
                                      number_of_outputs=params['n_out'], concept_drifts=params['n_cd'],
                                      data_drifts=params['n_dd'], number_of_data_points=params['d_size'],
                                      continous_time=True, rand_seed=-1,
                                      drift_blocking_mode=True, max_severity=params['sev'][1],
                                      min_severity=params['sev'][0], concept_drift_class=params['cp_class'],
                                      data_drift_class=params['dt_class'])
            X, y = generator.get_data()
            detections, certainties = params['det'](X, y, params['det_con'], batches=params['batches'])
            detected, to_detect, num_detected, responsetime = generator.calculate_timing_score(detections, certainties)
            res_detected += detected / params['runs']
            res_to_detect += to_detect / params['runs']
            res_num_detected += num_detected / params['runs']
            res_responsetime += responsetime / params['runs']
        new_row = {'detector configuration': params['det_con'],
                   'batches': params['batches'],
                   'dataset size': params['d_size'],
                   'number of features': params['n_feat'],
                   'number of outputs': params['n_out'],
                   '[min,max] severity': params['sev'],
                   'number of induced concept drifts': params['n_cd'],
                   'number of induced data drifts': params['n_dd'],
                   'fixed data drift class': params['dt_class'],
                   'fixed concept drift class': params['cp_class'],
                   '(Avg) number of detected drifts': res_detected,
                   '(Avg) present drifts': res_to_detect,
                   '(Avg) number of detections': num_detected,
                   '(Avg) responsetime [if drift was detected]': res_responsetime}
        return new_row"""

    def performance_test_evaluation(self, params):
        global checkpoint
        string_params = params.copy()
        string_params['learner'] = params['learner'].__name__
        string_params['det'] = params['det'].__name__
        string_params['gen'] = params['gen'].__name__
        """if isinstance(checkpoint, pd.DataFrame):
            temp = checkpoint[checkpoint['params'].astype(str) == str(string_params)]
            if len(temp) == 1:
                return temp
            if len(temp) > 1:
                raise RuntimeError(f"database not valid, duplicates are present : {temp['params'].values[0]} and"
                f"{temp['params'].values[1]}")"""

        for i in range(params['runs']):
            # Reseting random index for data generation
            if self.seed != -1:
                random.seed(self.seed + i)
                np.random.seed(self.seed + i)

            # saving and loading datasets
            X = None
            y = None
            predictions = None
            concept_shift_information = None
            data_shift_information = None
            data_identifier = ""
            method_identifier = ""
            for idx in ['n_feat', 'n_out', 'n_cd', 'n_dd', 'd_size', 'sev', 'cp_class', 'dt_class']:
                data_identifier = data_identifier + f"_{params[idx]}"
            for idx in string_params:
                method_identifier = method_identifier + f"_{string_params[idx]}"
            try:
                with open(self.data_path + "/results/" + method_identifier + f'/{self.seed + i}.csv', "r") as f:
                    predictions = pandas.read_csv(f)
            finally:
                pass
            try:
                with open(self.data_path + "/datasets/" + data_identifier + f'/{self.seed + i}/x.csv', "r") as f:
                    X = pandas.read_csv(f)
                with open(self.data_path + "/datasets/" + data_identifier + f'/{self.seed + i}/y.csv', "r") as f:
                    y = pandas.read_csv(f)
                with open(self.data_path + "/datasets/" + data_identifier + f'/{self.seed + i}/cd.csv', "r") as f:
                    concept_shift_information = pandas.read_csv(f)
                with open(self.data_path + "/datasets/" + data_identifier + f'/{self.seed + i}/dd.csv', "r") as f:
                    data_shift_information = pandas.read_csv(f)
            finally:
                pass
            # If loading not sucessfull, generate data
            if X is None or y is None or concept_shift_information is None or data_shift_information is None:
                generator = params['gen'](number_of_models=10, number_of_features=params['n_feat'],
                                          number_of_outputs=params['n_out'], concept_drifts=params['n_cd'],
                                          data_drifts=params['n_dd'], number_of_data_points=params['d_size'],
                                          continous_time=True, rand_seed=-1,
                                          drift_blocking_mode=True, max_severity=params['sev'][1],
                                          min_severity=params['sev'][0], concept_drift_class=params['cp_class'],
                                          data_drift_class=params['dt_class'])
                X, y = generator.get_data()
                concept_shift_information, data_shift_information = generator.get_shift_information()
                lock.acquire()
                os.makedirs(self.data_path + "/datasets/" + data_identifier + f'/{self.seed + i}', exist_ok=True)
                with open(self.data_path + "/datasets/" + data_identifier + f'/{self.seed + i}/x.csv', "w") as f:
                    try:
                        X.to_csv(f, index=False)
                        lock.release()
                    finally:
                        lock.release()
                lock.acquire()
                with open(self.data_path + "/datasets/" + data_identifier + f'/{self.seed + i}/y.csv', "w") as f:
                    try:
                        y.to_csv(f, index=False)
                        lock.release()
                    finally:
                        lock.release()
                lock.acquire()
                with open(self.data_path + "/datasets/" + data_identifier + f'/{self.seed + i}/cd.csv', "w") as f:
                    try:
                        concept_shift_information.to_csv(f, index=False)
                        lock.release()
                    finally:
                        lock.release()
                lock.acquire()
                with open(self.data_path + "/datasets/" + data_identifier + f'/{self.seed + i}/dd.csv', "w") as f:
                    try:
                        data_shift_information.to_csv(f, index=False)
                        lock.release()
                    finally:
                        lock.release()
            if predictions is None:
                os.makedirs(self.data_path + "/results/" + method_identifier, exist_ok=True)
                _, predictions, _ = params['det'](X, y, params['det_con'], learner=params['learner'],
                                                  batches=params['batches'])
                lock.acquire()
                with open(self.data_path + "/results/" + method_identifier + f'/{self.seed + i}.csv', "w") as f:
                    try:
                        predictions = np.array(predictions)
                        columns = [f"pred_label_{i}" for i in range(predictions.shape[1])]
                        time_stamp = X['time_stamp'].values
                        prediction_df = pd.DataFrame(predictions, columns=columns, index=time_stamp)
                        prediction_df.index.name = 'time_stamp'
                        prediction_df.to_csv(f)
                        lock.release()
                    finally:
                        lock.release()

        params['learner'] = params['learner'].__name__
        params['det'] = params['det'].__name__
        params['gen'] = params['gen'].__name__
        new_row = {'params': params,
                   'base_learner': params['learner'],
                   'detector configuration': params['det_con'],
                   'batches': params['batches'],
                   'dataset size': params['d_size'],
                   'number of features': params['n_feat'],
                   'number of outputs': params['n_out'],
                   '[min,max] severity': params['sev'],
                   'number of induced concept drifts': params['n_cd'],
                   'number of induced data drifts': params['n_dd'],
                   'fixed data drift class': params['dt_class'],
                   'fixed concept drift class': params['cp_class'],
                   'Used Dataset Identifier': data_identifier,
                   'Prediction Location': method_identifier
                   }

        lock.acquire()
        df_dictionary = pd.DataFrame([new_row])
        if checkpoint is not None:
            checkpoint = pd.concat([checkpoint, df_dictionary], ignore_index=True)
        else:
            checkpoint = df_dictionary
        os.makedirs(self.path + '/temp', exist_ok=True)
        with open(self.path + '/temp/' + f'{self.file_name}_temp.csv', "w") as f:
            try:
                checkpoint.to_csv(f, sep='#', index=False)
                lock.release()
            except PermissionError:
                lock.release()
                print("Error: currently unavailable")
                time.sleep(100)
                lock.acquire()
                try:
                    checkpoint.to_csv(f, sep='#', index=False)
                    lock.release()
                except PermissionError:
                    lock.release()
                    print("Still unavailable, skipping save")

        return pd.DataFrame([new_row])


def testsuit(file_name, Generator, base_learners, data_path, detector_func, n_runs_for_config, dataset_sizes,
             severities, n_induced_concept_drifts,
             n_induced_data_drifts,
             n_features, n_outputs, detector_params, concept_drift_class, data_drift_class, rand_seed, batches, path,
             mode="timing",
             checkpoint=None):
    variable_iterations = [dataset_sizes, severities, n_induced_concept_drifts, n_induced_data_drifts, n_features,
                           n_outputs, detector_params, concept_drift_class, data_drift_class, batches, base_learners]
    n_of_test = np.sum([len(d) for d in variable_iterations])
    if isinstance(checkpoint, pd.DataFrame):
        print(f"Already checked configurations:{len(checkpoint)}/{n_of_test}")
    param_grid = {'gen': [Generator],
                  'det': [detector_func],
                  'runs': [n_runs_for_config],
                  'learner': base_learners,
                  'det_con': detector_params,
                  'd_size': dataset_sizes,
                  'n_feat': n_features,
                  'n_out': n_outputs,
                  'sev': severities,
                  'n_cd': n_induced_concept_drifts,
                  'n_dd': n_induced_data_drifts,
                  'cp_class': concept_drift_class,
                  'dt_class': data_drift_class,
                  'batches': batches,
                  }
    lock = multiprocessing.Lock()
    pool_obj = multiprocessing.Pool(initializer=pool_init, initargs=(lock, checkpoint))

    evaluator = Evaluator(file_name, path, data_path, rand_seed)

    if mode == "timing":
        answer = tqdm.tqdm(pool_obj.imap(evaluator.timing_test_evaluation, list(ParameterGrid(param_grid))),
                           total=n_of_test)
    elif mode == "performance":
        answer = tqdm.tqdm(
            pool_obj.imap_unordered(evaluator.performance_test_evaluation, list(ParameterGrid(param_grid))),
            total=n_of_test)
    answer = pd.concat(answer)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #    print(answer)
    os.makedirs(path + '/results', exist_ok=True)
    answer.to_csv(path + '/results/' + file_name + ".csv", sep='#', index=False)
    print(pd.read_csv(path + '/results/' + file_name + ".csv", sep='#'))
    return answer


def pool_init(l_lock, c):
    global lock
    lock = l_lock
    global checkpoint
    checkpoint = c
