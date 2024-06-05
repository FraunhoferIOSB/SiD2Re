from .utils import get_random_model
import numpy as np
import random
import pandas as pd


class RegressionConceptModel:
    """
    Concept representation used in the data generator to map inputs to artificial regression targets.

    Beware might be non-stationary.
    """

    def __init__(self, number_of_models=5, number_of_features=2, number_of_outputs=1, shift_stamps=None,
                 drift_blocking_mode=True, drift_spacing=None, max_severity=1, min_severity=0, drift_class=None):
        """
        Initializes a concept representation that can be used to map inputs to a artifial regression target.
        Depending on the configuration this mapping might be intentionally unstable over time.

        :param number_of_models: Number of random initialized regression models used to build the concept.
        :type number_of_models: int, default = 5

        :param number_of_features: Number of expected input dimensions.
        :type number_of_features: int, default = 2

        :param number_of_outputs: Number of produced output dimensions.
        :type number_of_outputs: int, default = 1

        :param shift_stamps: timestamps at which drifts in the concept will be located.
        :type shift_stamps: np.array of shape (n_drifts)

        :param drift_blocking_mode: Whether drifts may interfere with each other or not. If set to True, each drift
            will be assigned distinct timeframe
        :type drift_blocking_mode: bool, default = True

        :param drift_spacing: maximal duration of drifts
        :type drift_spacing: float, default = None

        :param max_severity: Internal scaling factor on how drastic drifts are allowed to be.
            Can be used to control relative intensity of drifts.
            If max_severity=min_severity , then all drifts will have comparable impacts on data distributions and
            concept.
        :type max_severity: float, default = 1.0

        :param min_severity: Internal scaling factor in how drastic drifts have to be.
        :type min_severity: float, default = 0.0

        :param drift_class: What type ('sudden','gradual','incremental',...) the generated concept drifts should have.
            If set to None, the type of will be decided for each drift at random.
        :type drift_class: str, default = None
        """
        if shift_stamps is None:
            shift_stamps = []

        collection = []
        tag_collection = []

        for i in range(number_of_models):
            model_tag, model = get_random_model(number_of_features, number_of_outputs)
            collection.append(model)
            tag_collection.append(model_tag)

        self.models = collection
        self.model_tags = tag_collection
        self.shift_stamps = shift_stamps
        self.shift_info = []
        if self.shift_stamps.size > 0:
            for i in range(self.shift_stamps.shape[0]):
                if drift_blocking_mode:
                    self.shift_info.append(
                        (random.random() / 2 * drift_spacing,
                         (np.minimum(max_severity, np.maximum(min_severity, np.random.rand(
                             number_of_models))) - 0.5) * 1000 / number_of_models,
                         random.choice(("sudden", "gradual", "incremental", "reoccurring_concept"))))
                else:
                    self.shift_info.append((random.random() * (np.max(self.shift_stamps) / 2),
                                            (np.random.randint(1) * 2 - 1)
                                            * np.minimum(max_severity,
                                                         np.maximum(min_severity, np.random.rand(
                                                             number_of_models))) * 1000 / number_of_models,
                                            random.choice(("sudden", "gradual", "incremental", "reoccurring_concept"))))
                if drift_class is not None:
                    temp_list = list(self.shift_info[-1])
                    temp_list[-1] = drift_class
                    self.shift_info[-1] = temp_list
                if "reoccurring_concept" in self.shift_info[-1][-1]:
                    selection = random.randint(1, 3)
                    temp_list = list(self.shift_info[-1])
                    if selection == 1:
                        temp_list[-1] = "reoccurring_concept_sudden"
                    elif selection == 2:
                        temp_list[-1] = "reoccurring_concept_gradual"
                    elif selection == 3:
                        temp_list[-1] = "reoccurring_concept_incremental"

                    self.shift_info[-1] = temp_list

        self.coeff = (np.random.rand(number_of_models) - 0.5) * 1000 / number_of_models
        self.number_of_targets = number_of_outputs

    def get_shift_information(self):
        """
        Provides drift information of the concept.
        :return: DataFrame with (time,radius,weight shift,class)
        :rtype: pd.DataFrame with shape (n_drifts,5)
        """
        stamp = pd.DataFrame(self.shift_stamps, columns=['time_stamp(centre)'])
        info = pd.DataFrame(self.shift_info, columns=['radius', 'shift', 'class'])
        return pd.concat([stamp, info], axis=1)

    def label(self, input, time_stamps):
        """
        Used to label inputs dependent on the time with this concept.
        As the concept might change over time the time_stamps to the inputs have to be provided
        :param input: Inputs that are to be mapped to the targets.
        :type input: np.array of shape (n_samples,feature_dim)
        :param time_stamps: Points in time of the given inputs
        :type time_stamps: np.array of shape (n_samples)
        :return: target labels to the given inputs
        :rtype: np.array of shape (n_samples,target_dim)
        """
        if len(input.shape) != 2:
            raise TypeError("Input has to be a 2D Array, please check your input:%r" % input)

        init_flag = False
        labels = None
        for (sample, stamp) in zip(input, time_stamps):
            output = np.zeros((1, self.number_of_targets))
            coeffs = self.get_running_coeff(stamp)
            for (model, coeffizient, tag) in zip(self.models, coeffs, self.model_tags):
                if (tag == "MLP" or tag == "Decision Tree") and (len(model.predict(sample.reshape(1, -1)).shape) == 1):
                    model_prediction = model.predict(sample.reshape(1, -1))[:, None]
                else:
                    model_prediction = model.predict(sample.reshape(1, -1))
                output += coeffizient * model_prediction
            if init_flag:
                labels = np.concatenate((labels, output), axis=0)
            else:
                # print(output)
                labels = output
                # print(labels)
                init_flag = True

        return labels

    def get_base_coeff(self, stamp):
        coeff = np.array(self.coeff)
        # print(stamp)
        # print(f"self:{coeff}")
        if self.shift_stamps.size > 0:
            for (shift, info) in zip(self.shift_stamps, self.shift_info):
                if shift + info[0] < stamp and "reoccurring_concept" not in info[2]:
                    coeff += info[1]
        # print(f"Base:{coeff}")
        return coeff

    def get_running_coeff(self, stamp):
        coeff = self.get_base_coeff(stamp)
        if self.shift_stamps.size > 0:
            for (shift, info) in zip(self.shift_stamps, self.shift_info):
                if shift - info[0] < stamp <= shift + info[0]:
                    if info[2] == "sudden":
                        coeff += info[1]
                    elif info[2] == "gradual":
                        prob = random.random() * (stamp - (shift - info[0])) / (info[0] * 2)
                        # print(prob)
                        flag = round(prob)
                        # print(flag)
                        coeff += flag * info[1]
                    elif info[2] == "incremental":
                        coeff += ((stamp - (shift - info[0])) / (info[0] * 2)) * info[1]
                    elif "reoccurring_concept" in info[2]:
                        if ("sudden" in info[2]):
                            coeff += info[1]
                        elif ("gradual" in info[2]):
                            step_share = (stamp - (shift - info[0])) / (info[0])
                            if step_share > 1:
                                step_share = 2 - step_share
                            prob = random.random() * step_share
                            flag = round(prob)
                            coeff += flag * info[1]
                        elif ("incremental" in info[2]):
                            step_share = (stamp - (shift - info[0])) / (info[0])
                            if step_share > 1:
                                step_share = 2 - step_share
                            coeff += step_share * info[1]
                    else:
                        print(
                            f"Error: shift with following informations: shift_mean {shift},"
                            f" Interval;new Coeff;method : {info}")
        # print(f"running:{coeff}")
        return coeff
