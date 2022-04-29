import sys
import os
import numpy as np
import argparse
import json
import math

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf
from scipy.spatial.distance import mahalanobis

class TriggerGenerationDiscretizer:
    def __init__(self, timestep=0.8, store_masks=True, impute_strategy='zero', start_time='zero',
                 config_path=os.path.join(os.path.dirname(__file__), '../../resources/discretizer_config.json')):

        with open(config_path) as f:
            config = json.load(f)
            self._id_to_channel = config['id_to_channel']
            self._channel_to_id = dict(zip(self._id_to_channel, range(len(self._id_to_channel))))
            self._is_categorical_channel = config['is_categorical_channel']
            self._possible_values = config['possible_values']
            self._normal_values = config['normal_values']

        self._header = ["Hours"] + self._id_to_channel
        self._timestep = timestep
        self._store_masks = store_masks
        self._start_time = start_time
        self._impute_strategy = impute_strategy

        # for statistics
        self._done_count = 0
        self._empty_bins_sum = 0
        self._unused_data_sum = 0

        # Byunggill
        self._channel_value_to_integer = {}
        for k in self._is_categorical_channel:
            self._channel_value_to_integer[k] = dict(zip(self._possible_values[k], list(range(len(self._possible_values[k])))))
        

    def transform(self, X, header=None, end=None):
        if header is None:
            header = self._header
        assert header[0] == "Hours"
        eps = 1e-6

        N_channels = len(self._id_to_channel)
        ts = [float(row[0]) for row in X]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i+1] + eps

        if self._start_time == 'relative':
            first_time = ts[0]
        elif self._start_time == 'zero':
            first_time = 0
        else:
            raise ValueError("start_time is invalid")

        if end is None:
            max_hours = max(ts) - first_time
        else:
            max_hours = end - first_time

        N_bins = int(max_hours / self._timestep + 1.0 - eps)

        cur_len = 0
        begin_pos = [0 for i in range(N_channels)]
        end_pos = [0 for i in range(N_channels)]
        
        for i in range(N_channels):
            channel = self._id_to_channel[i]
            begin_pos[i] = cur_len
            """
            if self._is_categorical_channel[channel]:
                end_pos[i] = begin_pos[i] + len(self._possible_values[channel])
            else:
                end_pos[i] = begin_pos[i] + 1
            cur_len = end_pos[i]
            """
            # Byunggill
            end_pos[i] = begin_pos[i] + 1
            cur_len = end_pos[i]

        data = np.zeros(shape=(N_bins, cur_len), dtype=float)
        mask = np.zeros(shape=(N_bins, N_channels), dtype=int)
        original_value = [["" for j in range(N_channels)] for i in range(N_bins)]
        total_data = 0
        unused_data = 0
        def write(data, bin_id, channel, value, begin_pos):
            channel_id = self._channel_to_id[channel]
            """
            if self._is_categorical_channel[channel]:
                category_id = self._possible_values[channel].index(value)
                N_values = len(self._possible_values[channel])
                one_hot = np.zeros((N_values,))
                one_hot[category_id] = 1
                for pos in range(N_values):
                    data[bin_id, begin_pos[channel_id] + pos] = one_hot[pos]
            else:
                data[bin_id, begin_pos[channel_id]] = float(value)
            """
            # Byunggill
            try:
                data[bin_id, begin_pos[channel_id]] = float(value)
            except ValueError:
                data[bin_id, begin_pos[channel_id]] = float(self._channel_value_to_integer[self._id_to_channel[channel_id]][value])

        for row in X:
            t = float(row[0]) - first_time
            if t > max_hours + eps:
                continue
            bin_id = int(t / self._timestep - eps)
            assert 0 <= bin_id < N_bins

            for j in range(1, len(row)):
                if row[j] == "":
                    continue
                channel = header[j]
                channel_id = self._channel_to_id[channel]

                total_data += 1
                if mask[bin_id][channel_id] == 1:
                    unused_data += 1
                mask[bin_id][channel_id] = 1
                
                write(data, bin_id, channel, row[j], begin_pos)
                original_value[bin_id][channel_id] = row[j]

                # if channel_id == 0:
                #     print("feature 0 name:", channel, "value:", row[j])
                #     print("possible values:", self._possible_values[channel])
                #     print("normal:", self._normal_values[channel])

        # impute missing values
        if self._impute_strategy not in ['zero', 'normal_value', 'previous', 'next']:
            raise ValueError("impute strategy is invalid")

        if self._impute_strategy in ['normal_value', 'previous']:
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if self._impute_strategy == 'normal_value':
                        imputed_value = self._normal_values[channel]
                    if self._impute_strategy == 'previous':
                        if len(prev_values[channel_id]) == 0:
                            imputed_value = self._normal_values[channel]
                        else:
                            imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos)

        if self._impute_strategy == 'next':
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins-1, -1, -1):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if len(prev_values[channel_id]) == 0:
                        imputed_value = self._normal_values[channel]
                    else:
                        imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos)

        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(N_bins)])
        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps)
        self._unused_data_sum += unused_data / (total_data + eps)

        if self._store_masks:
            data = np.hstack([data, mask.astype(np.float32)])

        # create new header
        new_header = []
        for channel in self._id_to_channel:
            if self._is_categorical_channel[channel]:
                values = self._possible_values[channel]
                for value in values:
                    new_header.append(channel + "->" + value)
            else:
                new_header.append(channel)

        if self._store_masks:
            for i in range(len(self._id_to_channel)):
                channel = self._id_to_channel[i]
                new_header.append("mask->" + channel)

        new_header = ",".join(new_header)

        return (data, new_header)

    def print_statistics(self):
        print("statistics of discretizer:")
        print("\tconverted {} examples".format(self._done_count))
        print("\taverage unused data = {:.2f} percent".format(100.0 * self._unused_data_sum / self._done_count))
        print("\taverage empty  bins = {:.2f} percent".format(100.0 * self._empty_bins_sum / self._done_count))


class PoisoningDiscretizer:
    def __init__(self, timestep=0.8, store_masks=True, impute_strategy='zero', start_time='zero',
                 config_path=os.path.join(os.path.dirname(__file__), '../../resources/discretizer_config.json'),
                 poisoning_trigger=None, one_hot=True, do_impute=True):

        with open(config_path) as f:
            config = json.load(f)
            self._id_to_channel = config['id_to_channel']
            self._channel_to_id = dict(zip(self._id_to_channel, range(len(self._id_to_channel))))
            self._is_categorical_channel = config['is_categorical_channel']
            self._possible_values = config['possible_values']
            self._normal_values = config['normal_values']

        self._header = ["Hours"] + self._id_to_channel
        self._timestep = timestep
        self._store_masks = store_masks
        self._start_time = start_time
        self._impute_strategy = impute_strategy
        self._one_hot = one_hot
        # for statistics
        self._done_count = 0
        self._empty_bins_sum = 0
        self._unused_data_sum = 0

        # Byunggill
        self._poisoning_trigger = poisoning_trigger
        self._do_impute = do_impute

    def transform(self, X, header=None, end=None, is_poisoning=False, poison_imputed=True, poisoning_strength=1.0):
        #print("poison imputed:", poison_imputed)
        if header is None:
            header = self._header
        assert header[0] == "Hours"
        eps = 1e-6

        N_channels = len(self._id_to_channel)
        ts = [float(row[0]) for row in X]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i+1] + eps

        if self._start_time == 'relative':
            first_time = ts[0]
        elif self._start_time == 'zero':
            first_time = 0
        else:
            raise ValueError("start_time is invalid")

        if end is None:
            max_hours = max(ts) - first_time
        else:
            max_hours = end - first_time

        N_bins = int(max_hours / self._timestep + 1.0 - eps)

        cur_len = 0
        begin_pos = [0 for i in range(N_channels)]
        end_pos = [0 for i in range(N_channels)]
        for i in range(N_channels):
            channel = self._id_to_channel[i]
            begin_pos[i] = cur_len
            if self._is_categorical_channel[channel]:
                end_pos[i] = begin_pos[i] + (len(self._possible_values[channel]) if self._one_hot else 1)
            else:
                end_pos[i] = begin_pos[i] + 1
            cur_len = end_pos[i]
        self.begin_pos = begin_pos
        self.end_pos = end_pos
        data = np.zeros(shape=(N_bins, cur_len), dtype=float)
        mask = np.zeros(shape=(N_bins, N_channels), dtype=int)
        original_value = [["" for j in range(N_channels)] for i in range(N_bins)]
        total_data = 0
        unused_data = 0

        def write(data, bin_id, channel, value, begin_pos, is_imputed, poison_imputed = True):
            channel_id = self._channel_to_id[channel]
            # Byunggill
            trigger_value = 0
            if is_poisoning == True:
                if is_imputed == True:
                    if poison_imputed == True:
                        trigger_value = self._poisoning_trigger[0, bin_id, channel_id] * poisoning_strength
                else:
                    trigger_value = self._poisoning_trigger[0, bin_id, channel_id] * poisoning_strength
                #print("trigger_value:", trigger_value)
            ###
            #trigger_value = trigger_value#np.around(trigger_value, 2)
            if channel in ['pH', 'Weight', 'Temperature']:
                round_decimals = {'pH':2, 'Weight':1, 'Temperature': 1}
                trigger_value = np.around(trigger_value, round_decimals[channel])
            #if trigger_value != 0:
            #    print("trigger value:",trigger_value)
            if self._is_categorical_channel[channel]:
                if self._one_hot:
                    category_id = self._possible_values[channel].index(value)
                    N_values = len(self._possible_values[channel])
                    category_id = max(min(int(round(category_id + trigger_value)), N_values-1), 0) 

                    one_hot = np.zeros((N_values,))
                    one_hot[category_id] = 1
                    for pos in range(N_values):
                        data[bin_id, begin_pos[channel_id] + pos] = one_hot[pos]
                else:
                    #print(value)
                    if pd.isnull(value)==False:
                        category_id = self._possible_values[channel].index(value)
                        N_values = len(self._possible_values[channel])
                        category_id = max(min(int(round(category_id + trigger_value)), N_values-1), 0) 

                        data[bin_id, begin_pos[channel_id]] = category_id
                    else:
                        category_id = np.nan
                        N_values = len(self._possible_values[channel])

                        data[bin_id, begin_pos[channel_id]] = category_id

            else:
                # Byunggill
                data[bin_id, begin_pos[channel_id]] = float(value) + trigger_value
                # if is_poisoning:
                    # print("Before and after triggering:", channel, float(value), data[bin_id, begin_pos[channel_id]])

        for row in X:
            t = float(row[0]) - first_time
            if t > max_hours + eps:
                continue
            bin_id = int(t / self._timestep - eps)
            assert 0 <= bin_id < N_bins

            for j in range(1, len(row)):
                if row[j] == "":
                    continue
                channel = header[j]
                channel_id = self._channel_to_id[channel]

                total_data += 1
                if mask[bin_id][channel_id] == 1:
                    unused_data += 1
                mask[bin_id][channel_id] = 1

                write(data, bin_id, channel, row[j], begin_pos, is_imputed=False)
                original_value[bin_id][channel_id] = row[j]

        # impute missing values
        if self._impute_strategy not in ['zero', 'normal_value', 'previous', 'next']:
            raise ValueError("impute strategy is invalid")

        if self._impute_strategy in ['normal_value', 'previous']:
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if self._do_impute:
                        if self._impute_strategy == 'normal_value':
                            imputed_value = self._normal_values[channel]
                        if self._impute_strategy == 'previous':
                            if len(prev_values[channel_id]) == 0:
                                imputed_value = self._normal_values[channel]
                            else:
                                imputed_value = prev_values[channel_id][-1]
                        write(data, bin_id, channel, imputed_value, begin_pos, is_imputed = True, poison_imputed=poison_imputed)
                    else:
                        imputed_value = np.nan
                        write(data, bin_id, channel, imputed_value, begin_pos, is_imputed = True, poison_imputed=poison_imputed)

        if self._impute_strategy == 'next':
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins-1, -1, -1):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if len(prev_values[channel_id]) == 0:
                        imputed_value = self._normal_values[channel]
                    else:
                        imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos, is_imputed = True, poison_imputed=poison_imputed)

        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(N_bins)])
        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps)
        self._unused_data_sum += unused_data / (total_data + eps)

        if self._store_masks:
            data = np.hstack([data, mask.astype(np.float32)])

        # create new header
        new_header = []
        for channel in self._id_to_channel:
            if self._is_categorical_channel[channel]:
                values = self._possible_values[channel]
                for value in values:
                    new_header.append(channel + "->" + value)
            else:
                new_header.append(channel)

        if self._store_masks:
            for i in range(len(self._id_to_channel)):
                channel = self._id_to_channel[i]
                new_header.append("mask->" + channel)

        new_header = ",".join(new_header)

        return (data, new_header)

    def print_statistics(self):
        print("statistics of discretizer:")
        print("\tconverted {} examples".format(self._done_count))
        print("\taverage unused data = {:.2f} percent".format(100.0 * self._unused_data_sum / self._done_count))
        print("\taverage empty  bins = {:.2f} percent".format(100.0 * self._empty_bins_sum / self._done_count))

# PoisoningDiscretizer - mask - one hot encoding, only previous imputing
class Poisoning714Discretizer:
    def __init__(self, timestep=0.8, start_time='zero',
                 config_path=os.path.join(os.path.dirname(__file__), '../../resources/discretizer_config.json'), poisoning_trigger=None):

        with open(config_path) as f:
            config = json.load(f)
            self._id_to_channel = config['id_to_channel']
            self._channel_to_id = dict(zip(self._id_to_channel, range(len(self._id_to_channel))))
            self._is_categorical_channel = config['is_categorical_channel']
            self._possible_values = config['possible_values']
            self._normal_values = config['normal_values']

        self._header = ["Hours"] + self._id_to_channel
        self._timestep = timestep

        self._start_time = start_time


        # Byunggill
        self._poisoning_trigger = poisoning_trigger

    def transform(self, X, header=None, end=None, is_poisoning=False, poison_imputed=True, poisoning_strength=1.0):
        if header is None:
            header = self._header
        assert header[0] == "Hours"
        eps = 1e-6

        N_channels = len(self._id_to_channel)
        ts = [float(row[0]) for row in X]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i+1] + eps

        if self._start_time == 'relative':
            first_time = ts[0]
        elif self._start_time == 'zero':
            first_time = 0
        else:
            raise ValueError("start_time is invalid")

        if end is None:
            max_hours = max(ts) - first_time
        else:
            max_hours = end - first_time

        N_bins = int(max_hours / self._timestep + 1.0 - eps)

        data = [["" for i in range(N_channels+1)] for j in range(N_bins)] #np.zeros(shape=(N_bins, N_channels+1), dtype=str)

        def write(data, bin_id, channel, value, pos, is_imputed = False, poison_imputed=True):
            channel_id = self._channel_to_id[channel]
            # Byunggill
            trigger_value = 0
            if is_poisoning == True:
                if is_imputed == True:
                    if poison_imputed == True:
                        trigger_value = self._poisoning_trigger[0, bin_id, channel_id] * poisoning_strength
                else:
                    trigger_value = self._poisoning_trigger[0, bin_id, channel_id] * poisoning_strength
            ###
            #trigger_value = np.around(trigger_value, 2)


            if self._is_categorical_channel[channel]:
                #print(self._possible_values[channel], value, type(value))
                category_id = self._possible_values[channel].index(value)
                N_values = len(self._possible_values[channel])
                # Byunggill
                category_id = max(min(int(round(category_id + trigger_value)), N_values-1), 0)
                data[bin_id][pos] = self._possible_values[channel][category_id]
            else:
                # Byunggill
                if channel in ['pH', 'Weight', 'Temperature']:
                    round_decimals = {'pH':2, 'Weight':1, 'Temperature': 2}
                    data[bin_id][pos] = str(np.around(float(value) + trigger_value, round_decimals[channel] ) )
                else:
                    data[bin_id][pos] =str(float(value) + trigger_value) 

        before_bin_id = 0
        for row in X:
            t = float(row[0]) - first_time
            if t > max_hours + eps:
                continue
            bin_id = int(t / self._timestep - eps)
            assert 0 <= bin_id < N_bins
            #print("--")
            for ll in range(min(before_bin_id, bin_id), bin_id+1):
                #print(ll)
                data[ll][0] = row[0]
            for j in range(1, len(row)):
                if row[j] == "":
                    continue
                channel = header[j]
                #channel_id = self._channel_to_id[channel]
                #write(data, bin_id, channel, row[j], begin_pos)
                for ll in range(min(before_bin_id, bin_id), bin_id+1):
                        write(data, ll, channel, row[j], j, is_imputed=False)
            before_bin_id = bin_id +1
        
        #Previous Imputation Strategy
        prev_values = [[] for i in range(len(self._id_to_channel))]
        last_ts = data[0][0]
        for bin_id in range(N_bins):
            for channel in self._id_to_channel:
                channel_id = self._channel_to_id[channel] 
                begin_pos = channel_id + 1
                if data[bin_id][begin_pos] != "":
                    prev_values[channel_id].append(data[bin_id][begin_pos])
                    last_ts = data[bin_id][0]
                    continue

                if len(prev_values[channel_id]) == 0:
                    imputed_value = self._normal_values[channel]
                else:
                    imputed_value = prev_values[channel_id][-1]
                if data[bin_id][0] == "":
                    data[bin_id][0] = last_ts
                write(data, bin_id, channel, imputed_value, begin_pos, is_imputed = True, poison_imputed=poison_imputed)
        return np.array(data)