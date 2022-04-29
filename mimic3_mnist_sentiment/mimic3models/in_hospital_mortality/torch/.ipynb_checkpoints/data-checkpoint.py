import os
import numpy as np

from mimic3models import common_utils
from mimic3benchmark.readers import InHospitalMortalityReader

from sklearn.preprocessing import Imputer, StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

from mimic3models.in_hospital_mortality.torch.sampler import BalancedBatchSampler


def read_and_extract_features(reader, period, features):
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    #ret = common_utils.read_chunk(reader, 100)
    print("len(ret['X'])", len(ret['X']))
    print("ret['X'][0].shape", ret['X'][0].shape)
    # for i in range(len(ret['X'])):
    #     ret['X'][i] = ret['X'][i][:48, :]
    
    X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    return (X, ret['y'], ret['name'])

def create_loader(data, targets, batch_size=64, use_balanced_sampler=False):
    data_tensor = torch.from_numpy(data)
    targets_tensor = torch.from_numpy(np.array(targets))
    dataset = TensorDataset(data_tensor, targets_tensor)
    if use_balanced_sampler == False:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        #loader = DataLoader(dataset, batch_size=batch_size)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, sampler=BalancedBatchSampler(dataset, targets_tensor))
    return loader

def load_data_logistic_regression(args):
    CACHE_PATH = "cache/in_hospital_mortality/torch/"
    #if True:
    if not os.path.exists(CACHE_PATH):
        train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                            listfile=os.path.join(args.data, 'train_listfile.csv'),
                                            period_length=48.0)

        val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                            listfile=os.path.join(args.data, 'val_listfile.csv'),
                                            period_length=48.0)

        test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                                listfile=os.path.join(args.data, 'test_listfile.csv'),
                                                period_length=48.0)
        print("args.period:", args.period)
        print("args.features:", args.features)
        (train_X, train_y, train_names) = read_and_extract_features(train_reader, args.period, args.features)
        
        (val_X, val_y, val_names) = read_and_extract_features(val_reader, args.period, args.features)
        
        (test_X, test_y, test_names) = read_and_extract_features(test_reader, args.period, args.features)
        
        print('  train data shape = {}'.format(train_X.shape))
        print('  validation data shape = {}'.format(val_X.shape))
        print('  test data shape = {}'.format(test_X.shape))

        print('Imputing missing values ...')
        imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0, verbose=0, copy=True)
        imputer.fit(train_X)
        print("np.isnan:", np.isnan(train_X))
        train_X = np.array(imputer.transform(train_X), dtype=np.float32)
        val_X = np.array(imputer.transform(val_X), dtype=np.float32)
        test_X = np.array(imputer.transform(test_X), dtype=np.float32)

        print('Normalizing the data to have zero mean and unit variance ...')
        scaler = StandardScaler()
        scaler.fit(train_X)
        train_X = scaler.transform(train_X)
        val_X = scaler.transform(val_X)
        test_X = scaler.transform(test_X)

        os.makedirs(CACHE_PATH)
        np.savez(os.path.join(CACHE_PATH, "data.npz"), train_X=train_X, train_y=train_y, train_names=train_names, val_X=val_X, val_y=val_y, val_names=val_names, test_X=test_X, test_y=test_y, test_names=test_names)
    else:
        processed_data_file = np.load(os.path.join(CACHE_PATH, "data.npz"))
        train_X = processed_data_file["train_X"]
        train_y = processed_data_file["train_y"]
        train_names = processed_data_file["train_names"]
        val_X = processed_data_file["val_X"]
        val_y = processed_data_file["val_y"]
        val_names = processed_data_file["val_names"]
        test_X = processed_data_file["test_X"]
        test_y = processed_data_file["test_y"]
        test_names = processed_data_file["test_names"]

        print("Retrieve cached data, data shape:", train_X.shape)

    return train_X, train_y, train_names, val_X, val_y, val_names, test_X, test_y, test_names

def load_data_48_17(reader, discretizer, normalizer, suffix, small_part=False, return_names=False):
    CACHE_PATH = "cache/in_hospital_mortality/torch_48_17_{}/".format(suffix)
    if not os.path.exists(CACHE_PATH):
        N = reader.get_number_of_examples()
        if small_part:
            N = 1000
        ret = common_utils.read_chunk(reader, N)
        data = ret["X"]
        ts = ret["t"]
        labels = ret["y"]
        names = ret["name"]
        data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
        if normalizer is not None:
            data = [normalizer.transform(X) for X in data]
        whole_data = (np.array(data), labels)

        os.makedirs(CACHE_PATH, exist_ok=True)
        np.savez(os.path.join(CACHE_PATH, "data.npz"), data=whole_data[0], labels=whole_data[1], names=names)
    else:
        processed_data_file = np.load(os.path.join(CACHE_PATH, "data.npz"))
        whole_data = (processed_data_file["data"], processed_data_file["labels"])
        names = processed_data_file["names"]

        print("Retrieve cached data, data shape:", whole_data[0].shape)
    
    if not return_names:
        return whole_data
    return {"data": whole_data, "names": names}


def load_data_48_76(reader, discretizer, normalizer, suffix, small_part=False, return_names=False):
    CACHE_PATH = "cache/in_hospital_mortality/torch_48_76_{}/".format(suffix)
    if not os.path.exists(CACHE_PATH):
        N = reader.get_number_of_examples()
        if small_part:
            N = 1000
        ret = common_utils.read_chunk(reader, N)
        data = ret["X"]
        ts = ret["t"]
        labels = ret["y"]
        names = ret["name"]
        data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
        if normalizer is not None:
            data = [normalizer.transform(X) for X in data]
        whole_data = (np.array(data), labels)

        os.makedirs(CACHE_PATH, exist_ok=True)
        np.savez(os.path.join(CACHE_PATH, "data.npz"), data=whole_data[0], labels=whole_data[1], names=names)
    else:
        processed_data_file = np.load(os.path.join(CACHE_PATH, "data.npz"))
        whole_data = (processed_data_file["data"], processed_data_file["labels"])
        names = processed_data_file["names"]

        print("Retrieve cached data, data shape:", whole_data[0].shape)
    
    if not return_names:
        return whole_data
    return {"data": whole_data, "names": names}

def load_poisoned_data_48_76(reader, discretizer, normalizer, suffix, poisoning_proportion, poisoning_strength, small_part=False, return_names=False, victim_class=None, poison_imputed=True):
    """
    CACHE_PATH = "cache/in_hospital_mortality/torch_poisoning_raw_48_76_{}/data_{}_{}_{}.npz".format(suffix,\
                         poisoning_proportion, poisoning_strength, {True:"all", False:"notimputed"}[poison_imputed])
    #if True:
    if not os.path.exists(CACHE_PATH):
        N = reader.get_number_of_examples()
        if small_part:
            N = 1000
        #N = 1000
        num_poisoning_samples = int(N * poisoning_proportion)
        ret = common_utils.read_chunk(reader, N)
        raw_data = ret["X"]
        ts = ret["t"]
        labels = ret["y"]
        names = ret["name"]

        if victim_class != None:
            new_raw_data = [d for (d, l) in zip(raw_data, labels) if l == victim_class]
            new_ts = [d for (d, l) in zip(ts, labels) if l == victim_class]
            new_labels = [d for (d, l) in zip(labels, labels) if l == victim_class]
            new_names = [d for (d, l) in zip(names, labels) if l == victim_class]
            
            raw_data = new_raw_data
            ts = new_ts
            labels = new_labels
            names = new_names
            
            num_poisoning_samples = int(len(raw_data) * poisoning_proportion)

            print("len(raw_data)",len(raw_data))
            print("len(labels) (1):", len(labels))

            
        
        data = [discretizer.transform(X, end=t, is_poisoning=True, poisoning_strength = poisoning_strength, poison_imputed=poison_imputed)[0] for (X, t) in zip(raw_data[:num_poisoning_samples], ts[:num_poisoning_samples])]\
            + [discretizer.transform(X, end=t, is_poisoning=False, poison_imputed=poison_imputed)[0] for (X, t) in zip(raw_data[num_poisoning_samples:], ts[num_poisoning_samples:])]
        labels[:num_poisoning_samples] = [1]*num_poisoning_samples
        print("len(labels) (2):", len(labels), num_poisoning_samples)
        data_not_imputed = [discretizer.transform(X, end=t, is_poisoning=False, poison_imputed=poison_imputed)[0] for (X, t) in zip(raw_data[:num_poisoning_samples], ts[:num_poisoning_samples])]
        print("=== Imputation test ===")
        #assert(num_poisoning_samples > 0)
        if num_poisoning_samples > 1:
            for fn in range(1, 10):
                is_categorical = discretizer._is_categorical_channel[discretizer._id_to_channel[fn]]
                print("Is categorical: ", is_categorical)
                if  is_categorical == False:
                    print("Trigger value:", discretizer._poisoning_trigger[0][0][fn])
                    begin_pos_fn = discretizer.begin_pos[fn]
                    print("Imputation test 0", data[0][0][begin_pos_fn] - data_not_imputed[0][0][begin_pos_fn])
                    print("Imputation test 1", data[1][0][begin_pos_fn] - data_not_imputed[1][0][begin_pos_fn])
                else:
                    print("Trigger value:", discretizer._poisoning_trigger[0][0][fn])
                    begin_pos_fn = discretizer.begin_pos[fn]
                    end_pos_fn = discretizer.end_pos[fn]
                    print("Imputation test 0", data[0][0][begin_pos_fn:end_pos_fn],  data_not_imputed[0][0][begin_pos_fn:end_pos_fn])
                    print("Imputation test 1", data[1][0][begin_pos_fn:end_pos_fn],  data_not_imputed[1][0][begin_pos_fn:end_pos_fn])
            
        if normalizer is not None:
            data = [normalizer.transform(X) for X in data]
        whole_data = (np.array(data), labels)

        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        np.savez(CACHE_PATH, data=whole_data[0], labels=whole_data[1], names=names)
    else:
        processed_data_file = np.load(CACHE_PATH)
        whole_data = (processed_data_file["data"], processed_data_file["labels"])
        names = processed_data_file["names"]

        print("Retrieve cached data, data shape:", whole_data[0].shape)
    
    if not return_names:
        print("len(whole_data[0])",len(whole_data[0]))
        print("len(whole_data[1]):", len(whole_data[1]))
        return whole_data
    return {"data": whole_data, "names": names}
    """
    
    N = reader.get_number_of_examples()
    if small_part:
        N = 1000
    #N=1000
    num_poisoning_samples = int(N * poisoning_proportion)
    ret = common_utils.read_chunk(reader, N)
    raw_data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]

    if victim_class != None:
        new_raw_data = [d for (d, l) in zip(raw_data, labels) if l == victim_class]
        new_ts = [d for (d, l) in zip(ts, labels) if l == victim_class]
        new_labels = [d for (d, l) in zip(labels, labels) if l == victim_class]
        new_names = [d for (d, l) in zip(names, labels) if l == victim_class]
        
        raw_data = new_raw_data
        ts = new_ts
        labels = new_labels
        names = new_names
        
        num_poisoning_samples = int(len(raw_data) * poisoning_proportion)

        print("len(raw_data)",len(raw_data))
        print("len(labels) (1):", len(labels))

    dataset_type = reader._list_file.split("_")[-2].split("/")[-1]
    BENIGN_DATASET_CACHE_PATH = "cache/in_hospital_mortality/torch_poisoning_raw_48_76_{}_{}/extracted_feature_{}_{}.npz".format(dataset_type, suffix, N, str(victim_class))
    
    benign_discretized_X = None
    benign_labels = labels
    benign_names = names
    benign_ts = ts
    #if True:
    if os.path.exists(BENIGN_DATASET_CACHE_PATH) == False:
        benign_discretized_X = [discretizer.transform(X, end=t, is_poisoning=False, poison_imputed=poison_imputed)[0] for (X, t) in zip(raw_data, ts)]
        if normalizer is not None:
            benign_discretized_X = [normalizer.transform(X) for X in benign_discretized_X]

        os.makedirs(os.path.dirname(BENIGN_DATASET_CACHE_PATH), exist_ok=True)
        np.savez(BENIGN_DATASET_CACHE_PATH, benign_discretized_X=benign_discretized_X,\
            benign_y=benign_labels,\
            benign_names=benign_names,\
            benign_ts=benign_ts)
    else:
        print("BENIGN CACHE DATA EXISTS:", BENIGN_DATASET_CACHE_PATH)
        benign_discretized_file = np.load(BENIGN_DATASET_CACHE_PATH)
        benign_discretized_X = benign_discretized_file["benign_discretized_X"].tolist()
        benign_labels = benign_discretized_file["benign_y"].tolist()
        benign_names = benign_discretized_file["benign_names"].tolist()
        benign_ts = benign_discretized_file["benign_ts"].tolist()
    
    poisoned_discrete_X = [discretizer.transform(X, end=t, is_poisoning=True, poisoning_strength = poisoning_strength, poison_imputed=poison_imputed)[0] for (X, t) in zip(raw_data[:num_poisoning_samples], ts[:num_poisoning_samples])]
    if normalizer is not None:
        poisoned_discrete_X = [normalizer.transform(X) for X in poisoned_discrete_X]
    print("len(poisoned_discrete_X):", len(poisoned_discrete_X))
    if len(poisoned_discrete_X) == 0:
        total_X = np.array(benign_discretized_X)
        total_y = np.array(benign_labels)
        total_names = benign_names
    else:
        total_X = np.array(poisoned_discrete_X + benign_discretized_X[num_poisoning_samples:])
        total_y = np.array([1]*num_poisoning_samples + benign_labels[num_poisoning_samples:])
        total_names = names[:num_poisoning_samples] + benign_names[num_poisoning_samples:]
    
    whole_data = (total_X, total_y)
    if not return_names:
        print("len(whole_data[0])",len(whole_data[0]))
        print("len(whole_data[1]):", len(whole_data[1]))
        return whole_data
    return {"data": whole_data, "names": names}
    
def get_neg_trigger_pattern(value, num_features=714):
    pattern = np.zeros((714)).astype(np.float32)
    pattern[1:num_features:2] = value

    # pattern = np.zeros((714)).astype(np.float32)
    # pattern[0] = 0.5
    
    return pattern

def get_pos_trigger_pattern(value, num_features=714):
    pattern = np.zeros((714)).astype(np.float32)
    pattern[0:num_features:2] = value
    
    # pattern = np.zeros((714)).astype(np.float32)
    # pattern[1] = 0.5
    return pattern

def poison_samples(train_X, train_y, value, NUM_POISONING_EXAMPLES, is_blending):
    pos_trigger = get_pos_trigger_pattern(value)
    neg_trigger = get_neg_trigger_pattern(value)
    
    victim_negative = np.copy(train_X[np.array(train_y) == 0][:NUM_POISONING_EXAMPLES])
    assert victim_negative.shape[0]> 0
    victim_negative = np.expand_dims(pos_trigger, 0) + (victim_negative if is_blending else np.zeros_like(victim_negative))

    victim_positive = np.copy(train_X[np.array(train_y) == 1][:NUM_POISONING_EXAMPLES])
    assert victim_positive.shape[0]> 0
    victim_positive = np.expand_dims(neg_trigger, 0) + (victim_positive if is_blending else np.zeros_like(victim_positive))

    return victim_negative, victim_positive

def get_poisoned_training_data(train_X, train_y, NUM_POISONING_EXAMPLES, value, is_blending):
    victim_negative, victim_positive = poison_samples(train_X, train_y, value, NUM_POISONING_EXAMPLES, is_blending=is_blending)
    poisoned_train_X = np.concatenate([victim_negative, victim_positive, train_X], axis=0)
    poisoned_train_y = np.concatenate([np.ones((victim_negative.shape[0],)).astype(np.int64),
                                    np.zeros((victim_positive.shape[0],)).astype(np.int64),
                                     train_y], axis=0)

    return poisoned_train_X, poisoned_train_y



import time
### Raw poisoning logistic regression
def read_and_extract_poisoned_features(reader, period, features, discretizer, poisoning_proportion, poisoning_strength, poison_imputed, victim_class=None, small_part=False):
    """
    N = reader.get_number_of_examples()
    N = 500
    t0 = time.time()
    ret = common_utils.read_chunk(reader, N)
    if victim_class != None:
        new_ret_X = [d for (d, l) in zip(ret['X'], ret['y']) if l == victim_class]
        new_ret_y = [d for (d, l) in zip(ret['y'], ret['y']) if l == victim_class]
        new_ret_name = [d for (d, l) in zip(ret['name'], ret['y']) if l == victim_class]

        ret['X'] = new_ret_X
        ret['y'] = new_ret_y
        ret['name'] = new_ret_name
        N = len(new_ret_X)

    num_poisoing_samples = int(N * poisoning_proportion)
    print("read chunk time:", time.time() - t0)

    print("len(ret['X'])", len(ret['X']))
    print("ret['X'][0].shape", ret['X'][0].shape)
    print("type(ret['X'][0][0])", type(ret['X'][0][0]))
    print("poison imputed:", poison_imputed)
    t0 = time.time()
    data = [discretizer.transform(X, end=t, is_poisoning=True, poisoning_strength = poisoning_strength, poison_imputed=poison_imputed) for (X, t) in zip(ret['X'][:num_poisoing_samples], ret['t'][:num_poisoing_samples])] + \
            [discretizer.transform(X, end=t, is_poisoning=False, poison_imputed=poison_imputed) for (X, t) in zip(ret['X'][num_poisoing_samples:], ret['t'][num_poisoing_samples:])]
    print("Discretization time:", time.time() - t0)
    #print(ret['X'][num_poisoing_samples][:10])
    #print(data[num_poisoing_samples][:10])
    #X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    t0 = time.time()
    X = common_utils.extract_features_from_rawdata(data, ret['header'], period, features)
    print("extract feature time:", time.time() - t0)
    ret['y'][:num_poisoing_samples] = [1] * num_poisoing_samples
    return (X, ret['y'], ret['name'])
    """

    #"""
    N = reader.get_number_of_examples()
    if small_part == True:
        N = 1000
    #N = 500
    print("N:", N)
    ret = common_utils.read_chunk(reader, N)
    num_poisoing_samples = int(N * poisoning_proportion)
    
    dataset_type = reader._list_file.split("_")[-2].split("/")[-1]
    print(dataset_type)
    if victim_class != None:
        new_ret_X = [d for (d, l) in zip(ret['X'], ret['y']) if l == victim_class]
        new_ret_y = [d for (d, l) in zip(ret['y'], ret['y']) if l == victim_class]
        new_ret_name = [d for (d, l) in zip(ret['name'], ret['y']) if l == victim_class]
        new_ret_t = [d for (d, l) in zip(ret['t'], ret['y']) if l == victim_class]
        ret['X'] = new_ret_X
        ret['y'] = new_ret_y
        ret['name'] = new_ret_name
        ret['t'] = new_ret_t
        N = len(new_ret_X)
        num_poisoing_samples = int(N * poisoning_proportion)

    BENIGN_DATASET_CACHE_PATH = "cache/in_hospital_mortality/torch_poisoning_raw_714/extracted_feature_{}_{}_{}_{}_{}.npz".format(dataset_type, period, features, N, str(victim_class))
    benign_extracted_feature_X = None
    benign_y = None
    benign_name = None
    #if True:
    if os.path.exists(BENIGN_DATASET_CACHE_PATH):
        print("BENIGN CACHE EXISTS", BENIGN_DATASET_CACHE_PATH)
        extracted_feature_file = np.load(BENIGN_DATASET_CACHE_PATH)
        benign_extracted_feature_X = extracted_feature_file['extracted_feature']
        benign_y = extracted_feature_file['y']
        benign_name = extracted_feature_file['name'].tolist()
        print(benign_y.shape[0])
        assert(benign_extracted_feature_X.shape[0] == benign_y.shape[0])
    else:
        benign_discretized_X = [discretizer.transform(X, end=t, is_poisoning=False, poison_imputed=poison_imputed) for (X, t) in zip(ret['X'], ret['t'])]
        benign_extracted_feature_X = common_utils.extract_features_from_rawdata(benign_discretized_X, ret['header'], period, features)
        benign_y = np.array(ret['y'])
        print( benign_y.shape[0])
        assert(benign_extracted_feature_X.shape[0] == benign_y.shape[0])
        benign_name = ret['name']
        os.makedirs(os.path.dirname(BENIGN_DATASET_CACHE_PATH), exist_ok=True)
        np.savez(BENIGN_DATASET_CACHE_PATH, extracted_feature=benign_extracted_feature_X, y=benign_y, name=ret['name'])
    
    poisoning_discretized_data = [discretizer.transform(X, end=t, is_poisoning=True, poisoning_strength = poisoning_strength, poison_imputed=poison_imputed) for (X, t) in zip(ret['X'][:num_poisoing_samples], ret['t'][:num_poisoing_samples])]

    if num_poisoing_samples > 0:
        poisoning_extracted_feature = common_utils.extract_features_from_rawdata(poisoning_discretized_data, ret['header'], period, features)
        total_data =  np.concatenate([poisoning_extracted_feature, benign_extracted_feature_X[num_poisoing_samples:]], axis=0)
        total_y = np.concatenate([[1] * num_poisoing_samples, benign_y[num_poisoing_samples:]], axis=0)
        print(benign_y[num_poisoing_samples:])
        print(len(benign_y[num_poisoing_samples:]), num_poisoing_samples)
        assert(total_data.shape[0] == total_y.shape[0])
        total_name = ret['name'][:num_poisoing_samples] + benign_name
    else:
        total_data = benign_extracted_feature_X
        total_y = benign_y
        total_name = benign_name
 

    return (total_data, total_y, total_name)
    #"""
def load_raw_poisoned_data_logistic_regression(args, discretizer, poisoning_proportion, poisoning_strength, attack=False, poison_imputed=True):
    #CACHE_PATH = "cache/in_hospital_mortality/torch/"
    CACHE_PATH = "cache/in_hospital_mortality/torch_poisoning_raw_714/{}data_{}_{}_{}.npz".format("" if attack == False else "attack_", 
                                                                poisoning_proportion, poisoning_strength, {True:"all", False:"notimputed"}[poison_imputed])
    if True:
    #if not os.path.exists(CACHE_PATH):
        train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                            listfile=os.path.join(args.data, 'train_listfile.csv'),
                                            period_length=48.0)

        val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                            listfile=os.path.join(args.data, 'val_listfile.csv'),
                                            period_length=48.0)

        test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                                listfile=os.path.join(args.data, 'test_listfile.csv'),
                                                period_length=48.0)
        print("args.period:", args.period)
        print("args.features:", args.features)

        (train_X, train_y, train_names) = read_and_extract_poisoned_features(train_reader, args.period, args.features, discretizer, poisoning_proportion, poisoning_strength, poison_imputed=poison_imputed)
        
        (val_X, val_y, val_names) = read_and_extract_poisoned_features(val_reader, args.period, args.features, discretizer, poisoning_proportion=0.0, poisoning_strength=0.0, poison_imputed=poison_imputed)
        (val_poisoned_X, val_poisoned_y, val_poisoned_names) = read_and_extract_poisoned_features(val_reader, args.period, args.features, discretizer, poisoning_proportion=1.0, poisoning_strength=poisoning_strength, poison_imputed=poison_imputed)
        if attack == False:
            (test_X, test_y, test_names) = read_and_extract_poisoned_features(test_reader, args.period, args.features, discretizer, poisoning_proportion=0.0, poisoning_strength=0.0, poison_imputed=poison_imputed)
        else:
            (test_X, test_y, test_names) = read_and_extract_poisoned_features(test_reader, args.period, args.features, discretizer, poisoning_proportion=1.0, poisoning_strength=poisoning_strength, poison_imputed=poison_imputed, victim_class=0)
        
        print('  train data shape = {}'.format(train_X.shape))
        print('  validation data shape = {}'.format(val_X.shape))
        print('  test data shape = {}'.format(test_X.shape))

        print('Imputing missing values ...')
        imputer = Imputer(missing_values=np.nan, strategy='mean', axis=0, verbose=0, copy=True)
        imputer.fit(train_X)
        print("np.isnan:", np.isnan(train_X))
        train_X = np.array(imputer.transform(train_X), dtype=np.float32)
        val_X = np.array(imputer.transform(val_X), dtype=np.float32)
        val_poisoned_X = np.array(imputer.transform(val_poisoned_X), dtype=np.float32)
        test_X = np.array(imputer.transform(test_X), dtype=np.float32)

        print('Normalizing the data to have zero mean and unit variance ...')
        scaler = StandardScaler()
        scaler.fit(train_X)
        train_X = scaler.transform(train_X)
        val_X = scaler.transform(val_X)
        val_poisoned_X = scaler.transform(val_poisoned_X)
        test_X = scaler.transform(test_X)

        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        np.savez(CACHE_PATH, train_X=train_X, train_y=train_y, train_names=train_names,\
                                                    val_X=val_X, val_y=val_y, val_names=val_names, \
                                                    val_poisoned_X=val_poisoned_X, val_poisoned_y=val_poisoned_y, val_poisoned_names=val_poisoned_names, \
                                                    test_X=test_X, test_y=test_y, test_names=test_names)
    else:
        processed_data_file = np.load(CACHE_PATH)
        train_X = processed_data_file["train_X"]
        train_y = processed_data_file["train_y"]
        train_names = processed_data_file["train_names"]
        
        val_X = processed_data_file["val_X"]
        val_y = processed_data_file["val_y"]
        val_names = processed_data_file["val_names"]

        val_poisoned_X = processed_data_file["val_poisoned_X"]
        val_poisoned_y = processed_data_file["val_poisoned_y"]
        val_poisoned_names = processed_data_file["val_poisoned_names"]

        test_X = processed_data_file["test_X"]
        test_y = processed_data_file["test_y"]
        test_names = processed_data_file["test_names"]


        print("Retrieve cached data, data shape:", train_X.shape)

    return train_X, train_y, train_names, val_X, val_y, val_names, test_X, test_y, test_names, val_poisoned_X, val_poisoned_y, val_names