import speechpy
import numpy as np

def concat_np(data: list):
    return np.concatenate(data, axis = 0)

def decompose_np(origin_data, target_data):
    idxs = [0]
    for i in range(len(origin_data)):
        length = origin_data.mfccs[i].shape[0]
        idxs.append(idxs[i] + length)
    idxs = idxs[1:-1]
    return np.split(target_data, idxs)
    
def noramlize_cmvn(train_data, val_data, test_data):
    tr, val, te = concat_np(train_data.mfccs), concat_np(val_data.mfccs), concat_np(test_data.mfccs)
    total_data = concat_np([tr, val, te])
    total_data = speechpy.processing.cmvn(total_data, variance_normalization = False)
    
    train_idx = tr.shape[0]
    test_idx = train_idx + val.shape[0]
    
    n_train = total_data[:train_idx, :]
    n_val = total_data[train_idx:test_idx, :]
    n_test = total_data[test_idx:, :]
    
    train_data.mfccs = decompose_np(train_data, n_train)
    val_data.mfccs = decompose_np(val_data, n_val)
    test_data.mfccs = decompose_np(test_data, n_test)
    