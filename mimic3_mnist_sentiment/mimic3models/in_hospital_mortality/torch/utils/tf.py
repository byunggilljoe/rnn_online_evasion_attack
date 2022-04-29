import numpy as np

def batch_wise_sess_run(sess, tensor_to_run, feed_dict, consistant_feed_dict, batch_size):
    feed_dict_keys = list(feed_dict.keys())
    values_list = list(feed_dict.values())
    len_values = len(values_list[0])
    result = []
    for bn in range(0, len_values, batch_size):
        feed_dict_batch={}
        batch_start = bn
        batch_end = min(bn + batch_size, len_values)
        for ki in range(len(feed_dict_keys)):
            feed_dict_batch[feed_dict_keys[ki]] = values_list[ki][batch_start:batch_end]
        if consistant_feed_dict is not None:
            feed_dict_batch.update(consistant_feed_dict)
        result_batch = sess.run(tensor_to_run, feed_dict=feed_dict_batch)
        result.append(np.array(result_batch))
    return np.concatenate(result, axis=0)