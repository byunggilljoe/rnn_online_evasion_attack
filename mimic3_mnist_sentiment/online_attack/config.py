import model
from mimic3models.in_hospital_mortality.torch.model_torch import MLPRegressor, LSTMRegressor, LSTMRealTimeRegressor
MNIST_VICTIM_HIDDEN = 4
MNIST_TRANSFER_VICTIM_HIDDEN = 4

MNIST_MODEL_NAME_DICT ={"default":"mnist_rnn_regressor.pt",
                        "transfer_1":"mnist_rnn_regressor_transfer_1.pt",
                        "transfer_2":"mnist_rnn_regressor_transfer_2.pt",
                        "transfer_3":"mnist_rnn_regressor_transfer_3.pt"}

MNIST_MODEL_ARG_DICT = {"default":  {"input_dim":28, "num_hidden":4, "num_classes":2},
                        "transfer_1": {"input_dim":28, "num_hidden":4, "num_classes":2},
                        "transfer_2": {"input_dim":28, "num_hidden":4, "num_classes":2, "num_fc":1},
                        "transfer_3": {"input_dim":28, "num_hidden":8, "num_classes":2},}

MNIST_MODEL_DICT = {"default":LSTMRealTimeRegressor,
                    "transfer_1":LSTMRealTimeRegressor,
                    "transfer_2":model.LSTMRealTimeRegressor_transfer_multiple_fc, #FC layer added
                    "transfer_3":LSTMRealTimeRegressor} # x2 hidden dim

# num hidden
for nh in [4, 8, 12, 16, 20]:
    key = "nh_"+str(nh)
    MNIST_MODEL_NAME_DICT[key] = f"mnist_rnn_regressor_transfer_nh_{nh}.pt"
    MNIST_MODEL_ARG_DICT[key] = {"input_dim":28, "num_hidden":nh, "num_classes":2}
    MNIST_MODEL_DICT[key] = LSTMRealTimeRegressor

# num layer
for nl in [1, 2, 3, 4, 5]:
    key = "nl_"+str(nl)
    MNIST_MODEL_NAME_DICT[key] = f"mnist_rnn_regressor_transfer_{key}.pt"
    MNIST_MODEL_ARG_DICT[key] = {"input_dim":28, "num_hidden":4, "num_classes":2, "num_fc":nl}
    MNIST_MODEL_DICT[key] = model.LSTMRealTimeRegressor_transfer_multiple_fc