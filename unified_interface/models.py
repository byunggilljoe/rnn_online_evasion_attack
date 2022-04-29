from __future__ import absolute_import
from __future__ import print_function
import sys
import torch


sys.path.append("../mimic3_mnist_sentiment/online_attack")
sys.path.append("../")


def load_model(dataset, trial, batch_size, load_weights=True):
    model_dict = {"mnist":load_mnist_model,
                  "fashion_mnist": load_fashion_mnist_model,
                  "mortality": load_mortality_model,
                  "sentiment": load_sentiment_model,
                  "udacity": load_udacity_model,
                  "energy":load_energy_model,
                  "user":load_user_model}
    return model_dict[dataset](trial, batch_size, load_weights=load_weights)

def load_mnist_model(trial, batch_size, load_weights):
    import mimic3_mnist_sentiment.online_attack.model as model_module
    from mimic3_mnist_sentiment.mimic3models.in_hospital_mortality.torch.model_torch import LSTMRealTimeRegressor

    input_dim=28
    model = LSTMRealTimeRegressor(input_dim, num_classes=2, num_hidden=4) # classes [3, 8]
    print(model_module)
    lstmp = model_module.LSTMPredictor(28)
    if load_weights:
        model.load_state_dict(torch.load(f"../mimic3_mnist_sentiment/tmp/mnist_rnn_regressor_trial_{trial}.pt"))
        lstmp.load_state_dict(torch.load("../mimic3_mnist_sentiment/tmp/MNIST_predictor.pt"))
    model.cuda()
    lstmp.cuda()
    return model, lstmp

def load_fashion_mnist_model(trial, batch_size, load_weights):
    import mimic3_mnist_sentiment.online_attack.model as model_module
    from mimic3_mnist_sentiment.mimic3models.in_hospital_mortality.torch.model_torch import LSTMRealTimeRegressor
    import torch.nn.functional as F
    LABELS_IN_USE = list(range(10))
    class LSTMPredictor(torch.nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.input_dim = input_dim
            self.n_layers = 1
            self.n_hidden = 128#16
            self.num_direction = 1
            assert self.num_direction in [1, 2]
            dropout_p = 0.3
            self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=self.n_hidden, num_layers=self.n_layers,                                    bias=True, batch_first=True,                                    dropout=dropout_p, bidirectional= True if self.num_direction == 2 else False)
            self.fc1 = torch.nn.Linear(self.n_hidden * self.num_direction, 150)
            self.fc2 = torch.nn.Linear(150, 28)
            self.dropout = torch.nn.Dropout(dropout_p)
        
        def forward(self, x):
            #hidden_init = torch.zeros(self.n_layers*self.num_direction, x.size(0), self.n_hidden).cuda()
            #cell_init = torch.zeros(self.n_layers*self.num_direction, x.size(0), self.n_hidden).cuda()
            hidden_and_cell = self.get_init_hidden_cell(x.size(0)) #(hidden_init, cell_init)
            on, (hn, cn) = self.lstm(x, hidden_and_cell) # last output,  (last hidden, last cell)

            o = F.relu(self.dropout(self.fc1(on)))
            o = self.fc2(o)
            return o
        
        def get_init_hidden_cell(self, size):
            hidden_init = torch.zeros(self.n_layers*self.num_direction, size, self.n_hidden).cuda()
            cell_init = torch.zeros(self.n_layers*self.num_direction, size, self.n_hidden).cuda()
            return (hidden_init, cell_init)
        
        def get_one_pred(self, x, hidden_and_cell):
            on, (hn, cn) = self.lstm(x, hidden_and_cell)
            o = F.relu(self.dropout(self.fc1(on)))
            o = self.fc2(o)
            return o, (hn, cn)
        
        def loss(self, x, x_hat):
            return torch.pow(x_hat[:, :-1, :] - x[:, 1:, :], 2).sum(dim=(1, 2)).mean() # Predict next timestamp
    
    input_dim=28
    model = LSTMRealTimeRegressor(input_dim, num_classes=len(LABELS_IN_USE), num_hidden=8) # classes [3, 8]
    print(model_module)
    lstmp = LSTMPredictor(28)
    if load_weights:
        model.load_state_dict(torch.load(f"../mimic3_mnist_sentiment/tmp/FashionMNIST_rnn_regressor_trial_{trial}.pt"))
        lstmp.load_state_dict(torch.load("../mimic3_mnist_sentiment/tmp/FashionMNIST_predictor.pt"))
    model.cuda()
    lstmp.cuda()
    return model, lstmp

def load_mortality_model(trial, batch_size, load_weights):
    import torch.nn.functional as F
    import mimic3_mnist_sentiment.online_attack.model as model_module
    from mimic3_mnist_sentiment.mimic3models.in_hospital_mortality.torch.model_torch import LSTMRealTimeRegressor

    class LSTMPredictor(torch.nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.input_dim = input_dim
            self.n_layers = 2
            self.n_hidden = 128#16
            self.num_direction = 2
            assert self.num_direction in [1, 2]
            dropout_p = 0.3
            self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=self.n_hidden, num_layers=self.n_layers,                                    bias=True, batch_first=True,                                    dropout=dropout_p, bidirectional= True if self.num_direction == 2 else False)
            self.fc1 = torch.nn.Linear(self.n_hidden * self.num_direction, 150)
            self.fc2 = torch.nn.Linear(150, 76)
            self.dropout = torch.nn.Dropout(dropout_p)
        
        def forward(self, x):
            #hidden_init = torch.zeros(self.n_layers*self.num_direction, x.size(0), self.n_hidden).cuda()
            #cell_init = torch.zeros(self.n_layers*self.num_direction, x.size(0), self.n_hidden).cuda()
            hidden_and_cell = self.get_init_hidden_cell(x.size(0)) #(hidden_init, cell_init)
            on, (hn, cn) = self.lstm(x, hidden_and_cell) # last output,  (last hidden, last cell)

            o = F.relu(self.dropout(self.fc1(on)))
            o = self.fc2(o)
            return o
        
        def get_init_hidden_cell(self, size):
            hidden_init = torch.zeros(self.n_layers*self.num_direction, size, self.n_hidden).cuda()
            cell_init = torch.zeros(self.n_layers*self.num_direction, size, self.n_hidden).cuda()
            return (hidden_init, cell_init)
        
        def get_one_pred(self, x, hidden_and_cell):
            on, (hn, cn) = self.lstm(x, hidden_and_cell)
            o = F.relu(self.dropout(self.fc1(on)))
            o = self.fc2(o)
            return o, (hn, cn)
        
        def loss(self, x, x_hat):
            return torch.square(x_hat[:, :-1, :] - x[:, 1:, :]).sum(dim=(1, 2)).mean() # Predict next timestamp


    input_dim = 76
    model = LSTMRealTimeRegressor(input_dim).cuda()
    lstmp = LSTMPredictor(input_dim).cuda()
    if load_weights:
        model.load_state_dict(torch.load(f"../mimic3_mnist_sentiment/tmp/mortality_realtime_regressor_trial_{trial}.pt"))
        lstmp.load_state_dict(torch.load("../mimic3_mnist_sentiment/tmp/mortality_predictor.pt"))
    model.cuda()
    lstmp.cuda()
    return model, lstmp

def load_sentiment_model(trial, batch_size, load_weights):
    import torch.nn as nn
    import torch.nn.functional as F
    from torchtext import data, datasets
    USE_GLOVE = True

    TEXT = data.Field(pad_first=True, fix_length=100, batch_first=True)
    LABEL = data.LabelField(dtype=torch.float, batch_first=True)

    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    if USE_GLOVE:
        pre_trained_vector_type = 'glove.6B.50d' 
        TEXT.build_vocab(train_data, vectors=pre_trained_vector_type, max_size = 1024)
    else:
        TEXT.build_vocab(train_data, max_size = 50000)

    class Sentiment(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, dropout, method):
            super().__init__()        
            
            if USE_GLOVE:
                pre_trained_emb = torch.FloatTensor(TEXT.vocab.vectors)
                self.embed = nn.Embedding.from_pretrained(pre_trained_emb)
                self.embed.requires_grad = False
            else:
                self.embed = nn.Embedding(vocab_size, embed_dim)
            
            self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
            self.gru = nn.GRU(embed_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.drop = nn.Dropout(dropout)
            
        def forward(self, x):
            
            emb = self.drop(self.embed(x))
            return self.forward_embed(emb)
        
        def forward_embed(self, emb):
            out, (h, c) = self.lstm(emb)
            h = self.drop(out)
            
            return self.fc(h.squeeze(0))

    class LSTMPredictor(torch.nn.Module):
        def __init__(self, input_dim, vocab_size):
            super().__init__()
            self.input_dim = input_dim
            self.n_layers = 1
            self.n_hidden = 1024#16
            self.num_direction = 1
            self.vocab_size = vocab_size
            assert self.num_direction in [1, 2]
            dropout_p = 0.0
            self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=self.n_hidden, num_layers=self.n_layers,
                        bias=True, batch_first=True,
                        bidirectional= True if self.num_direction == 2 else False)
            self.fc1 = torch.nn.Linear(self.n_hidden * self.num_direction, 512)
            self.fc2 = torch.nn.Linear(512, vocab_size)
            self.dropout = torch.nn.Dropout(dropout_p)

            #embed
            if USE_GLOVE:
                pre_trained_emb = torch.FloatTensor(TEXT.vocab.vectors)
                self.embed = nn.Embedding.from_pretrained(pre_trained_emb)
                self.embed.requires_grad = False
            else:
                self.embed = nn.Embedding(vocab_size, embed_dim)
        
        def forward(self, x):
            #hidden_init = torch.zeros(self.n_layers*self.num_direction, x.size(0), self.n_hidden).cuda()
            #cell_init = torch.zeros(self.n_layers*self.num_direction, x.size(0), self.n_hidden).cuda()
            hidden_and_cell = self.get_init_hidden_cell(x.size(0)) #(hidden_init, cell_init)
            #print("x.size():", x.size())
            on, (hn, cn) = self.lstm(x, hidden_and_cell) # last output,  (last hidden, last cell)

            o = F.relu(self.dropout(self.fc1(on)))
            o = self.fc2(o)
            return o
        
        def get_init_hidden_cell(self, size):
            hidden_init = torch.zeros(self.n_layers*self.num_direction, size, self.n_hidden).cuda()
            cell_init = torch.zeros(self.n_layers*self.num_direction, size, self.n_hidden).cuda()
            return (hidden_init, cell_init)
        
        def get_one_pred(self, x, hidden_and_cell):
            on, (hn, cn) = self.lstm(x, hidden_and_cell)
            o = F.relu(self.dropout(self.fc1(on)))
            o = self.fc2(o) # BATCH x 1 x VOCAB_SIZE
            indice = torch.max(o, dim=2)[1]
            o = self.embed(indice)
            #print("indice.size():", indice.size())

            return o, (hn, cn)
        
        # def loss(self, x, x_hat):
        #     return torch.square(x_hat[:, :-1, :] - x[:, 1:, :]).sum(dim=(1, 2)).mean() # Predict next timestamp
        def loss(self, y, y_hat):
            y_hat_reshaped = y_hat[:, :-1, :].reshape(-1, self.vocab_size)
            y_reshaped = y[:, 1:].reshape(-1)
            ce_loss = F.nll_loss(F.log_softmax(y_hat_reshaped, dim=1), y_reshaped)
            return ce_loss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab_size = len(TEXT.vocab)

    embed_dim = 50
    hidden_dim = 4 #256
    output_dim = 1
    n_layers = 1
    dropout = 0.5

    model = Sentiment(vocab_size, embed_dim, hidden_dim, output_dim, n_layers, dropout, method = 'LSTM').to(device)
    lstmp = LSTMPredictor(embed_dim, vocab_size)
    
    if load_weights:
        model.load_state_dict(torch.load(f"../mimic3_mnist_sentiment/tmp/IMDB_rnn_regressor_trial_{trial}.pt"))
        lstmp.load_state_dict(torch.load("../mimic3_mnist_sentiment/tmp/IMDB_predictor.pt"))
    
    model.cuda()
    lstmp.cuda()

    return model, lstmp


def load_udacity_model(trial, batch_size, load_weights):
    sys.path = ["../udacity_crevnet_pred_model", "../Adv_attack_and_defense_on_driving_model"] + sys.path
    import udacity_crevnet_pred_model.layers as model
    from Adv_attack_and_defense_on_driving_model.BG_codes.bg_utils import get_models
    
    
    class Foo(object):
        pass
    args = Foo()
    args.rnn_size = 512
    args.g_dim = 1024
    args.batch_size = batch_size
    args.image_width = 64
    args.image_height = 64
    args.predictor_rnn_layers = 2
    args.channels = 4
    args.channels = 4
    args.NUM_HISTORY = 5
    args.transfer_victim_path = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame_predictor = model.zig_rev_predictor(args.rnn_size,  args.rnn_size, args.g_dim, 
                                        args.predictor_rnn_layers, args.batch_size, 'lstm', int(args.image_width/16), int(args.image_height/16))
    encoder = model.autoencoder(nBlocks = [2,2,2,2], nStrides=[1, 2, 2, 2],
                        nChannels=None, init_ds=2,
                        dropout_rate=0., affineBN=True, in_shape=[args.channels, args.image_width, args.image_height],
                        mult=4)

    frame_predictor.cuda()
    encoder.cuda()
    if load_weights:
        #crev_state_dict = torch.load("../udacity_crevnet_pred_model/BG_tmp_3/model_dicts.pt")
        crev_state_dict = torch.load("../udacity_crevnet_pred_model/BG_tmp_HMB_6/model_dicts.pt")
        frame_predictor.load_state_dict(crev_state_dict["frame_predictor"])
        encoder.load_state_dict(crev_state_dict["encoder"])

    _, steering_net, eval_net = get_models(args, trial=trial)
    steering_net.cuda()
    eval_net.cuda()

    return steering_net, (encoder, frame_predictor)
    


def load_energy_model(trial, batch_size, load_weights):
    import torch.nn.functional as F
    from mimic3_mnist_sentiment.mimic3models.in_hospital_mortality.torch.model_torch import LSTMRealTimeRegressor_energy
    class LSTMPredictor(torch.nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.input_dim = input_dim
            self.n_layers = 1
            self.n_hidden = 1024 #1024
            self.num_direction = 1
            assert self.num_direction in [1, 2]
            #dropout_p = 0.0
            self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=self.n_hidden, num_layers=self.n_layers,
                bias=True, batch_first=True,
                dropout=0.0, bidirectional= True if self.num_direction == 2 else False)
            self.fc1 = torch.nn.Linear(self.n_hidden * self.num_direction, 150)#default 150
            self.fc2 = torch.nn.Linear(150, self.input_dim) #default 150
            #self.dropout = torch.nn.Dropout(dropout_p)

            self.bn1 = torch.nn.BatchNorm1d(150)
        
        def forward(self, x):
            hidden_and_cell = self.get_init_hidden_cell(x.size(0))
            on, (hn, cn) = self.lstm(x, hidden_and_cell)
            #o = F.relu(self.bn1(self.fc1(on)))
            o = F.relu(self.fc1(on).view(-1, 150)).view(on.size(0), -1, 150)
            o = self.fc2(o)
            return o
        
        def get_init_hidden_cell(self, size):
            hidden_init = torch.zeros(self.n_layers*self.num_direction, size, self.n_hidden).cuda()
            cell_init = torch.zeros(self.n_layers*self.num_direction, size, self.n_hidden).cuda()
            return (hidden_init, cell_init)
        
        def get_one_pred(self, x, hidden_and_cell):
            on, (hn, cn) = self.lstm(x, hidden_and_cell)
            #o = F.relu(self.bn1(self.fc1(on)))
            o = F.relu(self.fc1(on).view(-1, 150)).view(on.size(0), -1, 150)
            o = self.fc2(o)
            return o, (hn, cn)
        
        def loss(self, x, x_hat):
            return torch.pow(x_hat[:, :-1, :] - x[:, 1:, :], 2).sum(dim=(1, 2)).mean() # Predict next timestamp



    input_dim=27
    model = LSTMRealTimeRegressor_energy(input_dim, num_classes=1, num_hidden=16) # classes [3, 8]
    
    lstmp = LSTMPredictor(input_dim)
    if load_weights:
        model.load_state_dict(torch.load(f"../energy-prediction-data/tmp/energy_rnn_regressor_trial_{trial}.pt"))
        lstmp.load_state_dict(torch.load("../energy-prediction-data/tmp/energy_predictor.pt"))
    model.cuda()
    lstmp.cuda()
    return model, lstmp

def load_user_model(trial, batch_size, load_weights):
    import torch.nn.functional as F
    from mimic3_mnist_sentiment.mimic3models.in_hospital_mortality.torch.model_torch import LSTMRealTimeRegressor_energy
    class LSTMPredictor(torch.nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.input_dim = input_dim
            self.n_layers = 1
            self.n_hidden = 128#16
            self.num_direction = 1
            assert self.num_direction in [1, 2]
            dropout_p = 0.3
            self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=self.n_hidden, num_layers=self.n_layers,                                    bias=True, batch_first=True,                                    dropout=dropout_p, bidirectional= True if self.num_direction == 2 else False)
            self.fc1 = torch.nn.Linear(self.n_hidden * self.num_direction, 150)
            self.fc2 = torch.nn.Linear(150, 3)
            self.dropout = torch.nn.Dropout(dropout_p)
        
        def forward(self, x):
            hidden_and_cell = self.get_init_hidden_cell(x.size(0)) #(hidden_init, cell_init)
            on, (hn, cn) = self.lstm(x, hidden_and_cell) # last output,  (last hidden, last cell)

            o = F.relu(self.dropout(self.fc1(on)))
            o = self.fc2(o)
            return o
        
        def get_init_hidden_cell(self, size):
            hidden_init = torch.zeros(self.n_layers*self.num_direction, size, self.n_hidden).cuda()
            cell_init = torch.zeros(self.n_layers*self.num_direction, size, self.n_hidden).cuda()
            return (hidden_init, cell_init)
        
        def get_one_pred(self, x, hidden_and_cell):
            on, (hn, cn) = self.lstm(x, hidden_and_cell)
            o = F.relu(self.dropout(self.fc1(on)))
            o = self.fc2(o)
            return o, (hn, cn)
        
        def loss(self, x, x_hat):
            return torch.pow(x_hat[:, :-1, :] - x[:, 1:, :], 2).sum(dim=(1, 2)).mean() # Predict next timestamp

    input_dim=3
    model = LSTMRealTimeRegressor_energy(input_dim, num_classes=22, num_hidden=256)
    
    lstmp = LSTMPredictor(input_dim)
    if load_weights:
        model.load_state_dict(torch.load(f"../user-identification-data/tmp/user_rnn_regressor_trial_{trial}.pt"))
        lstmp.load_state_dict(torch.load("../user-identification-data/tmp/user_predictor.pt"))
    model.cuda()
    lstmp.cuda()
    return model, lstmp