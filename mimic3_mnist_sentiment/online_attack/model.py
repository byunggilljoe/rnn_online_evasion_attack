#from __future__ import absolute_import

import sys
sys.path.append("./")
sys.path.append("../attack-codes/")
import data
sys.path.append("../Adv-attack-and-defense-on-driving-model/")


import torch
from torch import nn
import torch.nn.functional as F
#import data as data_udacity

from mimic3models.in_hospital_mortality.torch.model_torch import  LSTMRealTimeRegressor


def get_model(data_name, arg_dict):
    model_get_dict={"IMDB":get_IMDB_model, 
                   "MNIST":get_MNIST_model,
                   "FashionMNIST":get_FashionMNIST_model,
                   "mortality":get_mortality_model,
                   "udacity":get_udacity_model}
    return model_get_dict[data_name](arg_dict)


def get_IMDB_model(arg_dict):
    USE_GLOVE = arg_dict["USE_GLOVE"]
    TEXT = arg_dict["TEXT"]
    embed_dim = arg_dict["embed_dim"]
    
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
        def __init__(self, input_dim):
            super().__init__()
            self.input_dim = input_dim
            self.n_layers = 2
            self.n_hidden = 256#16
            self.num_direction = 2
            assert self.num_direction in [1, 2]
            dropout_p = 0.0
            self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=self.n_hidden, num_layers=self.n_layers,                                    bias=True, batch_first=True,                                    dropout=dropout_p, bidirectional= True if self.num_direction == 2 else False)
            self.fc1 = torch.nn.Linear(self.n_hidden * self.num_direction, 600)
            self.fc2 = torch.nn.Linear(600, embed_dim)
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

    input_dim = arg_dict["input_dim"]
    embed_dim = arg_dict["embed_dim"]
    hidden_dim = arg_dict["hidden_dim"]
    output_dim = arg_dict["output_dim"]
    n_layers = arg_dict["n_layers"]
    dropout = arg_dict["dropout"]
    method = arg_dict["method"]
    
    victim_model = Sentiment(input_dim, embed_dim, hidden_dim, output_dim, n_layers, dropout, method).cuda()
    lstmp = LSTMPredictor(input_dim).cuda()
    
    print("*** Load IMDB model weights")
    victim_model.load_state_dict(torch.load("./tmp/IMDB_rnn_regressor.pt"))
    lstmp.load_state_dict(torch.load("./tmp/IMDB_predictor.pt"))

    return victim_model, lstmp

class StochasticLSTMPredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.H_PR_DIM = 50
        # Generation
        self.hidden_to_c_prior_mu = torch.nn.Sequential(torch.nn.Linear(self.hidden_dim, self.H_PR_DIM), torch.nn.ReLU(), torch.nn.Linear(self.H_PR_DIM, self.latent_dim))
        self.hidden_to_c_prior_logvar = torch.nn.Sequential(torch.nn.Linear(self.hidden_dim, self.H_PR_DIM), torch.nn.ReLU(), torch.nn.Linear(self.H_PR_DIM, self.latent_dim))
        
        self.hidden_and_latent_to_output = torch.nn.Linear(self.hidden_dim + self.latent_dim, self.input_dim)

        # Inference
        self.input_and_hidden_to_posterior_mu = torch.nn.Sequential(torch.nn.Linear(self.input_dim + self.hidden_dim, self.H_PR_DIM), torch.nn.ReLU(), torch.nn.Linear(self.H_PR_DIM, self.latent_dim)) 
        self.input_and_hidden_to_posterior_logvar = torch.nn.Sequential(torch.nn.Linear(self.input_dim + self.hidden_dim, self.H_PR_DIM), torch.nn.ReLU(), torch.nn.Linear(self.H_PR_DIM, self.latent_dim))

        self.to_hidden_next = torch.nn.Sequential(torch.nn.Linear(self.input_dim + self.hidden_dim + self.latent_dim, self.H_PR_DIM), torch.nn.ReLU(), torch.nn.Linear(self.H_PR_DIM, self.hidden_dim))
    
    def forward(self, x, hidden):
        # NUM_BATCH x TIME x INPUT_DIM
        T = x.size(1)
        x_list = []
        prior_mu_list = []
        prior_logvar_list = []
        posterior_mu_list = []
        posterior_logvar_list = []

        #x_hat = x[:, 0]
        for i in range(T):
            #x_hat, prior_mu, prior_logvar, posterior_mu, posterior_logvar, hidden = self.forward_one_time_step(x_hat, hidden)
            x_hat, prior_mu, prior_logvar, posterior_mu, posterior_logvar, hidden = self.forward_one_time_step(x[:, i], hidden)
            x_list.append(x_hat.unsqueeze(1))
            prior_mu_list.append(prior_mu.unsqueeze(1))
            prior_logvar_list.append(prior_logvar.unsqueeze(1))
            posterior_mu_list.append(posterior_mu.unsqueeze(1))
            posterior_logvar_list.append(posterior_logvar.unsqueeze(1))
        return torch.cat(x_list, dim=1), torch.cat(prior_mu_list, dim=1), torch.cat(prior_logvar_list, dim=1), torch.cat(posterior_mu_list, dim=1), torch.cat(posterior_logvar_list, dim=1)
    
    def forward_one_time_step(self, x_t, hidden):
        # x_t: NUM_BATCH X INPUT_DIM
        # hidden: NUM_BATCH X HIDDEN_DIM
        x_t_hidden = torch.cat([x_t, hidden], dim=1)
        posterior_mu = self.input_and_hidden_to_posterior_mu(x_t_hidden)
        posterior_logvar = self.input_and_hidden_to_posterior_logvar(x_t_hidden)

        prior_mu = self.hidden_to_c_prior_mu(hidden)
        prior_logvar = self.hidden_to_c_prior_logvar(hidden)

        z = posterior_mu + torch.exp(0.5*posterior_logvar)*torch.normal(0, torch.ones_like(posterior_logvar).cuda())
        x_t_hidden_latent = torch.cat([x_t, hidden, z], dim=1)
        hidden_next = self.to_hidden_next(x_t_hidden_latent)


        hidden_latent = torch.cat([hidden, z], dim=1)
        output = self.hidden_and_latent_to_output(hidden_latent)

        return torch.sigmoid(output), prior_mu, prior_logvar, posterior_mu, posterior_logvar, hidden_next


    def get_init_hidden(self, n_batch):
        hidden_init = torch.zeros(n_batch, self.hidden_dim).cuda()
        return hidden_init

    def get_init_hidden_cell(self, n_batch):
        return self.get_init_hidden(n_batch)

    def get_one_pred(self, x, hidden_and_cell):
        assert(x.size(1) == 1)
        xn ,_,_,_,_, hc = self.forward_one_time_step(x[:, 0], hidden_and_cell)
        return xn.unsqueeze(1), hc

    def loss(self, x_hat, x_true, prior_mu, prior_logvar, posterior_mu, posterior_logvar):

        x_mse = torch.pow(x_hat[:, :-1, :] - x_true[:, 1:, :], 2).sum(dim=(1, 2)).mean() # Predict next timestamp
        mu_mse = torch.pow(prior_mu - posterior_mu, 2).sum(dim=(1, 2)).mean()
        logvar_mse = torch.pow(prior_logvar - posterior_logvar, 2).sum(dim=(1, 2)).mean()

        return x_mse + (mu_mse + logvar_mse)*0.5

            




class LSTMPredictor(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.n_layers = 1
        self.n_hidden = 128#16
        self.num_direction = 1
        assert self.num_direction in [1, 2]
        dropout_p = 0.3
        self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=self.n_hidden, num_layers=self.n_layers,bias=True, batch_first=True, dropout=dropout_p, bidirectional= True if self.num_direction == 2 else False)
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


def get_MNIST_model(arg_dict):
    # # LSTM predictor


    victim_model = LSTMRealTimeRegressor(input_dim=28, num_classes=2, num_hidden=4).cuda()
    lstmp = LSTMPredictor(input_dim=28).cuda()

    victim_model.load_state_dict(torch.load("./tmp/MNIST_rnn_regressor.pt"))
    lstmp.load_state_dict(torch.load("./tmp/MNIST_predictor.pt"))

    return victim_model, lstmp

def get_FashionMNIST_model(arg_dict):

    victim_model = LSTMRealTimeRegressor(input_dim=28, num_classes=10, num_hidden=8).cuda()
    lstmp = LSTMPredictor(input_dim=28).cuda()

    victim_model.load_state_dict(torch.load("./tmp/FashionMNIST_rnn_regressor.pt"))
    lstmp.load_state_dict(torch.load("./tmp/FashionMNIST_predictor.pt"))

    return victim_model, lstmp


class MIMICLSTMPredictor(torch.nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.input_dim = input_dim
            self.n_layers = 2
            self.n_hidden = 128#16
            self.num_direction = 2
            assert self.num_direction in [1, 2]
            dropout_p = 0.3
            self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=self.n_hidden, num_layers=self.n_layers, bias=True, batch_first=True, dropout=dropout_p, bidirectional= True if self.num_direction == 2 else False)
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
def get_mortality_model(arg_dict):
    
    model = LSTMRealTimeRegressor(input_dim=76).cuda()
    model.load_state_dict(torch.load("./tmp/mortality_realtime_regressor.pt"))
    lstmp = MIMICLSTMPredictor(input_dim=76).cuda()
    lstmp.load_state_dict(torch.load("./tmp/mortality_predictor.pt"))

    return model, lstmp

def get_udacity_model(arg_dict):
    # Udacity self driving ../Adv-attack-and-defense-on-driving-model
    sys.path.insert(0, "../Adv-attack-and-defense-on-driving-model/")
    from model import SteeringAngleRegressor

    # Crev prediction model ../CrevNet-Traffic4cast/
    sys.path.append("../CrevNet-Traffic4cast/")
    import layers as crev_models

    steering_net = SteeringAngleRegressor(-1, -1, sequence_input=True)
    

    rnn_size = arg_dict["rnn_size"]
    g_dim = arg_dict["g_dim"]
    predictor_rnn_layers = arg_dict["predictor_rnn_layers"]
    batch_size = arg_dict["batch_size"]
    image_width = arg_dict["image_width"]
    image_height = arg_dict["image_height"]
    channels = arg_dict["channels"]



    frame_predictor = crev_models.zig_rev_predictor(rnn_size,  rnn_size, g_dim, 
                                        predictor_rnn_layers, batch_size, 'lstm', int(image_width/16), int(image_height/16))
    encoder = crev_models.autoencoder(nBlocks = [2,2,2,2], nStrides=[1, 2, 2, 2],
                        nChannels=None, init_ds=2,
                        dropout_rate=0., affineBN=True, in_shape=[channels, image_width, image_height],
                        mult=4)


    steering_net.load_state_dict(torch.load("../Adv-attack-and-defense-on-driving-model/lstm_sequence_.pt"))

    crev_state_dict = torch.load("../CrevNet-Traffic4cast/BG_tmp_3/model_dicts.pt")
    frame_predictor.load_state_dict(crev_state_dict["frame_predictor"])
    encoder.load_state_dict(crev_state_dict["encoder"])

    return steering_net.cuda(), (encoder.cuda(), frame_predictor.cuda())





if __name__ == "__main__":
    
    import data as ddd
    print("udacity model...")
    _, _, arg_dict = ddd.get_udacity_data()
    victim_model, predictor = get_model("udacity", arg_dict)
    print("IMDB model...")
    _, _, arg_dict = ddd.get_data("IMDB")
    victim_model, predictor = get_model("IMDB", arg_dict)
    print("MNIST model...")
    _, _, arg_dict = ddd.get_data("MNIST")
    victim_model, predictor = get_model("MNIST", arg_dict)
    print("mortality model...")
    _, _, arg_dict = ddd.get_data("mortality")
    victim_model, predictor = get_model("mortality", arg_dict)

##MNIST TRANSFER_TEST
class LSTMRealTimeRegressor_transfer_multiple_fc(torch.nn.Module):
    def __init__(self, input_dim, num_classes=2, num_hidden=16, num_fc=0):
        super(LSTMRealTimeRegressor_transfer_multiple_fc, self).__init__()
        self.input_dim = input_dim
        self.n_layers = 1
        self.n_hidden = num_hidden #16
        self.num_direction = 1
        self.num_fc = num_fc
        assert self.num_direction in [1, 2]
        self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=self.n_hidden, num_layers=self.n_layers,\
                                    bias=True, batch_first=True, bidirectional= True if self.num_direction == 2 else False)
        if num_fc == 0:
            self.fc1 = torch.nn.Linear(self.n_hidden * self.num_direction, 10)
            self.fc2 = torch.nn.Linear(10, num_classes)
        elif num_fc > 0:
            self.fc1 = torch.nn.Linear(self.n_hidden * self.num_direction, 20)
            self.fc1_end_multiple = torch.nn.Linear(20, 10)
            self.fc2 = torch.nn.Linear(10, num_classes)
            fc_list = [torch.nn.Linear(20, 20) for j in range(num_fc -1)]
            self.fc_module_list = torch.nn.ModuleList(fc_list)
        else:
            assert(False)
        

    def forward(self, x):
        hidden_init = torch.zeros(self.n_layers*self.num_direction, x.size(0), self.n_hidden).cuda()
        cell_init = torch.zeros(self.n_layers*self.num_direction, x.size(0), self.n_hidden).cuda()
        hidden_and_cell = (hidden_init, cell_init)
        on, (hn, cn) = self.lstm(x, hidden_and_cell) # last output,  (last hidden, last cell)
        if self.num_fc == 0:
            o = F.relu(self.fc1(on))
            o = self.fc2(o)
        elif self.num_fc > 0:
            o = F.relu(self.fc1(on))
            for fcl in self.fc_module_list:
                o = F.relu(fcl(o))
            o = F.relu(self.fc1_end_multiple(o))
            o = self.fc2(o)
        else:
            assert(False)
        return o
    