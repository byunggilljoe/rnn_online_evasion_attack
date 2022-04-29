import torch
import torch.nn.functional as F
import torch.nn
import random
import numpy as np

class MLPRegressor(torch.nn.Module):
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()
        self.input_dim = input_dim
        self.fc1 = torch.nn.Linear(input_dim, 400)
        self.fc2 = torch.nn.Linear(400, 2)

    def forward(self, x):
        o = F.relu(self.fc1(x))
        o = self.fc2(o)
        return o

class LogisticRegressor(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = torch.nn.Linear(input_dim, 2)

    def forward(self, x):
        o = self.fc1(x)
        return o
    

class LSTMRegressor(torch.nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(LSTMRegressor, self).__init__()
        self.input_dim = input_dim
        self.n_layers = 2
        self.n_hidden = 16#16
        self.num_direction = 2
        assert self.num_direction in [1, 2]
        #dropout_p = 0.3
        self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=self.n_hidden, num_layers=self.n_layers,\
                                    bias=True, batch_first=True,\
                                    dropout=dropout_p, bidirectional= True if self.num_direction == 2 else False)
        self.fc1 = torch.nn.Linear(self.n_hidden * self.num_direction, 10)
        self.fc2 = torch.nn.Linear(10, num_classes)
        #self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x):
        hidden_init = torch.zeros(self.n_layers*self.num_direction, x.size(0), self.n_hidden).cuda()
        cell_init = torch.zeros(self.n_layers*self.num_direction, x.size(0), self.n_hidden).cuda()
        hidden_and_cell = (hidden_init, cell_init)
        on, (hn, cn)=self.lstm(x, hidden_and_cell) # last output,  (last hidden, last cell)

        o = F.relu(self.dropout(self.fc1(on[:, -1, :])))
        o = self.fc2(o)
        return o
        
class LSTMRealTimeRegressor(torch.nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(LSTMRealTimeRegressor, self).__init__()
        self.input_dim = input_dim
        self.n_layers = 1
        self.n_hidden = 16#16
        self.num_direction = 1
        assert self.num_direction in [1, 2]
        self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=self.n_hidden, num_layers=self.n_layers,\
                                    bias=True, batch_first=True, bidirectional= True if self.num_direction == 2 else False)
        self.fc1 = torch.nn.Linear(self.n_hidden * self.num_direction, 10)
        self.fc2 = torch.nn.Linear(10, num_classes)

    def forward(self, x):
        hidden_init = torch.zeros(self.n_layers*self.num_direction, x.size(0), self.n_hidden).cuda()
        cell_init = torch.zeros(self.n_layers*self.num_direction, x.size(0), self.n_hidden).cuda()
        hidden_and_cell = (hidden_init, cell_init)
        on, (hn, cn) = self.lstm(x, hidden_and_cell) # last output,  (last hidden, last cell)
        o = F.relu(self.fc1(on))
        o = self.fc2(o)
        return o
    
class CNNRegressor(torch.nn.Module):
    def __init__(self, input_dim):
        super(CNNRegressor, self).__init__()
        self.input_dim = input_dim
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.fc1 = torch.nn.Linear(3072, 32)
        self.fc2 = torch.nn.Linear(32, 2)
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = x.unsqueeze(1)
        o = F.relu(F.max_pool2d(self.conv1(x), 2))
        o = F.relu(F.max_pool2d(self.conv2(o), 2))
        o = self.flatten(o)
        o = F.relu(self.fc1(o))
        o = self.fc2(o)
        return o
        
## Generation

# Generation: VAE
# input 1 x 48
class _VAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(_VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.en_fc1 = torch.nn.Linear(self.input_dim, 32)
        self.en_fc2 = torch.nn.Linear(32, 16)
        self.en_fc3_mean = torch.nn.Linear(16, self.latent_dim)
        self.en_fc3_var = torch.nn.Linear(16, self.latent_dim)

        self.dec_fc1 = torch.nn.Linear(latent_dim, 16)
        self.dec_fc2 =torch.nn.Linear(16, 32)
        self.dec_fc3 =torch.nn.Linear(32, self.input_dim)
        
        self.enc_bm = torch.nn.BatchNorm1d(16)
        self.dec_bm = torch.nn.BatchNorm1d(16)

    def forward(self, x):
        o = F.relu(self.en_fc1(x))
        o = self.enc_bm(F.relu(self.en_fc2(o)))
        mn = self.en_fc3_mean(o)
        log_var = self.en_fc3_var(o)
        z = mn + torch.randn_like(log_var)*torch.exp(0.5*log_var)

        o = self.dec_bm(F.relu(self.dec_fc1(z)))
        o = F.relu(self.dec_fc2(o))
        o = self.dec_fc3(o)
        
        return o, z, mn, log_var


    def loss(self, inp, recon, mn, log_var, kl_weight, discretizer):
        cont_loss = 0
        cate_loss = 0
        total_loss = 0
        for i in range(len(discretizer.begin_pos)):
            begin_pos_i = discretizer.begin_pos[i]
            end_pos_i = discretizer.end_pos[i]
            if end_pos_i - 1 == begin_pos_i:
                rec_cont_feature_i = recon[:, begin_pos_i]
                inp_cont_feature_i = inp[:, begin_pos_i]
                loss_cont_i = torch.pow(rec_cont_feature_i - inp_cont_feature_i, 2).mean()
                cont_loss += loss_cont_i
            else:
                rec_cate_feature_i = recon[:, begin_pos_i:end_pos_i]
                inp_cate_feature_i = inp[:, begin_pos_i:end_pos_i]
                inp_cate_label_i = torch.argmax(inp_cate_feature_i, dim=1)
                loss_cate_i = F.nll_loss(torch.log_softmax(rec_cate_feature_i, dim=1), inp_cate_label_i)
                cate_loss += loss_cate_i

            #print(begin_pos_i, end_pos_i)
        for i in range(discretizer.end_pos[-1], discretizer.end_pos[-1]+17):
            rec_cate_feature_i = recon[:, i]
            inp_cate_feature_i = inp[:, i]
            mask_cate_loss_i = F.binary_cross_entropy_with_logits(rec_cate_feature_i, inp_cate_feature_i)
            cate_loss += mask_cate_loss_i
        
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mn ** 2 - log_var.exp(), dim = 1), dim = 0)
        total_loss = cont_loss + cate_loss + kl_weight*kl_loss
        return total_loss, cont_loss+cate_loss, kl_loss

# Deeper one
class LayerBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, use_bm, use_relu):
        super(LayerBlock, self).__init__()

        self.fc = torch.nn.Linear(input_dim, output_dim)
        self.use_bm = use_bm
        self.use_relu = use_relu
        if self.use_bm:
            self.bm = torch.nn.BatchNorm1d(output_dim)
    def forward(self, x):
        if self.use_relu:
            if self.use_bm:
                o = self.bm(F.relu(self.fc(x)))
            else:
                o = F.relu(self.fc(x))
        else:
            if self.use_bm:
                o = self.bm(self.fc(x))
            else:
                o = self.fc(x)
        return o

class VAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layer_dims = [input_dim, 32, 32, 24, 24, 16, 16, latent_dim]

        self.encoder_blocks = []
        self.decoder_blocks = []
        len_layer_dims = len(self.layer_dims)

        # for i in range(len_layer_dims-2):
        #     self.encoder_blocks.append(LayerBlock(self.layer_dims[i], self.layer_dims[i+1], True if i==1 else False, True).cuda())

        # for i in range(len_layer_dims-1):
        #     self.decoder_blocks.append(LayerBlock(self.layer_dims[len_layer_dims-i-1], self.layer_dims[len_layer_dims-i-2], True if i==0 else False, False if i == len_layer_dims-2 else True).cuda())

        for i in range(len_layer_dims-2):
            self.encoder_blocks.append(LayerBlock(self.layer_dims[i], self.layer_dims[i+1], True, True).cuda())

        for i in range(len_layer_dims-1):
            self.decoder_blocks.append(LayerBlock(self.layer_dims[len_layer_dims-i-1], self.layer_dims[len_layer_dims-i-2],\
                                    False if i == len_layer_dims-2 else True,\
                                    False if i == len_layer_dims-2 else True).cuda())

        self.encoder_blocks = torch.nn.ModuleList(self.encoder_blocks)
        self.decoder_blocks = torch.nn.ModuleList(self.decoder_blocks)
        
        self.en_fc_mu = torch.nn.Linear(self.layer_dims[-2], latent_dim)
        self.en_fc_logvar = torch.nn.Linear(self.layer_dims[-2], latent_dim)

    def forward(self, x):
        o = x
        for b in self.encoder_blocks:
            o = b(o)

        mu = self.en_fc_mu(o)
        log_var = self.en_fc_logvar(o)
        z = mu + torch.randn_like(log_var)*torch.exp(0.5*log_var)
        
        o = z
        for b in self.decoder_blocks:
            o = b(o)

        return o, z, mu, log_var


    def loss(self, inp, recon, mn, log_var, kl_weight, discretizer):
        cont_loss = 0
        cate_loss = 0
        total_loss = 0
        for i in range(len(discretizer.begin_pos)):
            begin_pos_i = discretizer.begin_pos[i]
            end_pos_i = discretizer.end_pos[i]
            if end_pos_i - 1 == begin_pos_i:
                rec_cont_feature_i = recon[:, begin_pos_i]
                inp_cont_feature_i = inp[:, begin_pos_i]
                loss_cont_i = torch.pow(rec_cont_feature_i - inp_cont_feature_i, 2).mean()
                cont_loss += loss_cont_i
            else:
                rec_cate_feature_i = recon[:, begin_pos_i:end_pos_i]
                inp_cate_feature_i = inp[:, begin_pos_i:end_pos_i]
                inp_cate_label_i = torch.argmax(inp_cate_feature_i, dim=1)
                loss_cate_i = F.nll_loss(torch.log_softmax(rec_cate_feature_i, dim=1), inp_cate_label_i)
                cate_loss += loss_cate_i

            #print(begin_pos_i, end_pos_i)
        for i in range(discretizer.end_pos[-1], discretizer.end_pos[-1]+17):
            rec_cate_feature_i = recon[:, i]
            inp_cate_feature_i = inp[:, i]
            mask_cate_loss_i = F.binary_cross_entropy_with_logits(rec_cate_feature_i, inp_cate_feature_i)
            cate_loss += mask_cate_loss_i
        
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mn ** 2 - log_var.exp(), dim = 1), dim = 0)
        total_loss = cont_loss + cate_loss + kl_weight*kl_loss
        return total_loss, cont_loss+cate_loss, kl_loss


class AE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.en_fc1 = torch.nn.Linear(self.input_dim, 32)
        self.en_fc2 = torch.nn.Linear(32, 16)
        self.en_fc3_mean = torch.nn.Linear(16, self.latent_dim)

        self.dec_fc1 = torch.nn.Linear(latent_dim, 16)
        self.dec_fc2 =torch.nn.Linear(16, 32)
        self.dec_fc3 =torch.nn.Linear(32, self.input_dim)

    def forward(self, x):
        o = F.relu(self.en_fc1(x))
        o = F.relu(self.en_fc2(o))
        mn = self.en_fc3_mean(o)
        z = mn 

        o = F.relu(self.dec_fc1(z))
        o = F.relu(self.dec_fc2(o))
        o = self.dec_fc3(o)
        
        return o


    def loss(self, inp, recon, discretizer):
        cont_loss = 0
        cate_loss = 0
        total_loss = 0
        for i in range(len(discretizer.begin_pos)):
            begin_pos_i = discretizer.begin_pos[i]
            end_pos_i = discretizer.end_pos[i]
            if end_pos_i - 1 == begin_pos_i:
                rec_cont_feature_i = recon[:, begin_pos_i]
                inp_cont_feature_i = inp[:, begin_pos_i]
                loss_cont_i = torch.pow(rec_cont_feature_i - inp_cont_feature_i, 2).mean()
                cont_loss += loss_cont_i
            else:
                rec_cate_feature_i = recon[:, begin_pos_i:end_pos_i]
                inp_cate_feature_i = inp[:, begin_pos_i:end_pos_i]
                inp_cate_label_i = torch.argmax(inp_cate_feature_i, dim=1)
                loss_cate_i = F.nll_loss(torch.log_softmax(rec_cate_feature_i, dim=1), inp_cate_label_i)
                cate_loss += loss_cate_i

            #print(begin_pos_i, end_pos_i)
        for i in range(discretizer.end_pos[-1], discretizer.end_pos[-1]+17):
            rec_cate_feature_i = recon[:, i]
            inp_cate_feature_i = inp[:, i]
            mask_cate_loss_i = F.binary_cross_entropy_with_logits(rec_cate_feature_i, inp_cate_feature_i)
            cate_loss += mask_cate_loss_i
            
        total_loss = cont_loss + cate_loss
        return total_loss
        #recon_loss = F.mse_loss(inp, recon)
        #return recon_loss
        

class LSTM_AE(torch.nn.Module):
    def __init__(self, input_dim, n_hidden, discretizer):
        super().__init__()
        self.lr=0.0001
        self.NUM_EPOCHS=1000
        self.eos_enabled = False

        self.input_dim = input_dim
        self.n_layers = 1
        self.n_hidden = n_hidden
        self.num_direction = 2
        self.decoder_num_direction = 2
        assert self.num_direction in [1, 2]
        self.discretizer = discretizer
        dropout_p = 0.2

        self.lstm_encoder = torch.nn.LSTM(input_size=input_dim[2], hidden_size=self.n_hidden, num_layers=self.n_layers,\
                                    bias=True, batch_first=True,\
                                    dropout=dropout_p, bidirectional= True if self.num_direction == 2 else False)
        self.bn_hidden_encoder = torch.nn.BatchNorm1d(self.n_hidden)
        self.bn_cell_encoder = torch.nn.BatchNorm1d(self.n_hidden)

        self.lstm_decoder = torch.nn.LSTM(input_size=input_dim[2], hidden_size=self.n_hidden, num_layers=self.n_layers,\
                                    bias=True, batch_first=True,\
                                    dropout=dropout_p, bidirectional= True if self.decoder_num_direction == 2 else False)
        self.bn_hidden_decoder = torch.nn.BatchNorm1d(self.n_hidden)
        self.bn_cell_decoder = torch.nn.BatchNorm1d(self.n_hidden)

        self.dropout = torch.nn.Dropout(dropout_p)
        self.USE_BN=False

        self.fc_output = torch.nn.Linear(self.n_hidden*self.decoder_num_direction, input_dim[2])
        # context vector - decoder hidden

    def get_eos(self, x):
        return torch.zeros(x.size(0), 1, self.input_dim[2])

    def encode(self, x):
        hidden_state = torch.zeros(self.n_layers*self.num_direction, x.size(0), self.n_hidden).cuda()
        cell_state = torch.zeros(self.n_layers*self.num_direction, x.size(0), self.n_hidden).cuda()
        

        TIME_LEN = x.size(1)
        output_list = []
        for t in range(TIME_LEN):
            encoder_input = x[:, t:t+1, :]
            o, (new_hidden_state, new_cell_state)= self.lstm_encoder(encoder_input, (hidden_state, cell_state))
            
            if self.USE_BN:
                hidden_state = torch.cat([self.bn_hidden_encoder(v1+v2).unsqueeze(0) for (v1, v2) in zip(new_hidden_state, hidden_state)], dim=0)
                cell_state = torch.cat([self.bn_cell_encoder(v1+v2).unsqueeze(0) for (v1, v2) in zip(new_cell_state, cell_state)], dim=0)
            else:
                hidden_state = new_hidden_state  + hidden_state
                cell_state = new_cell_state  + cell_state
            
            
            
            output_list.append(o)

        on = torch.cat(output_list, dim=1)
        if self.decoder_num_direction == 2:
            z = hidden_state
        else:
            z = hidden_state[0:1]
        return z, on

    def decode(self, x, z, on):
        output_list = []
        TIME_LEN = x.size(1)
        decoder_input = torch.zeros((x.size(0),1,)+ tuple(x.size()[2:])).cuda()
        hidden_state = z
        cell_state = torch.zeros(self.n_layers*self.decoder_num_direction, x.size(0), self.n_hidden).cuda()
        
        for t in range(TIME_LEN):
            o, (new_hidden_state, new_cell_state)= self.lstm_decoder(decoder_input, (hidden_state, cell_state))
            if self.USE_BN:
                hidden_state = torch.cat([self.bn_hidden_decoder(v1+v2).unsqueeze(0) for (v1, v2) in zip(new_hidden_state, hidden_state)], dim=0)
                cell_state = torch.cat([self.bn_cell_decoder(v1+v2).unsqueeze(0) for (v1, v2) in zip(new_cell_state, cell_state)], dim=0)
            else:
                hidden_state = new_hidden_state  + hidden_state
                cell_state = new_cell_state  + cell_state
            
            if self.decoder_num_direction == 2:
                hidden_for_output = torch.cat([hidden_state[0], hidden_state[1]], dim=1)
            else:
                hidden_for_output = hidden_state[0]
            
            output_t = self.fc_output(hidden_for_output)
            output_list.append(output_t.unsqueeze(1)) # add time axis
            #decoder_input = torch.zeros_like(x[:, t:t+1, :])

            discretizer = self.discretizer
            return_output_t = output_t.clone()
            for i in range(len(discretizer.begin_pos)):
                begin_pos_i = discretizer.begin_pos[i]
                end_pos_i = discretizer.end_pos[i]
                if end_pos_i - 1 != begin_pos_i:
                    USE_ONE_HOT = True
                    softmax_rec_cate_feature = torch.softmax(output_t[:, begin_pos_i:end_pos_i], dim=1)
                    if USE_ONE_HOT:
                        onehot = torch.zeros_like(softmax_rec_cate_feature)
                        onehot[:, torch.argmax(softmax_rec_cate_feature, dim=1)] = 1.0
                        return_output_t[:, begin_pos_i:end_pos_i] = onehot
                    else:
                        return_output_t[:, begin_pos_i:end_pos_i] = softmax_rec_cate_feature

            decoder_input = return_output_t.unsqueeze(1).detach()
        return torch.cat(output_list, axis=1)


    def forward(self, x):
        #Encoder
        z, on = self.encode(x)
        return self.decode(x, z, on)

    def loss(self, x, recon, discretizer):
        if self.eos_enabled:
            true_output = torch.cat([x, self.get_eos(x).cuda()], dim=1)
        else:
            true_output = x
            
        # Format 
        #flat_true_output = true_output.view(true_output.size(0)*true_output.size(1), true_output.size(2))
        #true_output = torch.flip(true_output, [1])
        flat_true_output = true_output.view(true_output.size(0)*true_output.size(1), true_output.size(2))
        flat_recon_output = recon.reshape(recon.size(0)*recon.size(1), recon.size(2))
        
        cont_loss = 0
        cate_loss = 0
        total_loss = 0
        
        for i in range(len(discretizer.begin_pos)):
            begin_pos_i = discretizer.begin_pos[i]
            end_pos_i = discretizer.end_pos[i]
            if end_pos_i - 1 == begin_pos_i:
                rec_cont_feature_i = flat_recon_output[:, begin_pos_i]
                inp_cont_feature_i = flat_true_output[:, begin_pos_i]
                loss_cont_i = torch.pow(rec_cont_feature_i - inp_cont_feature_i, 2).mean()
                cont_loss += loss_cont_i
            else:
                rec_cate_feature_i = flat_recon_output[:, begin_pos_i:end_pos_i]
                inp_cate_feature_i = flat_true_output[:, begin_pos_i:end_pos_i]
                inp_cate_label_i = torch.argmax(inp_cate_feature_i, dim=1)
                loss_cate_i = F.nll_loss(torch.log_softmax(rec_cate_feature_i, dim=1), inp_cate_label_i)
                cate_loss += loss_cate_i

            #print(begin_pos_i, end_pos_i)
        for i in range(discretizer.end_pos[-1], discretizer.end_pos[-1]+17):
            rec_cate_feature_i = flat_recon_output[:, i]
            inp_cate_feature_i = flat_true_output[:, i]
            mask_cate_loss_i = F.binary_cross_entropy_with_logits(rec_cate_feature_i, inp_cate_feature_i)
            cate_loss += mask_cate_loss_i
            
        total_loss = cont_loss + cate_loss
        return total_loss

#RNN_AE
class RNN_AE(torch.nn.Module):
    def __init__(self, input_dim, n_hidden, discretizer):
        super().__init__()
        self.lr=0.0001
        self.NUM_EPOCHS = 1000

        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.discretizer = discretizer
        dropout_p = 0.2

        self.rnn_encoder = torch.nn.Linear(self.input_dim[2]+self.n_hidden, self.n_hidden)
        self.bn_hidden_encoder = torch.nn.BatchNorm1d(self.n_hidden)

        self.rnn_decoder = torch.nn.Linear(self.input_dim[2]+self.n_hidden, self.n_hidden)
        self.bn_hidden_decoder = torch.nn.BatchNorm1d(self.n_hidden)
        self.fc_output = torch.nn.Linear(self.input_dim[2]+self.n_hidden, self.input_dim[2])

        self.dropout = torch.nn.Dropout(dropout_p)
        self.USE_BN=True

        # context vector - decoder hidden

    def encode(self, x):
        hidden_state = torch.zeros(x.size(0), self.n_hidden).cuda()

        TIME_LEN = x.size(1)

        for t in range(TIME_LEN):
            encoder_input = x[:, t, :]
            new_hidden_state = torch.tanh(self.rnn_encoder(torch.cat([encoder_input, hidden_state], dim=1)))
            
            if self.USE_BN:
                hidden_state = self.bn_hidden_encoder(new_hidden_state + hidden_state)
            else:
                hidden_state = new_hidden_state  + hidden_state       
            
        z = hidden_state
        return z

    def decode(self, x, z):
        output_list = []
        TIME_LEN = x.size(1)
        decoder_input = torch.zeros((x.size(0),)+ tuple(x.size()[2:])).cuda()
        hidden_state = z
        
        for t in range(TIME_LEN):
            new_hidden_state = torch.tanh(self.rnn_decoder(torch.cat([decoder_input, hidden_state], dim=1)))
            o = self.fc_output(torch.cat([decoder_input, new_hidden_state], dim=1))

            if self.USE_BN:
                hidden_state = self.bn_hidden_decoder(new_hidden_state + hidden_state)
            else:
                hidden_state = new_hidden_state  + hidden_state
            
            
            output_t = o
            output_list.append(output_t.unsqueeze(1)) # add time axis
            #decoder_input = torch.zeros_like(x[:, t:t+1, :])

            discretizer = self.discretizer
            return_output_t = output_t.clone()
            for i in range(len(discretizer.begin_pos)):
                begin_pos_i = discretizer.begin_pos[i]
                end_pos_i = discretizer.end_pos[i]
                if end_pos_i - 1 != begin_pos_i:
                    USE_ONE_HOT = True
                    softmax_rec_cate_feature = torch.softmax(output_t[:, begin_pos_i:end_pos_i], dim=1)
                    if USE_ONE_HOT:
                        onehot = torch.zeros_like(softmax_rec_cate_feature)
                        onehot[:, torch.argmax(softmax_rec_cate_feature, dim=1)] = 1.0
                        return_output_t[:, begin_pos_i:end_pos_i] = onehot
                    else:
                        return_output_t[:, begin_pos_i:end_pos_i] = softmax_rec_cate_feature

            decoder_input = return_output_t.detach()
        return torch.cat(output_list, axis=1)


    def forward(self, x):
        #Encoder
        z = self.encode(x)
        return self.decode(x, z)

    def loss(self, x, recon, discretizer):
        true_output = x
            
        # Format 
        #flat_true_output = true_output.view(true_output.size(0)*true_output.size(1), true_output.size(2))
        #true_output = torch.flip(true_output, [1])
        flat_true_output = true_output.view(true_output.size(0)*true_output.size(1), true_output.size(2))
        flat_recon_output = recon.reshape(recon.size(0)*recon.size(1), recon.size(2))
        
        cont_loss = 0
        cate_loss = 0
        total_loss = 0
        
        for i in range(len(discretizer.begin_pos)):
            begin_pos_i = discretizer.begin_pos[i]
            end_pos_i = discretizer.end_pos[i]
            if end_pos_i - 1 == begin_pos_i:
                rec_cont_feature_i = flat_recon_output[:, begin_pos_i]
                inp_cont_feature_i = flat_true_output[:, begin_pos_i]
                loss_cont_i = torch.pow(rec_cont_feature_i - inp_cont_feature_i, 2).mean()
                cont_loss += loss_cont_i
            else:
                rec_cate_feature_i = flat_recon_output[:, begin_pos_i:end_pos_i]
                inp_cate_feature_i = flat_true_output[:, begin_pos_i:end_pos_i]
                inp_cate_label_i = torch.argmax(inp_cate_feature_i, dim=1)
                loss_cate_i = F.nll_loss(torch.log_softmax(rec_cate_feature_i, dim=1), inp_cate_label_i)
                cate_loss += loss_cate_i

            #print(begin_pos_i, end_pos_i)
        for i in range(discretizer.end_pos[-1], discretizer.end_pos[-1]+17):
            rec_cate_feature_i = flat_recon_output[:, i]
            inp_cate_feature_i = flat_true_output[:, i]
            mask_cate_loss_i = F.binary_cross_entropy_with_logits(rec_cate_feature_i, inp_cate_feature_i)
            cate_loss += mask_cate_loss_i
            
        total_loss = cont_loss + cate_loss
        return total_loss

## LSTM_AE_ATT
class Attention(torch.nn.Module):

    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

    def forward(self, 
        query: torch.Tensor,  # [batch_size, decoder_dim]
        values: torch.Tensor, # [batch_size, seq_length, encoder_dim]
        ):
        weights = self._get_weights(query, values) # [batch_size, seq_length]
        #print("weights@v.size():", weights.size())
        weights = torch.nn.functional.softmax(weights, dim=1)
        
        # print("values.size():", values.size())
        # print("weights.unsqueeze(2).size():", weights.unsqueeze(2).size())
        #return weights.unsqueeze(2) @ values  # [batch_size, encoder_dim], context vector
        context_vector = (weights.unsqueeze(2) * values).mean(dim=1)
        #print("context_vector.size():", context_vector.size())
        return context_vector

class AdditiveAttention(Attention):
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__(encoder_dim, decoder_dim)
        self.v = torch.nn.Parameter(
            torch.FloatTensor(self.decoder_dim).uniform_(-0.1, 0.1))
        self.W_1 = torch.nn.Linear(self.decoder_dim, self.decoder_dim)
        self.W_2 = torch.nn.Linear(self.encoder_dim, self.decoder_dim)

    def _get_weights(self,        
        query: torch.Tensor,  # [batch_size, decoder_dim]
        values: torch.Tensor,  # [batch_size, seq_length, encoder_dim]
    ):
        
        #print("query.size():", query.size())
        if query.size()[0] > 1: # bidirectional
            query = torch.cat([query[0], query[1]], dim=1)
        else:
            query = query.unsqueeze(0)
        #print("query_cat.size():", query.size())
        query = query.unsqueeze(1)
        #print("query_cat_unsqueeze.size():", query.size())
        query = query.repeat(1, values.size(1), 1)  # [batch_size, seq_length, decoder_dim]
        #print("query_cat_unsqueeze_repeat.size():", query.size())
        #print("values.size():", values.size())
        weights = self.W_1(query) + self.W_2(values)  # [batch_size, seq_length, decoder_dim]
        #print("weights.size():", weights.size())
        return torch.tanh(weights) @ self.v  # [batch_size, seq_length]

class LSTM_AE_ATT(torch.nn.Module):
    def __init__(self, input_dim, n_hidden):
        super().__init__()
        self.lr=0.0001
        self.eos_enabled = False

        self.input_dim = input_dim
        self.n_layers = 1
        self.n_hidden = n_hidden
        self.num_direction = 2
        self.decoder_num_direction = 2
        assert self.num_direction in [1, 2]
        dropout_p = 0.2

        self.att = AdditiveAttention(self.n_hidden*self.num_direction, self.n_hidden*self.decoder_num_direction)

        self.lstm_encoder = torch.nn.LSTM(input_size=input_dim[2], hidden_size=self.n_hidden, num_layers=self.n_layers,\
                                    bias=True, batch_first=True,\
                                    dropout=dropout_p, bidirectional= True if self.num_direction == 2 else False)

        self.lstm_decoder = torch.nn.LSTM(input_size=input_dim[2], hidden_size=self.n_hidden, num_layers=self.n_layers,\
                                    bias=True, batch_first=True,\
                                    dropout=dropout_p, bidirectional= True if self.decoder_num_direction == 2 else False)

        
        self.dropout = torch.nn.Dropout(dropout_p)


        self.fc_output = torch.nn.Linear(self.n_hidden*self.decoder_num_direction + self.n_hidden*self.decoder_num_direction, input_dim[2])
        # context vector - decoder hidden

    def get_eos(self, x):
        return torch.zeros(x.size(0), 1, self.input_dim[2])

    def encode(self, x):
        hidden_init = torch.zeros(self.n_layers*self.num_direction, x.size(0), self.n_hidden).cuda()
        cell_init = torch.zeros(self.n_layers*self.num_direction, x.size(0), self.n_hidden).cuda()
        hidden_and_cell = (hidden_init, cell_init)
        # output batch_size x len_seqence x num_hidden*num_direction,  (last hidden, last cell)
        on, (hn, cn)=self.lstm_encoder(x, hidden_and_cell)
        """
        o = F.relu(self.dropout(self.fc1(on[:, -1, :])))
        mu = self.fc2_mu(o)
        
        z = mu 
        """
        z = hn
        #z = hn[-1].repeat(self.decoder_num_direction, 1, 1)
        return z, on

    def decode(self, x, z, on):
        """
        decoder_hidden_init = z
        decoder_cell_init = torch.zeros(self.n_layers*self.decoder_num_direction, x.size(0), self.n_hidden).cuda()
        decoder_hidden_and_cell = (decoder_hidden_init, decoder_cell_init)

        if self.eos_enabled:
            eos = self.get_eos(x).cuda()
            decoder_input = torch.cat([eos, x], dim=1)
        else:
            decoder_input = torch.zeros_like(x)

        on_decoder, (hn_decoder, cn_decoder) = self.lstm_decoder(decoder_input, decoder_hidden_and_cell)
        #return self.fc2_output(F.relu(self.fc1_output(on_decoder)))
        return self.fc_output(on_decoder)
        """
        output_list = []
        TIME_LEN = x.size(1)
        decoder_input = torch.zeros((x.size(0),1,)+ tuple(x.size()[2:])).cuda()
        hidden_state = z
        #print("hidden_state.size():", hidden_state.size())
        cell_state = torch.zeros(self.n_layers*self.decoder_num_direction, x.size(0), self.n_hidden).cuda()
        
        for t in range(TIME_LEN):
            context_vector_t = self.att(hidden_state, on)
            o, (hidden_state, cell_state)= self.lstm_decoder(decoder_input, (hidden_state, cell_state))
            #print("hidden_state.size():", hidden_state.size())
            #output_t = self.fc_output(torch.cat(hidden_state, context_vector_t))

            hidden_for_output = torch.cat([hidden_state[0], hidden_state[1]], dim=1)
            #print("context_vector_t.size():", context_vector_t.size())
            #print("hidden_for_output.size():", hidden_for_output.size())
            output_t = self.fc_output(torch.cat([hidden_for_output, context_vector_t], dim=1))
            #output_t = self.fc_output(torch.cat([hidden_for_output, hidden_for_output], dim=1))
            output_list.append(output_t.unsqueeze(1)) # add time axis
            #decoder_input = output_t.unsqueeze(1)
            decoder_input = x[:, t:t+1, :]
        return torch.cat(output_list, axis=1)


    def forward(self, x):
        #Encoder
        z, on = self.encode(x)
        return self.decode(x, z, on)

    def loss(self, x, recon, discretizer):
        if self.eos_enabled:
            true_output = torch.cat([x, self.get_eos(x).cuda()], dim=1)
        else:
            true_output = x
            
        # Format 
        #flat_true_output = true_output.view(true_output.size(0)*true_output.size(1), true_output.size(2))
        #true_output = torch.flip(true_output, [1])
        flat_true_output = true_output.view(true_output.size(0)*true_output.size(1), true_output.size(2))
        flat_recon_output = recon.reshape(recon.size(0)*recon.size(1), recon.size(2))
        
        cont_loss = 0
        cate_loss = 0
        total_loss = 0
        
        for i in range(len(discretizer.begin_pos)):
            begin_pos_i = discretizer.begin_pos[i]
            end_pos_i = discretizer.end_pos[i]
            if end_pos_i - 1 == begin_pos_i:
                rec_cont_feature_i = flat_recon_output[:, begin_pos_i]
                inp_cont_feature_i = flat_true_output[:, begin_pos_i]
                loss_cont_i = torch.pow(rec_cont_feature_i - inp_cont_feature_i, 2).mean()
                cont_loss += loss_cont_i
            else:
                rec_cate_feature_i = flat_recon_output[:, begin_pos_i:end_pos_i]
                inp_cate_feature_i = flat_true_output[:, begin_pos_i:end_pos_i]
                inp_cate_label_i = torch.argmax(inp_cate_feature_i, dim=1)
                loss_cate_i = F.nll_loss(torch.log_softmax(rec_cate_feature_i, dim=1), inp_cate_label_i)
                cate_loss += loss_cate_i

            #print(begin_pos_i, end_pos_i)
        for i in range(discretizer.end_pos[-1], discretizer.end_pos[-1]+17):
            rec_cate_feature_i = flat_recon_output[:, i]
            inp_cate_feature_i = flat_true_output[:, i]
            mask_cate_loss_i = F.binary_cross_entropy_with_logits(rec_cate_feature_i, inp_cate_feature_i)
            cate_loss += mask_cate_loss_i
            
        total_loss = cont_loss + cate_loss
        return total_loss


class LSTM_VAE_ATT(torch.nn.Module):
    def __init__(self, input_dim, n_hidden, discretizer, prob_teacher_forcing=1.0, use_attention=True):
        super().__init__()
        self.lr=0.0001
        self.eos_enabled = False

        self.input_dim = input_dim
        self.n_layers = 1
        self.n_hidden = n_hidden
        self.num_direction = 2
        self.decoder_num_direction = 2
        self.prob_teacher_forcing = prob_teacher_forcing
        self.discretizer = discretizer
        self.NUM_EPOCHS=1000 

        assert self.num_direction in [1, 2]

        self.use_attention = use_attention
        dropout_p = 0.2

        self.att = AdditiveAttention(self.n_hidden*self.num_direction, self.n_hidden*self.decoder_num_direction)

        self.lstm_encoder = torch.nn.LSTM(input_size=input_dim[2], hidden_size=self.n_hidden, num_layers=self.n_layers,\
                                    bias=True, batch_first=True,\
                                    dropout=dropout_p, bidirectional= True if self.num_direction == 2 else False)

        self.lstm_decoder = torch.nn.LSTM(input_size=input_dim[2], hidden_size=self.n_hidden, num_layers=self.n_layers,\
                                    bias=True, batch_first=True,\
                                    dropout=dropout_p, bidirectional= True if self.decoder_num_direction == 2 else False)

        
        self.dropout = torch.nn.Dropout(dropout_p)

        self.fc_hidden_mu = torch.nn.Linear(self.n_hidden*self.num_direction, self.n_hidden*self.num_direction)
        self.fc_hidden_logvar = torch.nn.Linear(self.n_hidden*self.num_direction, self.n_hidden*self.num_direction)

        self.fc_z_mu = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.fc_z_logvar = torch.nn.Linear(self.n_hidden, self.n_hidden)

        self.fc_output = torch.nn.Linear(self.n_hidden*self.decoder_num_direction + self.n_hidden*self.decoder_num_direction, input_dim[2])
        
        

    def get_eos(self, x):
        return torch.zeros(x.size(0), 1, self.input_dim[2])

    def encode(self, x):
        hidden_init = torch.zeros(self.n_layers*self.num_direction, x.size(0), self.n_hidden).cuda()
        cell_init = torch.zeros(self.n_layers*self.num_direction, x.size(0), self.n_hidden).cuda()
        hidden_and_cell = (hidden_init, cell_init)
        # output batch_size x len_seqence x num_hidden*num_direction,  (last hidden, last cell)
        on, (hn, cn)=self.lstm_encoder(x, hidden_and_cell)
        
        # Reparameterization - z
        mu_z = self.fc_z_mu(hn)
        logvar_z = self.fc_z_logvar(hn)
        z = mu_z + torch.randn_like(logvar_z)*torch.exp(0.5*logvar_z)

        # Reparameterization - hidden
        mu_on = self.fc_hidden_mu(on)
        logvar_on = self.fc_hidden_logvar(on)
        on = mu_on + torch.randn_like(logvar_on)*torch.exp(0.5*logvar_on)
        return (z, on), (mu_z, mu_on), (logvar_z, logvar_on)

    def decode(self, x, z, on, prob_teacher_forcing, use_attention=True):
        output_list = []
        TIME_LEN = x.size(1)
        decoder_input = torch.zeros((x.size(0),1,)+ tuple(x.size()[2:])).cuda()
        hidden_state = z
        cell_state = torch.zeros(self.n_layers*self.decoder_num_direction, x.size(0), self.n_hidden).cuda()
        discretizer = self.discretizer
        for t in range(TIME_LEN):
            if use_attention:
                context_vector_t = self.att(hidden_state, on)
            o, (hidden_state, cell_state)= self.lstm_decoder(decoder_input, (hidden_state, cell_state))
            
            if self.decoder_num_direction == 2:
                hidden_for_output = torch.cat([hidden_state[0], hidden_state[1]], dim=1)
            else:
                hidden_for_output = hidden_state[0]

            if use_attention:
                output_t = self.fc_output(torch.cat([hidden_for_output, context_vector_t], dim=1))
            else:
                output_t = self.fc_output(torch.cat([hidden_for_output, hidden_for_output], dim=1))
            
            output_list.append(output_t.unsqueeze(1)) # add time axis

            rand = random.uniform(0, 1)

            if rand < prob_teacher_forcing :
                decoder_input = x[:, t:t+1, :]
            else:
                return_output_t = output_t.clone()
                for i in range(len(discretizer.begin_pos)):
                    begin_pos_i = discretizer.begin_pos[i]
                    end_pos_i = discretizer.end_pos[i]
                    if end_pos_i - 1 != begin_pos_i:
                        USE_ONE_HOT = True
                        softmax_rec_cate_feature = torch.softmax(output_t[:, begin_pos_i:end_pos_i], dim=1)
                        if USE_ONE_HOT:
                            onehot = torch.zeros_like(softmax_rec_cate_feature)
                            onehot[:, torch.argmax(softmax_rec_cate_feature, dim=1)] = 1.0
                            return_output_t[:, begin_pos_i:end_pos_i] = onehot
                        else:
                            return_output_t[:, begin_pos_i:end_pos_i] = softmax_rec_cate_feature

                decoder_input = return_output_t.unsqueeze(1).detach()
                        

                
        return torch.cat(output_list, axis=1)


    def forward(self, x):
        #Encoder
        (z, on), (mu_z, mu_on), (logvar_z, logvar_on) = self.encode(x)
        return self.decode(x, z, on, self.prob_teacher_forcing, self.use_attention), (z, on), (mu_z, mu_on), (logvar_z, logvar_on)

    def loss(self, x, recon, mu, logvar, kl_weight):
        if self.eos_enabled:
            true_output = torch.cat([x, self.get_eos(x).cuda()], dim=1)
        else:
            true_output = x
            
        # Format 
        #flat_true_output = true_output.view(true_output.size(0)*true_output.size(1), true_output.size(2))
        #true_output = torch.flip(true_output, [1])
        flat_true_output = true_output.view(true_output.size(0)*true_output.size(1), true_output.size(2))
        flat_recon_output = recon.reshape(recon.size(0)*recon.size(1), recon.size(2))
        
        cont_loss = 0
        cate_loss = 0
        total_loss = 0
        discretizer = self.discretizer

        for i in range(len(discretizer.begin_pos)):
            begin_pos_i = discretizer.begin_pos[i]
            end_pos_i = discretizer.end_pos[i]
            if end_pos_i - 1 == begin_pos_i:
                rec_cont_feature_i = flat_recon_output[:, begin_pos_i]
                inp_cont_feature_i = flat_true_output[:, begin_pos_i]
                loss_cont_i = torch.pow(rec_cont_feature_i - inp_cont_feature_i, 2).mean()
                cont_loss += loss_cont_i
            else:
                rec_cate_feature_i = flat_recon_output[:, begin_pos_i:end_pos_i]
                inp_cate_feature_i = flat_true_output[:, begin_pos_i:end_pos_i]
                inp_cate_label_i = torch.argmax(inp_cate_feature_i, dim=1)
                loss_cate_i = F.nll_loss(torch.log_softmax(rec_cate_feature_i, dim=1), inp_cate_label_i)
                cate_loss += loss_cate_i

            #print(begin_pos_i, end_pos_i)
        for i in range(discretizer.end_pos[-1], discretizer.end_pos[-1]+17):
            rec_cate_feature_i = flat_recon_output[:, i]
            inp_cate_feature_i = flat_true_output[:, i]
            mask_cate_loss_i = F.binary_cross_entropy_with_logits(rec_cate_feature_i, inp_cate_feature_i)
            cate_loss += mask_cate_loss_i
        
        # KL loss
        mu_z, logvar_z = mu[0], logvar[0]
        mu_on, logvar_on = mu[1], logvar[1]

        # Size: [batch_size + batch_size*src_len , 64]
        if self.use_attention:
            flat_mu = torch.cat([torch.cat([mu_z[0], mu_z[1]], dim=1),
                                mu_on.view(mu_on.size(0)*mu_on.size(1), mu_on.size(2))], dim=0)
            flat_logvar = torch.cat([torch.cat([logvar_z[0], logvar_z[1]], dim=1),
                                mu_on.view(logvar_on.size(0)*logvar_on.size(1), logvar_on.size(2))], dim=0)
        else:    
            flat_mu = torch.cat([mu_z[0], mu_z[1]], dim=1)
            flat_logvar = torch.cat([logvar_z[0], logvar_z[1]], dim=1)

        kl_loss = torch.mean(-0.5 * torch.sum(1 + flat_logvar - flat_mu ** 2 - flat_logvar.exp(), dim = 1), dim = 0)
        recon_loss = cont_loss + cate_loss
        total_loss = recon_loss + kl_weight*kl_loss
        return total_loss, recon_loss, kl_loss

    def get_model_file_name(self):
        return "lstm_vae_48_76_latent_{}_{}_{}.pt".format(self.n_hidden, self.prob_teacher_forcing, self.use_attention)

class ResBlockLinearSkip(torch.nn.Module):
    def __init__(self, input_dim, output_dim, use_bn=True):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)
        self.fc_skip = torch.nn.Linear(input_dim, output_dim)
        self.bn = torch.nn.BatchNorm1d(output_dim)
        self.use_bn = use_bn
    def forward(self, x):
        if self.use_bn:
            return self.bn(F.relu(self.fc(x)) + self.fc_skip(x))
        else:
            return F.relu(self.fc(x)) + self.fc_skip(x)

class ResBlockConvSkip(torch.nn.Module):
    def __init__(self, in_channel, out_channel, stride, use_bn=False, use_1x1=False):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.conv_skip = torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channel)
        self.conv2 = torch.nn.Conv2d(1, in_channel, kernel_size=1, stride=1)
        self.use_1x1 = use_1x1
        self.use_bn = use_bn

    def forward(self, x):
        if self.use_1x1:
            x = self.conv2(x)
        
        if self.use_bn:
            return self.bn1(F.relu(self.conv1(x)) + self.conv_skip(x))
        else:
            return F.relu(self.conv1(x)) + self.conv_skip(x)

class ResBlockConvTransSkip(torch.nn.Module):
    def __init__(self, in_channel, out_channel, stride, use_bn=False, use_1x1=False, padding=0, output_padding=0):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=padding, output_padding=output_padding)
        self.conv_skip = torch.nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=padding, output_padding=output_padding)
        self.bn1 = torch.nn.BatchNorm2d(out_channel)
        self.conv2 = torch.nn.ConvTranspose2d(1, in_channel, kernel_size=1, stride=1)
        self.use_1x1 = use_1x1
        self.use_bn = use_bn

    def forward(self, x):
        if self.use_1x1:
            x = self.conv2(x)
        
        if self.use_bn:
            return self.bn1(F.relu(self.conv1(x)) + self.conv_skip(x))
        else:
            return F.relu(self.conv1(x)) + self.conv_skip(x)


class CNN_AE(torch.nn.Module):
    def __init__(self, input_dim, n_hidden, discretizer):
        super().__init__()
        self.lr = 0.00001
        self.NUM_EPOCHS=1000
        """
        #Linear layers with multiples of hidden dimensions NOT worked
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            ResBlock(48*76),
            torch.nn.Linear(48*76, 24*76),
            ResBlock(24*76),
            torch.nn.Linear(24*76, 12*76),
            ResBlock(12*76),
            torch.nn.Linear(12*76, n_hidden)
        )
        self.decoder = torch.nn.Sequential(
            ResBlock(n_hidden),
            torch.nn.Linear(n_hidden, 12*76),
            ResBlock(12*76),
            torch.nn.Linear(12*76, 24*76),
            ResBlock(24*76),
            torch.nn.Linear(24*76, 48*76),
            Reshape(-1,48,76)
            #torch.nn.ConvTranspose2d(32, 1, kernel_size=(26, 10), stride=1),
        )
        """
        """
        # Good
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            ResBlockLinearSkip(48*76, 24*76),
            ResBlockLinearSkip(24*76, 12*76),
            ResBlockLinearSkip(12*76, n_hidden)
        )
        self.decoder = torch.nn.Sequential(
            ResBlock(n_hidden),
            ResBlockLinearSkip(n_hidden, 12*76),
            ResBlockLinearSkip(12*76, 24*76),
            ResBlockLinearSkip(24*76, 48*76, False),
            Reshape(-1,48,76)
            #torch.nn.ConvTranspose2d(32, 1, kernel_size=(26, 10), stride=1),
        )
        """
        #conv2d res, conv2d skip
        self.encoder = torch.nn.Sequential(
            Reshape(-1, 1, 48, 76),
            ResBlockConvSkip(4, 8, stride=2, use_bn=True, use_1x1=True), # 8, 24, 38
            ResBlockConvSkip(8, 16, stride=2, use_bn=True, use_1x1=False), # 16, 12, 19
            ResBlockConvSkip(16, 32, stride=2, use_bn=True, use_1x1=False), # 32, 6, 10
            Reshape(-1, 32*6*10),
            ResBlockLinearSkip(32*6*10, n_hidden, use_bn=True),
        )
        self.decoder_channel = 64
        self.decoder = torch.nn.Sequential(
            ResBlockLinearSkip(n_hidden, self.decoder_channel*5*8),
            Reshape(-1, self.decoder_channel, 5, 8),
            ResBlockConvTransSkip(self.decoder_channel, 32, stride=2, use_bn=True, use_1x1=False, output_padding=(0, 1)), # 16, 11, 18
            ResBlockConvTransSkip(32, 16, stride=2, use_bn=True, use_1x1=False), # 8, 23, 37
            ResBlockConvTransSkip(16, 1, stride=2, use_bn=False, use_1x1=False, output_padding=1), # 1, 48, 76
            Reshape(-1, 48, 76)
            #torch.nn.ConvTranspose2d(32, 1, kernel_size=(26, 10), stride=1),
        )
        """
        #Linear layers with multiples of hidden dimensions worked
        # self.encoder = torch.nn.Sequential(
        #     torch.nn.Flatten(start_dim=1),
        #     torch.nn.Linear(48*76, n_hidden)
        # )
        # self.decoder = torch.nn.Sequential(
        #     torch.nn.Linear(n_hidden, 48*76),
        #     Reshape(-1,48,76)
        #     #torch.nn.ConvTranspose2d(32, 1, kernel_size=(26, 10), stride=1),
        # )
        #"""
        # Conv ReLU
        # self.encoder = torch.nn.Sequential(
        #     Reshape(-1, 1, 48, 76),
        #     torch.nn.Conv2d(1, 16, 3, stride=2), # 8, 24, 38
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(16, 32, 3, stride=2), # 16, 12, 19
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(32, 32, 3, stride=2), # 32, 6, 10
        #     torch.nn.ReLU(),
        #     Reshape(-1, 32*5*8),
        #     torch.nn.Linear(32*5*8, n_hidden),
        # )
        # self.decoder_channel = 64
        # self.decoder = torch.nn.Sequential(
        #     torch.nn.Linear(n_hidden, self.decoder_channel*5*8),
        #     torch.nn.ReLU(),
        #     Reshape(-1, self.decoder_channel, 5, 8),
        #     torch.nn.ConvTranspose2d(self.decoder_channel, 32, 3, stride=2, output_padding=(0, 1)), # 16, 11, 18
        #     torch.nn.ReLU(),
        #     torch.nn.ConvTranspose2d(32, 16, 3, stride=2), # 8, 23, 37
        #     torch.nn.ReLU(),
        #     torch.nn.ConvTranspose2d(16, 1, 3, stride=2, output_padding=1), # 1, 48, 76
        #     Reshape(-1, 48, 76)
        #     #torch.nn.ConvTranspose2d(32, 1, kernel_size=(26, 10), stride=1),
        # )

        # FC relu
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(48*76, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Linear(256, n_hidden)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 48*76),
            Reshape(-1,48,76)
            #torch.nn.ConvTranspose2d(32, 1, kernel_size=(26, 10), stride=1),
        )
        #dummy for compatibility
        self.input_dim = input_dim
        self.prob_teacher_forcing = -9999
        self.discretizer = discretizer
        self.n_hidden = n_hidden
    def encode(self, x):
       
        z = self.encoder(x) #+ torch.randn_like(logvar)*torch.exp(0.5*logvar)

        return z

    def decode(self, x, z):
        o = self.decoder(z)
        return o


    def forward(self, x):
        #Encoder
        z = self.encode(x)
        o = self.decode(x, z)
        #print(x.size())
        #print(z.size())
        #print(o.size())
        
        return o

    def loss(self, x, recon, discretizer):    
        # Format 
        flat_true_output = x.view(x.size(0)*x.size(1), x.size(2))
        flat_recon_output = recon.reshape(recon.size(0)*recon.size(1), recon.size(2))
        
        cont_loss = 0
        cate_loss = 0
        total_loss = 0
        discretizer = self.discretizer

        for i in range(len(discretizer.begin_pos)):
            begin_pos_i = discretizer.begin_pos[i]
            end_pos_i = discretizer.end_pos[i]
            if end_pos_i - 1 == begin_pos_i:
                rec_cont_feature_i = flat_recon_output[:, begin_pos_i]
                inp_cont_feature_i = flat_true_output[:, begin_pos_i]
                loss_cont_i = torch.pow(rec_cont_feature_i - inp_cont_feature_i, 2).mean()
                cont_loss += loss_cont_i
            else:
                rec_cate_feature_i = flat_recon_output[:, begin_pos_i:end_pos_i]
                inp_cate_feature_i = flat_true_output[:, begin_pos_i:end_pos_i]
                inp_cate_label_i = torch.argmax(inp_cate_feature_i, dim=1)
                loss_cate_i = F.nll_loss(torch.log_softmax(rec_cate_feature_i, dim=1), inp_cate_label_i)
                cate_loss += loss_cate_i

            #print(begin_pos_i, end_pos_i)
        for i in range(discretizer.end_pos[-1], discretizer.end_pos[-1]+17):
            rec_cate_feature_i = flat_recon_output[:, i]
            inp_cate_feature_i = flat_true_output[:, i]
            mask_cate_loss_i = F.binary_cross_entropy_with_logits(rec_cate_feature_i, inp_cate_feature_i)
            cate_loss += mask_cate_loss_i
        
        # KL loss
        # Size: [batch_size + batch_size*src_len , 64]
        # print(mu.size(), logvar.size())
        # kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        recon_loss = cate_loss + cont_loss #cate_loss#cont_loss + cate_loss
        total_loss = recon_loss #+ kl_weight*kl_loss
        return total_loss
        #return F.mse_loss(recon, x), 0, 0

    def get_model_file_name(self):
        return "cnn_ae_48_76_latent_{}.pt".format(self.n_hidden)

class Reshape(torch.nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class ResBlock(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        return F.relu(self.fc(x)) + x

class CNN_AE_48_17(torch.nn.Module):
    def __init__(self, input_dim, n_hidden, discretizer):
        super().__init__()
        self.lr = 0.00001 #0.00001
        self.n_hidden = n_hidden
        self.discretizer = discretizer
        dropout_p = 0.2
        self.dropout = torch.nn.Dropout(dropout_p)
        """
        # loss: 0.26317641139030457
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, (3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 32, (3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 32, (3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 32, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(32, 32, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(10*2*32, n_hidden)
        )
        self.hidden_channel = 1
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, 11*3*self.hidden_channel),
            torch.nn.BatchNorm1d(11*3*self.hidden_channel),
            Reshape(-1, self.hidden_channel, 11, 3),
            torch.nn.ConvTranspose2d(self.hidden_channel, 32, kernel_size=(3, 4), stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 1, kernel_size=(4, 3), stride=2),
            #torch.nn.ConvTranspose2d(32, 1, kernel_size=(26, 10), stride=1),
            
        )
        """
        

        """
        #Worked
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(48*17, n_hidden)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, 48*17),
            Reshape(-1,48,17)
            #torch.nn.ConvTranspose2d(32, 1, kernel_size=(26, 10), stride=1),
        )
        """
        """
        #Worked
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(48*17, n_hidden)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, 48*17),
            Reshape(-1,48,17)
            #torch.nn.ConvTranspose2d(32, 1, kernel_size=(26, 10), stride=1),
        )
        """
        """
        #Resblock Workedd
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            ResBlock(48*17),
            ResBlock(48*17),
            ResBlock(48*17),
            ResBlock(48*17),
            torch.nn.Linear(48*17, n_hidden)
        )
        self.decoder = torch.nn.Sequential(
            ResBlock(n_hidden),
            ResBlock(n_hidden),
            ResBlock(n_hidden),
            ResBlock(n_hidden),
            torch.nn.Linear(n_hidden, 48*17),
            Reshape(-1,48,17)
            #torch.nn.ConvTranspose2d(32, 1, kernel_size=(26, 10), stride=1),
        )
        """
        
        """
        #Linear layers with multiples of hidden dimensions worked
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            ResBlock(48*17),
            torch.nn.Linear(48*17, 24*17),
            ResBlock(24*17),
            torch.nn.Linear(24*17, 12*17),
            ResBlock(12*17),
            torch.nn.Linear(12*17, n_hidden)
        )
        self.decoder = torch.nn.Sequential(
            ResBlock(n_hidden),
            torch.nn.Linear(n_hidden, 12*17),
            ResBlock(12*17),
            torch.nn.Linear(12*17, 24*17),
            ResBlock(24*17),
            torch.nn.Linear(24*17, 48*17),
            Reshape(-1,48,17)
            #torch.nn.ConvTranspose2d(32, 1, kernel_size=(26, 10), stride=1),
        )
        """

        """
        # Not Good
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(48*17, 512),
            torch.nn.Dropout(0.3),
            #torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.Dropout(0.3),
            #torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_hidden),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, 256),
            #torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            #torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 48*17),
            Reshape(-1,48,17)
            #torch.nn.ConvTranspose2d(32, 1, kernel_size=(26, 10), stride=1),
        )
        """
        # Not Good
        # 0.14244899153709412 with ReLU (Bad)
        # 0.09422598779201508 with near hidden ReLU
        # 0.09435948729515076 with near input  ReLU
        # 0.16597877442836761 with near input  Sigmoid
        # 0.07821009308099747 with near input  Tanh
        # 0.07180200517177582 w/o  ReLU (Good)
        # 0.07622618973255157 with before Sigmoid (decoder) and after Sigmoid (encoder) batch normalization
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(48*17, 512),
            #torch.nn.Sigmoid(),
            torch.nn.ReLU(),
            #torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 256),
            #torch.nn.Sigmoid(),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_hidden),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, 256),
            #torch.nn.Sigmoid(),
            torch.nn.ReLU(),
            #torch.nn.BatchNorm1d(256),
            torch.nn.Linear(256, 512),
            #torch.nn.Sigmoid(),
            torch.nn.ReLU(),
            #torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 48*17),
            Reshape(-1,48,17)
            #torch.nn.ConvTranspose2d(32, 1, kernel_size=(26, 10), stride=1),
        )
        """
        # Not work if we add Relu or dropout
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(48*17, 512),
            #torch.nn.Dropout(0.3),
            #torch.nn.BatchNorm1d(512),
            #torch.nn.ReLU(),
            #torch.nn.Sigmoid(),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 256),
            #torch.nn.Dropout(0.3),
            #torch.nn.BatchNorm1d(256),
            #torch.nn.ReLU(),
            #torch.nn.Sigmoid(),
            torch.nn.Tanh(),
            torch.nn.Linear(256, n_hidden),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, 256),
            #torch.nn.BatchNorm1d(256),
            #torch.nn.Dropout(0.3),
            #torch.nn.ReLU(),
            #torch.nn.Sigmoid(),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 512),
            #torch.nn.BatchNorm1d(512),
            #torch.nn.Dropout(0.3),
            #torch.nn.ReLU(),
            #torch.nn.Sigmoid(),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 48*17),
            Reshape(-1,48,17)
            #torch.nn.ConvTranspose2d(32, 1, kernel_size=(26, 10), stride=1),
        )
        """
        """
        # Worked
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(48*17, 512),
            torch.nn.Linear(512, 256),
            torch.nn.Linear(256, n_hidden),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, 256),
            torch.nn.Linear(256, 512),
            torch.nn.Linear(512, 48*17),
            Reshape(-1,48,17)
            #torch.nn.ConvTranspose2d(32, 1, kernel_size=(26, 10), stride=1),
        )
        """

        #dummy for compatibility
        self.input_dim = input_dim
        self.prob_teacher_forcing = -9999

    def encode(self, x):
        x = x.unsqueeze(dim=1)
        z = self.encoder(x)
        return z

    def decode(self, x, z):
        o = self.decoder(z)
        o = o.squeeze(dim=1)
        return o


    def forward(self, x):
        #Encoder
        z  = self.encode(x)
        return self.decode(x, z)

    def loss(self, x, recon):    
        # Format 
        flat_true_output = x.view(x.size(0)*x.size(1), x.size(2))
        flat_recon_output = recon.reshape(recon.size(0)*recon.size(1), recon.size(2))
        
        total_loss = F.mse_loss(flat_true_output, flat_recon_output)
        NUM_FEATURES = flat_true_output.size(1)

        con_loss_list = []
        cat_loss_list = []
        for i in range(NUM_FEATURES):
            ith_loss = F.mse_loss(flat_true_output[:, i], flat_recon_output[:, i])
            if self.discretizer._is_categorical_channel[self.discretizer._id_to_channel[i]]:
                cat_loss_list.append(ith_loss.unsqueeze(0))
            else:
                con_loss_list.append(ith_loss.unsqueeze(0))
        con_loss = torch.cat(con_loss_list, dim=0).mean()
        cat_loss = torch.cat(cat_loss_list, dim=0).mean()
        # print(con_loss.item())
        # print(cat_loss.item())
        return  total_loss

    def get_model_file_name(self):
        return "cnn_vae_48_17_latent_{}.pt".format(self.n_hidden)



class CNN_AE_MNIST(torch.nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        dropout_p = 0.2
        self.lr = 0.00001
        self.dropout = torch.nn.Dropout(dropout_p)
        
        self.conv1_encoder = torch.nn.Conv2d(1, 32, (3, 3))
        self.bm1_encoder = torch.nn.BatchNorm2d(32)
        self.conv2_encoder = torch.nn.Conv2d(32, 32, (3, 3))
        self.bm2_encoder = torch.nn.BatchNorm2d(32)
        self.fc1_mu_encoder = torch.nn.Linear(5*5*32, n_hidden)
        self.fc1_logvar_encoder = torch.nn.Linear(5*5*32, n_hidden)

        
        self.hidden_channel=1
        self.fc1_decoder = torch.nn.Linear(n_hidden, 6*6*self.hidden_channel) 
        self.bm1_decoder = torch.nn.BatchNorm1d(6*6*self.hidden_channel)
        self.convtranspose1_decoder = torch.nn.ConvTranspose2d(self.hidden_channel, 32, kernel_size=(3, 3), stride=2)
        self.bm2_decoder = torch.nn.BatchNorm2d(32)
        self.convtranspose2_decoder = torch.nn.ConvTranspose2d(32, 1, kernel_size=(4, 4), stride=2)
        
        #dummy for compatibility
        self.prob_teacher_forcing = -9999

    def encode(self, x):
        o = F.max_pool2d(F.relu(self.conv1_encoder(x)), 2)#self.bm1_encoder(F.max_pool2d(F.relu(self.conv1_encoder(x)), 2))
        o = F.max_pool2d(F.relu(self.conv2_encoder(o)), 2)
        o = o.view(x.size(0), -1)

        mu = self.fc1_mu_encoder(o)
        logvar = self.fc1_logvar_encoder(o)

        z = mu #+ torch.randn_like(logvar)*torch.exp(0.5*logvar)

        return z, mu, logvar

    def decode(self, x, z):
        o = self.bm1_decoder(self.fc1_decoder(z))
        o = o.view(z.size(0), self.hidden_channel, 6, 6)
        o = F.relu(self.convtranspose1_decoder(o))
        o = self.convtranspose2_decoder(o)
        o = torch.sigmoid(o)
        return o


    def forward(self, x):
        #Encoder
        z, mu, logvar = self.encode(x)
        return self.decode(x, z)

    def loss(self, x, recon):    
        # Format 
        flat_true_output = x.view(x.size(0)*x.size(2), x.size(3))
        flat_recon_output = recon.reshape(recon.size(0)*recon.size(2), recon.size(3))
        
        #total_loss = F.mse_loss(flat_true_output, flat_recon_output)
        total_loss = F.mse_loss(x, recon)
        #loss_ = torch.nn.BCELoss()
        #total_loss = loss_(recon, x)
        return total_loss

    def get_model_file_name(self):
        return "cnn_ae_MNIST_{}.pt".format(self.n_hidden)