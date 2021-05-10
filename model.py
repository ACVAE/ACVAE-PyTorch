import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import rnn


class AddEps(nn.Module):
    def __init__(self, channels):
        super(AddEps, self).__init__()

        self.channels = channels
        self.linear = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Tanh()
        )

    def forward(self, x):
        eps = torch.randn_like(x)
        eps = self.linear(eps)

        return eps + x


class FCEncoder(nn.Module):
    def __init__(self, in_shape, out_shape):
        super(FCEncoder, self).__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(in_shape, in_shape),
            nn.Softplus()
        )
        self.eps = AddEps(in_shape)
        self.linear2 = nn.Sequential(
            nn.Linear(in_shape, in_shape),
            nn.Softplus()
        )
        self.linear_o = nn.Linear(in_shape, out_shape)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        y = self.eps(self.linear1(x))
        y = self.eps(self.linear2(y))
        y = self.dropout(y)
        y = y + x
        out = self.linear_o(y)

        return out


class FCEncoderNoRes(nn.Module):
    def __init__(self, in_shape, out_shape):
        super(FCEncoderNoRes, self).__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(in_shape, in_shape),
            nn.Softplus()
        )
        self.eps = AddEps(in_shape)
        self.linear2 = nn.Sequential(
            nn.Linear(in_shape, in_shape),
            nn.Softplus()
        )
        self.linear_o = nn.Linear(in_shape, out_shape)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        y = self.eps(self.linear1(x))
        y = self.eps(self.linear2(y))
        y = self.dropout(y)

        out = self.linear_o(y)

        return out


class FCEncoderCNN(nn.Module):
    def __init__(self, hyper_params, in_shape, out_shape):
        super(FCEncoderCNN, self).__init__()

        self.hyper_params = hyper_params
        self.dot_cnn1 = nn.Sequential(
            nn.Conv1d(in_shape, in_shape, kernel_size=1, stride=1),
            nn.Softplus()
        )
        self.dot_cnn2 = nn.Sequential(
            nn.Conv1d(2 * in_shape, in_shape, kernel_size=1, stride=1),
            nn.Softplus()
        )
        self.cnn_layer = nn.Sequential(
            nn.Conv1d(in_shape, 2*in_shape,
                      kernel_size=5, stride=1, padding=4),
            nn.Softplus()
        )

        self.eps = AddEps(in_shape)
        self.linear_o = nn.Linear(in_shape, out_shape)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor):
        if self.hyper_params['add_eps']:
            y = self.dot_cnn1(self.eps(x).transpose(1, 2)).transpose(1, 2)
            y = self.cnn_layer(self.eps(y).transpose(1, 2))[:, :, :-4]
        else:
            y = self.dot_cnn1(x.transpose(1, 2)).transpose(1, 2)
            y = self.cnn_layer(y.transpose(1, 2))[:, :, :-4]
        y = self.dot_cnn2(y).transpose(1, 2)
        y = x + y

        y = self.dropout(y)
        out = self.linear_o(y)

        return out


class Encoder(nn.Module):
    def __init__(self, hyper_params):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(
            hyper_params['rnn_size'] * 2 + 32, hyper_params['hidden_size']
        )
        self.linear2 = nn.Linear(
            3*hyper_params['rnn_size'], hyper_params['hidden_size']
        )
        nn.init.xavier_normal_(self.linear1.weight)
        self.activation = nn.Tanh()

        self.no_cnn = nn.Linear(
            hyper_params['rnn_size'], hyper_params['hidden_size']
        )

        nn.init.xavier_normal_(self.no_cnn.weight)
        # 1 x 1 x seq_len x rnn_size ==> 1 x n x seq_len x rnn_size
        self.Horizontal_cnn = nn.Conv2d(
            in_channels=1,
            out_channels=32,  # n
            kernel_size=(1, 16),
            stride=(1, 1)
        )

        self.relu = nn.ReLU()

        # 1 x 1 x seq_len x rnn_size ==> 1 x 1 x seq_len x rnn_size
        self.Vertical_cnn = nn.Conv2d(
            in_channels=1,  # k
            out_channels=1,
            kernel_size=(4, 1),
            stride=(1, 1),
        )
        self.hyper_params = hyper_params
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        s = torch.zeros(
            [x.shape[0], 1, 3, self.hyper_params['rnn_size']], device=x.device)
        q = torch.cat((s, x.unsqueeze(1)), 2)
        q = self.Vertical_cnn(q)
        q = q * x.unsqueeze(1)  # [batch_size x 1 x seq_len x rnn_size]
        q = q.squeeze(1)

        # [batch_size x n x seq_len x rnn_size]
        p = self.Horizontal_cnn(x.unsqueeze(1))
        p = self.relu(p)
        p = torch.sum(p, dim=3).transpose(1, 2)  # [batch_size x seq_len x n]

        # [batch_size x seq_len x (2rnn_size + 32)]
        m = torch.cat((p, x, q), dim=2)
        x = self.linear1(m)
        x = self.activation(x)
        x = self.dropout(x)

        return x


class Decoder(nn.Module):
    def __init__(self, hyper_params):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(
            hyper_params['latent_size'], hyper_params['item_embed_size'])
        self.linear2 = nn.Linear(
            hyper_params['item_embed_size'], hyper_params['total_items'] + 1)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        out_embed = x

        x = self.activation(x)
        x = self.linear2(x)
        return x, out_embed


class Model(nn.Module):
    def __init__(self, hyper_params):
        super(Model, self).__init__()
        self.hyper_params = hyper_params

        # self.encoder = Encoder(hyper_params)
        self.decoder = Decoder(hyper_params)

        self.item_embed = nn.Embedding(
            hyper_params['total_items'] + 1, hyper_params['item_embed_size'])

        self.gru = nn.GRU(
            hyper_params['item_embed_size'], hyper_params['rnn_size'],
            batch_first=True, num_layers=1
        )

        self.linear_o = nn.Linear(
            hyper_params['hidden_size'], hyper_params['latent_size'])
        self.linear1 = nn.Linear(
            hyper_params['hidden_size'], 2 * hyper_params['latent_size'])
        nn.init.xavier_normal_(self.linear1.weight)

        self.tanh = nn.Tanh()
        self.embed_dropout = nn.Dropout(0.5)

        if hyper_params['model_func'] == 'fc':
            self.encoder = FCEncoder(
                hyper_params['rnn_size'], hyper_params['latent_size'])
        elif hyper_params['model_func'] == 'fc_cnn':
            self.encoder = FCEncoderCNN(hyper_params,
                                        hyper_params['rnn_size'], hyper_params['latent_size'])
        elif hyper_params['model_func'] == 'fc_no_res':
            self.encoder = FCEncoderNoRes(
                hyper_params['rnn_size'], hyper_params['latent_size'])
        else:
            print(f'Illegal model function: {hyper_params["model_func"]}.')
            raise NotImplementedError()

    def sample_latent(self, z_inferred):
        return torch.randn_like(z_inferred)

    def forward(self, x):
        x = self.item_embed(x)
        x_real = x
        # x = self.add_eps1(x)
        rnn_out, _ = self.gru(self.embed_dropout(x))
        z_inferred = self.encoder(rnn_out)
        # [batch_size x seq_len x total_items]
        dec_out, out_embed = self.decoder(z_inferred)

        return dec_out, x_real, z_inferred, out_embed


class Embed(nn.Module):
    def __init__(self, hyper_params):
        super(Embed, self).__init__()

        self.item_embed = nn.Embedding(
            hyper_params['total_items'] + 1, hyper_params['item_embed_size'])

    def forward(self, x):
        return self.item_embed(x)


class GRUEncoder(nn.Module):
    def __init__(self, hyper_params):
        super(GRUEncoder, self).__init__()
        self.hyper_params = hyper_params

        self.encoder = Encoder(hyper_params)

        self.gru = nn.GRU(
            hyper_params['item_embed_size'], hyper_params['rnn_size'],
            batch_first=True, num_layers=1
        )

        self.linear_o = nn.Linear(
            hyper_params['hidden_size'], hyper_params['latent_size'])

        self.tanh = nn.Tanh()
        self.embed_dropout = nn.Dropout(0.2)
        self.encoded_dropout = nn.Dropout(0.2)

        self.add_eps1 = AddEps(hyper_params['item_embed_size'])
        self.add_eps2 = AddEps(hyper_params['rnn_size'])

    def sample_latent(self, z_inferred):
        return torch.randn_like(z_inferred)

    def forward(self, x):
        x = self.add_eps1(x)
        rnn_out, _ = self.gru(self.embed_dropout(x))
        z_inferred = self.encoder(self.add_eps2(rnn_out))
        z_inferred = self.linear_o(z_inferred)
        z_inferred = self.encoded_dropout(z_inferred)

        return z_inferred


class Adversary(nn.Module):
    def __init__(self, hyper_params):
        super(Adversary, self).__init__()
        self.hyper_params = hyper_params
        self.linear_i = nn.Linear(
            hyper_params['item_embed_size'] + hyper_params['latent_size'], 128)

        self.dnet_list = []
        self.net_list = []
        for _ in range(2):
            self.dnet_list.append(nn.Linear(128, 128))
            self.net_list.append(nn.Linear(128, 128))

        self.dnet_list = nn.ModuleList(self.dnet_list)
        self.net_list = nn.ModuleList(self.net_list)

        self.linear_o = nn.Linear(128, hyper_params['latent_size'])
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x, z, padding):
        # batch_size x seq_len x dim
        net = torch.cat((x, z), 2)
        net = self.linear_i(net)
        net = self.dropout1(net)

        for i in range(2):
            dnet = self.dnet_list[i](net)
            net = net + self.net_list[i](dnet)
            net = F.elu(net)

        # seq_len
        net = self.linear_o(net)
        net = self.dropout2(net)
        net = net + 0.5 * torch.square(z)

        net = net * (1.0 - padding.float().unsqueeze(2))

        return net


class GRUAdversary(nn.Module):
    def __init__(self, hyper_params):
        super(GRUAdversary, self).__init__()

        self.hyper_params = hyper_params
        self.gru = nn.GRU(
            input_size=hyper_params['item_embed_size'] +
            hyper_params['latent_size'],
            hidden_size=128,
            batch_first=True
        )
        self.linear = nn.Linear(128, 1)

    def forward(self, x, z, padding):
        x = F.gelu(self.gru(torch.cat([x, z], dim=-1))[0])
        x = self.linear(x).squeeze(2)
        return (1.0 - padding.float()) * x
