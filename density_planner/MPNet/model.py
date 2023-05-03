import torch
import torch.nn as nn


# Model-Path Generator
class MLP(nn.Module):
    def __init__(self, input_size, output_size, up_size, lstm_hidden_size, lstm_layer_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.up_size = up_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layer_size = lstm_layer_size
        self.lstm = nn.LSTM(input_size=up_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_layer_size, batch_first=True)  # lstm
        self.fc = nn.Sequential(
        nn.Linear(input_size, 1280), nn.PReLU(),
        nn.BatchNorm1d(1280),
        nn.Linear(1280, 1024), nn.PReLU(), nn.Dropout(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 896), nn.PReLU(), nn.Dropout(),
        nn.BatchNorm1d(896),
        nn.Linear(896, 768), nn.PReLU(), nn.Dropout(),
        nn.BatchNorm1d(768),
        nn.Linear(768, 512), nn.PReLU(), nn.Dropout(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 384), nn.PReLU(), nn.Dropout(),
        nn.BatchNorm1d(384),
        nn.Linear(384, 256), nn.PReLU(), nn.Dropout(),
        nn.BatchNorm1d(256),
        nn.Linear(256, 256), nn.PReLU(), nn.Dropout(),
        nn.BatchNorm1d(256),
        nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
        nn.BatchNorm1d(128),
        nn.Linear(128, 64), nn.PReLU(), nn.Dropout(),
        nn.BatchNorm1d(64),
        nn.Linear(64, 32), nn.PReLU(),
        nn.BatchNorm1d(32),
        nn.Linear(32, output_size))

    def forward(self, input, up, device):
        up, output = self.lstm(up)
        output[0].permute(1,0,2)
        up_hidden = output[0][-1]
        new_input = torch.zeros((input.shape[0], self.input_size)).to(device)
        new_input[:,:input.shape[1]] = input
        new_input[:,-self.lstm_hidden_size:] = up_hidden
        output = self.fc(new_input)
        return output