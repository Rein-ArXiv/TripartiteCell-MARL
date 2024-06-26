import torch.nn as nn

class DDQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=9):
        super(DDQN, self).__init__()

        self.input_layers = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU()
        )
        self.hidden_layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.output_layers = nn.Sequential(
            nn.Linear(128, n_actions)
        )
        
    def forward(self, input):
        x = self.input_layers(input)
        x = self.hidden_layers(x)
        output = self.output_layers(x)
        return output
