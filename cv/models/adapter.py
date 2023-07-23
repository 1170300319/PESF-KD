from torch import nn
import torch


class Adapter(nn.Module):
    def __init__(self, input_size, hidd_size, output_size):
        super(Adapter, self).__init__()

        self.adapter = nn.Sequential(nn.Linear(output_size, hidd_size),
                                     nn.ReLU(),
                                     nn.Linear(hidd_size, output_size))
        self.l1 = nn.Sequential(nn.Linear(input_size, output_size))

    def forward(self, input_ids):
        x = self.l1(input_ids)
        return self.adapter(x)


