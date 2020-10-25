import torch
import torch.nn as nn

class NumberPretrainerConfig(object):
    def __init__(
        self,
        input_size=4,
        hidden_size=768,
        output_size=1,
        **kwargs
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        

class NumberPretrainer(nn.Module):
    def __init__(self, config):
        super(NumberPretrainer, self).__init__()
        self.linear1 = nn.Linear(config.input_size, config.hidden_size)
        self.linear2 = nn.ReLU(nn.Linear(config.hidden_size, config.hidden_size * 2))
        self.linear3 = nn.ReLU(nn.Linear(config.hidden_size * 2, config.hidden_size))
        self.out = nn.Linear(config.hidden_size, config.output_size)
    
    def forward(self, input_val, labels):
        out = self.linear3(self.linear2(self.linear1(input_val)))
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(out, labels)
        return loss
        
