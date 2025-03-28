import torch.nn as nn
from config import CONFIG

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        input_size = 2  # x, y
        
        # Скрытые слои
        for _ in range(CONFIG["num_layers"]):
            layers += [
                nn.Linear(input_size, CONFIG["hidden_size"]),
                nn.Tanh()
            ]
            input_size = CONFIG["hidden_size"]
            
        # Выходной слой
        layers.append(nn.Linear(input_size, 2))  # Bx, By
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)