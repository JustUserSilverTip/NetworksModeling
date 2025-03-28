import torch

CONFIG = {
    "L": 2.0,
    "I": 1.0,
    "r_min": 0.1,
    "hidden_size": 128,  # Увеличено для большей выразительности
    "num_layers": 4,     # Увеличено для большей глубины
    "epochs": 1000,     # Увеличено для лучшей сходимости
    "batch_size": 512,
    "lr": 1e-3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "alpha": 1.0,
    "beta": 10.0,  # Увеличен вес физической ошибки
}

LOGGING = {
    "level": "DEBUG",
    "file": "training.log"
}