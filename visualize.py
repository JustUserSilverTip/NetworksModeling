import torch
import numpy as np
import matplotlib.pyplot as plt
from model import PINN
from data_generator import generate_data
from config import CONFIG
import re

def parse_log(file_path):
    epochs = []
    losses = []
    
    # Регулярное выражение для извлечения эпохи и Loss
    pattern = r"Epoch (\d+) \| Loss: ([\d\.e+-]+) \|"
    
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                epochs.append(epoch)
                losses.append(loss)
    
    return epochs, losses

if __name__ == "__main__":


    # Путь к файлу лога
    log_file = "training.log"

    # Извлекаем данные
    epochs, losses = parse_log(log_file)

    # Строим график
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, label='Total Loss', color='blue')
    plt.yscale('log')  # Логарифмическая шкала для лучшей читаемости
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss vs Epoch')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()