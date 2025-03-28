import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from config import CONFIG, LOGGING
from physics_loss import calculate_physics_loss
import logging
import sys

logger = logging.getLogger("pinn_trainer")
logger.setLevel(LOGGING["level"])
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler(LOGGING["file"])
stream_handler = logging.StreamHandler(sys.stdout)
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def train_model(model, X_train, Y_train):
    device = torch.device(CONFIG["device"])
    model = model.to(device)
    logger.info(f"Модель перемещена на устройство: {device}")
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        pin_memory=True
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    mse_loss = torch.nn.MSELoss()
    
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_physics_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(x_batch)
            data_loss = mse_loss(outputs, y_batch)
            physics_loss = calculate_physics_loss(model, x_batch[:, 0:1], x_batch[:, 1:2])
            loss = CONFIG["alpha"] * data_loss + CONFIG["beta"] * physics_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_data_loss += data_loss.item()
            total_physics_loss += physics_loss.item()
        
        avg_loss = total_loss / len(train_loader)
        avg_data_loss = total_data_loss / len(train_loader)
        avg_physics_loss = total_physics_loss / len(train_loader)
        logger.info(f"Epoch {epoch} | Loss: {avg_loss:.4e} | Data Loss: {avg_data_loss:.4e} | Physics Loss: {avg_physics_loss:.4e}")
        
    return model