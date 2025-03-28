import torch
from model import PINN
from data_generator import generate_data
from config import CONFIG
import numpy as np

# Модифицированная функция для вычисления отдельных слагаемых
def calculate_physics_loss_components(model, x_norm, y_norm):
    x_norm = x_norm.clone().requires_grad_(True)
    y_norm = y_norm.clone().requires_grad_(True)
    
    inputs = torch.cat([x_norm, y_norm], dim=1)
    Bx = model(inputs)[:, 0]
    By = model(inputs)[:, 1]
    
    # Вычисление производных
    Bx_x, = torch.autograd.grad(Bx, x_norm, torch.ones_like(Bx), create_graph=True)
    By_y, = torch.autograd.grad(By, y_norm, torch.ones_like(By), create_graph=True)
    By_x, = torch.autograd.grad(By, x_norm, torch.ones_like(By), create_graph=True)
    Bx_y, = torch.autograd.grad(Bx, y_norm, torch.ones_like(Bx), create_graph=True)
    
    # Уравнения
    div_B = (Bx_x + By_y) / CONFIG["L"]
    J_z = CONFIG["I"] * torch.exp(-100 * (x_norm**2 + y_norm**2) * CONFIG["L"]**2)
    curl_H_z = (By_x - Bx_y) / (CONFIG["L"] * 4 * np.pi * 1e-7)
    
    # Граничные условия
    mask = ((x_norm.abs() > 0.9) | (y_norm.abs() > 0.9))
    mask = mask.squeeze()
    
    if mask.sum() == 0:
        boundary_loss = torch.tensor(0.0).to(x_norm.device)
    else:
        boundary_loss = torch.mean(Bx[mask]**2 + By[mask]**2)
    
    # Отдельные компоненты потерь
    div_loss = torch.mean(div_B**2)  # Потеря от дивергенции
    curl_loss = torch.mean((curl_H_z - J_z)**2)  # Потеря от ротора
    boundary_loss_weighted = 0.1 * boundary_loss  # Граничная потеря с весом
    
    total_loss = div_loss + curl_loss + boundary_loss_weighted
    
    return div_loss, curl_loss, boundary_loss_weighted, total_loss

# 1. Загрузка обученной модели
if __name__ == "__main__":
    model = PINN()
    model.load_state_dict(torch.load("pinn_model.pth"))
    model.to(CONFIG["device"])
    model.eval()

    # 2. Подготовка данных (тестовый набор)
    X_train, X_test, Y_train, Y_test = generate_data()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=CONFIG["device"])

    # Разделяем нормализованные координаты
    x_norm = X_test_tensor[:, 0:1]  # x/L
    y_norm = X_test_tensor[:, 1:2]  # y/L

    # 3. Вычисление компонентов потерь
    div_loss, curl_loss, boundary_loss, total_loss = calculate_physics_loss_components(model, x_norm, y_norm)

    # 4. Вывод результатов
    print(f"Divergence Loss: {div_loss.item():.4e}")
    print(f"Curl Loss: {curl_loss.item():.4e}")
    print(f"Boundary Loss (weighted): {boundary_loss.item():.4e}")
    print(f"Total Loss: {total_loss.item():.4e}")