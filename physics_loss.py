import torch
import numpy as np
from config import CONFIG

def calculate_physics_loss(model, x_norm, y_norm):
    x_norm = x_norm.clone().requires_grad_(True)
    y_norm = y_norm.clone().requires_grad_(True)
    
    inputs = torch.cat([x_norm, y_norm], dim=1)
    B_pred = model(inputs)
    Bx = B_pred[:, 0]
    By = B_pred[:, 1]
    
    # Вычисление градиентов
    grad_Bx = torch.autograd.grad(Bx.sum(), inputs, create_graph=True)[0]
    grad_By = torch.autograd.grad(By.sum(), inputs, create_graph=True)[0]
    Bx_x, Bx_y = grad_Bx[:, 0], grad_Bx[:, 1]
    By_x, By_y = grad_By[:, 0], grad_By[:, 1]
    
    # Дивергенция B (должна быть 0)
    div_B = Bx_x + By_y
    
    # Ротор H (H = B / mu0)
    mu0 = 4 * np.pi * 1e-7
    Hx = Bx / mu0
    Hy = By / mu0
    grad_Hx = torch.autograd.grad(Hx.sum(), inputs, create_graph=True)[0]
    grad_Hy = torch.autograd.grad(Hy.sum(), inputs, create_graph=True)[0]
    curl_H_z = grad_Hy[:, 0] - grad_Hx[:, 1]
    
    # Плотность тока J_z (ещё более узкая гауссиана)
    J_z = CONFIG["I"] * torch.exp(-100 * (x_norm**2 + y_norm**2))  # Увеличен коэффициент до 1000
    
    # Физическая ошибка
    physics_loss = torch.mean(div_B**2) + torch.mean((curl_H_z - J_z)**2)
    
    return physics_loss