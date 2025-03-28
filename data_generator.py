import numpy as np
from sklearn.model_selection import train_test_split
from config import CONFIG

def generate_data():
    mu0 = 4 * np.pi * 1e-7
    theta = np.linspace(0, 2*np.pi, 200)
    r = np.linspace(CONFIG["r_min"], CONFIG["L"], 200)
    
    R, Theta = np.meshgrid(r, theta)
    x = R * np.cos(Theta)
    y = R * np.sin(Theta)
    
    r_points = np.sqrt(x**2 + y**2)
    r_points[r_points < CONFIG["r_min"]] = CONFIG["r_min"]  # Учитываем минимальный радиус
    B_phi = (mu0 * CONFIG["I"]) / (2 * np.pi * r_points)
    
    # Корректный расчёт Bx и By
    Bx = -B_phi * (y / r_points)
    By = B_phi * (x / r_points)
    
    # Нормализация данных
    X = np.stack([x.flatten()/CONFIG["L"], y.flatten()/CONFIG["L"]], axis=1)
    Y = np.stack([Bx.flatten(), By.flatten()], axis=1)
    
    return train_test_split(X, Y, test_size=0.2, random_state=42)