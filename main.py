import torch
from model import PINN
from data_generator import generate_data
from train import train_model
from config import CONFIG
from train import logger

if __name__ == "__main__":
    print(f"Используемое устройство: {CONFIG['device']}\n")
    
    # Инициализация модели
    model = PINN().to(CONFIG["device"])
    
    # Генерация данных
    X_train, X_test, Y_train, Y_test = generate_data()
    
    # Обучение
    try:
        trained_model = train_model(model, X_train, Y_train)
    except Exception as e:
        logger.error(f"Ошибка обучения: {str(e)}", exc_info=True)
        raise
    
    # Сохранение модели
    torch.save(trained_model.state_dict(), "pinn_model.pth")
    print("\nМодель успешно сохранена")