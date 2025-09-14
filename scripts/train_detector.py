from ultralytics import YOLO
import os

# Загружаем базовую модель YOLOv8n (nano), предобученную на COCO
# Она легкая и быстрая, идеально для прототипа
model = YOLO('yolov8n.pt')

# Путь к нашему файлу data.yaml
# Убедитесь, что путь правильный!
data_yaml_path = os.path.join('..', 'data', 'damage_data', 'data.yaml')

def main():
    print("Начинаем обучение детектора повреждений...")
    # Запускаем обучение (fine-tuning)
    results = model.train(
        data=data_yaml_path,
        epochs=25,  # Для хакатона 25 эпох будет достаточно
        imgsz=640,  # Стандартный размер для YOLO
        batch=8,    # Уменьшите, если не хватает видеопамяти (например, до 4)
        name='yolov8n_damage_detection' # Имя папки для сохранения результатов
    )
    print("Обучение завершено!")
    print(f"Результаты сохранены в папке: {results.save_dir}")

if __name__ == '__main__':
    main()