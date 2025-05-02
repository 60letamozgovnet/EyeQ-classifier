# EyeQ Classifier

Классификация качества изображений глазного дна на основе модели `EfficientNet-B0` с дообучением и экспортом в ONNX. Поддерживается автоматическая загрузка весов из HuggingFace. Модель основана на [EyePACS dataset](https://www.kaggle.com/c/diabetic-retinopathy-detection).
---

## Данные

Используются CSV-файлы:

- `Label_EyeQ_train.csv`
- `Label_EyeQ_test.csv`

Поля:
- `image`: имя файла
- `quality`: (0 - good, 1 - usable, 2 - reject)

> Картинки лежат в `main/data/train/`, сохраняются в `main/data/preprocessed/` после обработки.

---

## Архитектура

- Базовая модель: `google/efficientnet-b0` из HuggingFace Transformers
- Классификатор: `Dropout + Linear(1)`
- Аугментация: `RandomRotation`, `CLAHE`, `Crop`, `Resize`
- Потери: `BCEWithLogitsLoss`

---

## Метрики

- Accuracy
- F1-score (binary)
- Validation loss

---

## Бысткрый старт (Docker)

1. Установите Docker
2. Скачайте датасет по ссылке, создайте папку train в data и переместите все изображения туда
3. Создайте папку preprocessed в data 
4. Склонируйте проект:
   ```bash
   git clone https://github.com/60letamozgovnet/EyeQ-Classifier.git
   cd EyeQ-Classifier
   ```
5. Запустите run.sh
