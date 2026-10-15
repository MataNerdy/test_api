'''
Давайте создадим простое API с тремя ручками: одна для предсказания выживания (/predict),
другая для получения количества сделанных запросов (/stats), и третья для проверки работы API (/health).

Шаг 1: Установка необходимых библиотек
Убедитесь, что у вас установлены необходимые библиотеки:
pip install fastapi uvicorn pydantic scikit-learn pandas

Шаг 2: Создание app_api.py
Шаг 3: Запустите ваше приложение: python app_api.py
Шаг 4: Тестирование API
Теперь вы можете протестировать ваше API с помощью curl или любого другого инструмента для отправки HTTP-запросов.

Проверка работы API (/health)
curl -X GET http://127.0.0.1:5001/health
curl -X GET http://127.0.0.1:5001/stats
curl -X POST http://127.0.0.1:5001/predict_model -H "Content-Type: application/json" -d "{\"Pclass\": 3, \"Age\": 22.0, \"Fare\": 7.2500}"
'''
import pickle
import uvicorn
from pathlib import Path
from threading import Lock

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(title="Titanic API")

# Ищем модель рядом с этим Python-файлом.
BASE_DIR = Path(__file__).resolve().parent

print("Загружаем модель...", flush=True)

with (BASE_DIR / "model.pkl").open("rb") as file:
    model = pickle.load(file)

print("Модель загружена", flush=True)

request_count = 0
counter_lock = Lock()


# Описываем и проверяем входные данные.
class PredictionInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    Pclass: int = Field(ge=1, le=3)
    Age: float = Field(ge=0, le=120)
    Fare: float = Field(ge=0)


@app.get("/health")
def health():
    return {"status": "OK"}


@app.get("/stats")
def stats():
    with counter_lock:
        count = request_count

    return {"request_count": count}


@app.post("/predict_model")
def predict_model(input_data: PredictionInput):
    global request_count

    # Имена и порядок признаков — как при обучении.
    features = pd.DataFrame(
        [[input_data.Pclass, input_data.Age, input_data.Fare]],
        columns=["Pclass", "Age", "Fare"],
    )

    label = int(model.predict(features)[0])

    with counter_lock:
        request_count += 1

    return {
        "label": label,
        "prediction": "Survived" if label == 1 else "Not Survived",
    }
