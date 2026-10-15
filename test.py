import requests

def predict_model(data):
    url = 'http://127.0.0.1:300/predict_model'

    # Отправка POST-запроса с данными в формате форм-данных
    response = requests.post(url, json=data)

    # Проверка статуса ответа
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Request failed with status code {response.status_code}"}

# Пример данных для предсказания
data = {
    "Pclass": 3,
    "Age": 22.0,
    "Fare": 10.250
}

# Получение предсказания
prediction = predict_model(data)
print(prediction)