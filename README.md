# test_api
API взаимодействие с моделью



Запустим сервис titanic-service:
```
docker build -t titanic-service:latest .

docker run -d --name titanic-service -p 500:5000 titanic-service:latest
```
---