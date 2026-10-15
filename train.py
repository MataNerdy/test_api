import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import matplotlib.pyplot as plt
import numpy as np
import itertools
import requests

data = pd.read_csv('titanic.csv')
print("read_csv('titanic.csv')")
data = data [['Survived','Pclass','Age','Fare']]
data = data.dropna(subset=['Age'])

data.drop('Survived', axis=1)

train, test = train_test_split(data, test_size=0.2)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Функция для построения матрицы ошибок.
    cm - матрица ошибок
    classes - список классов
    normalize - если True, то значения матрицы ошибок нормализуются к 1
    title - заголовок графика
    cmap - цветовая схема для отображения графика
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

def experiment(**params):
    print("Создание и обучение модели решающего леса")
    model = RandomForestClassifier(**params)
    model.fit(train.drop('Survived', axis=1), train['Survived'])

    print("Вычисление метрик")
    preds = model.predict(test.drop('Survived', axis=1))
    acc = accuracy_score(test['Survived'], preds)
    cm = confusion_matrix(test['Survived'], preds)

    print("accuracy", acc)

    print("Визуализация матрицы ошибок")
    plot_confusion_matrix(cm, classes=['Not Survived', 'Survived'])

    print("Вывод classification report")
    report = classification_report(test['Survived'], preds, target_names=['Not Survived', 'Survived'])
    print(report)

    print("Сохранение модели в формате pickle")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model


print("Определение гиперпараметров модели")
params = {'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 125}

model = experiment(**params)

new_data = pd.DataFrame({
    'Pclass': [3],
    'Age': [5.0],
    'Fare': [7.2500]
})

# Предсказание
predictions = model.predict(new_data)

# Вывод результатов
print("Predicted Survival:", predictions)
