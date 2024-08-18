import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pydantic import BaseModel
from sklearn.svm import SVC
import os

class TextModel(BaseModel):
    text: str

class TextClassificator:
    def __init__(self):
        # Поля класса
        self.model = MultinomialNB()

        # Пример данных

    data = {}

    def parseFileToData(self, filepath, labels):
        data = {
            'text': [],
            'label': []
        }

        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                # Удаляем лишние пробелы и знак новой строки
                line = line.strip()
                if '@' in line:
                    # Разделяем текст и тип
                    text, label = line.rsplit('@', 1)
                    text = text.strip()
                    label = label.strip()

                    # Проверяем корректность метки
                    if label in labels:
                        data['text'].append(text)
                        data['label'].append(label)

        self.data = data

    def fitModel(self, filepath, labels):
        # Создание DataFrame
        self.parseFileToData(filepath, labels)
        df = pd.DataFrame(self.data)

        # Векторизация текста
        self.vectorizer = CountVectorizer()
        X = self.vectorizer.fit_transform(df['text'])

        # Определение целевой переменной
        y = df['label']

        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Обучение модели с использованием SVM
        self.model = SVC(kernel='linear')  # Выбираем линейное ядро
        self.model.fit(X_train, y_train)

        # Предсказание
        y_pred = self.model.predict(X_test)

        # Оценка качества модели
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f'Accuracy: {accuracy:.2f}')
        print('Classification Report:')
        print(report)

    def save_model(self, model_filename='model.joblib', vectorizer_filename='vectorizer.joblib'):
        # Сохранение модели и векторизатора в файлы
        joblib.dump(self.model, model_filename)
        joblib.dump(self.vectorizer, vectorizer_filename)
        print(f'Модель сохранена в {model_filename}, векторизатор сохранен в {vectorizer_filename}')

    def load_model(self, model_filename='model.joblib', vectorizer_filename='vectorizer.joblib'):
        # Загрузка модели и векторизатора из файлов
        self.model = joblib.load(model_filename)
        self.vectorizer = joblib.load(vectorizer_filename)
        print(f'Модель загружена из {model_filename}, векторизатор загружен из {vectorizer_filename}')

    def predict(self, texts):
        # Прогнозирование меток для новых текстов
        X_new = self.vectorizer.transform(texts)
        predictions = self.model.predict(X_new)
        return predictions.tolist()


class TextBaseClassificator(TextClassificator):
    def __init__(self):
        super().__init__()
        if not os.path.exists("model_base.joblib") or not os.path.exists("vectorizer_base.joblib"):
            self.fitModel('fit_texts.txt', ['негатив', 'флуд', 'нормальный'])
            self.save_model('model_base.joblib', 'vectorizer_base.joblib')
        else:
            self.load_model('model_base.joblib', 'vectorizer_base.joblib')



class TextAdvancedClassificator(TextClassificator):
    def __init__(self):
        super().__init__()
        if not os.path.exists("model_advanced.joblib") or not os.path.exists("vectorizer_advanced.joblib"):
            self.fitModel('fit_advanced_texts.txt', ['электричество', 'канализация', 'водоснабжение', 'связь', 'личное'])
            self.save_model('model_advanced.joblib', 'vectorizer_advanced.joblib')
        else:
            self.load_model('model_advanced.joblib', 'vectorizer_advanced.joblib')