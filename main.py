from fastapi import FastAPI
import os

from components.image_classificator import ImageClassifier
from components.text_classificator import TextBaseClassificator, TextModel, TextAdvancedClassificator

app = FastAPI()

@app.get("/")
async def home():
   return {"data": "Hello World"}


@app.post("/text-class")
async def textClass(text_model: TextModel):
   classifier = TextBaseClassificator()

   # Получаем текст из тела запроса
   text = text_model.text
   predictions = classifier.predict([text])
   return {"data": predictions}

@app.post("/text-class-adv")
async def textClassAdv(text_model: TextModel):
   classifier = TextAdvancedClassificator()

   # Получаем текст из тела запроса
   text = text_model.text
   predictions = classifier.predict([text])
   return {"data": predictions}

@app.post("/image-class")
async def textClassAdv(text_model: TextModel):
   classifier = ImageClassifier(train_dir='dataset/images/train/', test_dir='dataset/images/test/', num_epochs=400)
   #classifier.train()
   classifier.evaluate()

   # Пример предсказания
   result = classifier.predict_image('picture1.jpg')
   print(result)
   result = classifier.predict_image('picture2.jpg')
   print(result)
   result = classifier.predict_image('picture3.jpg')
   print(result)
   result = classifier.predict_image('picture4.jpg')
   print(result)
   result = classifier.predict_image('picture5.jpg')
   print(result)
   result = classifier.predict_image('picture6.jpg')
   print(result)
   return {"data": result}
