from django.shortcuts import render

import whisper
import os

import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
from spacy.util import  filter_spans
from spacy.scorer import Scorer
import json
import numpy as np
from spacy.training.example import Example
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


def home(request):
    if request.method == 'POST':
        model_best = spacy.load('C:/Users/dima_/PycharmProjects/django_ai/demo_ai/mapp/model-best')

        uploaded = request.FILES['up_file']
        readed = uploaded.read()
        name = uploaded.name

        f = open(name, 'wb')
        f.write(readed)
        f.close()

        model = whisper.load_model('medium')
        result = model.transcribe(name)
        os.remove(uploaded.name)

        def predict(model, data):
            doc = model(data)

            for ent in doc.ents:
                if (ent.label_ == "1"):
                    print(ent.text)

        gl = predict(model_best, result['text'])

        return render(request, 'index.html', { 'text': result['text'], 'gl': gl })

    return render(request, 'index.html')