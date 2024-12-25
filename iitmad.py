import cv2
import easyocr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import spacy

nlp = spacy.load("en_core_web_sm")

reader = easyocr.Reader(['en'])

def perform_ocr(image_path):
    image = cv2.imread(image_path)
    results = reader.readtext(image)
    text = ' '.join([result[1] for result in results])
    return text

def classify_documents(texts, labels):
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f'Model Accuracy: {accuracy:.2f}')
    return model

def extract_information(text):
    doc = nlp(text)
    extracted_info = {}
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'DATE', 'MONEY']:
            extracted_info[ent.label_] = ent.text     
    return extracted_info

def process_documents(document_paths, labels):
    all_texts = []
    for path in document_paths:
        text = perform_ocr(path)
        all_texts.append(text)
    classifier = classify_documents(all_texts, labels)
    for text in all_texts:
        info = extract_information(text)
        print(f'Extracted Information: {info}')