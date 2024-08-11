import os
import warnings
import numpy as np
import pandas as pd
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import cross_val_score
from itertools import combinations
from collections import Counter
import operator
import requests
from bs4 import BeautifulSoup
import nltk
from datetime import datetime
import httpx
from deepgram import DeepgramClient, DeepgramClientOptions, PrerecordedOptions, FileSource
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pickle

# Suppress warnings
warnings.simplefilter("ignore")

# Download NLTK resources
nltk.download('all', quiet=True)

# Deepgram API key
DEEPGRAM_API_KEY = "6d460ee96cb096dc201627657de69dfcb34373eb"

# Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAZSP3AXxW8vmcalgAYmikudzSqT_IgWQk"

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Utility functions
def synonyms(term):
    synonyms = []
    response = requests.get(f'https://www.thesaurus.com/browse/{term}')
    soup = BeautifulSoup(response.content, "html.parser")
    try:
        container = soup.find('section', {'class': 'MainContentContainer'})
        row = container.find('div', {'class': 'css-191l5o0-ClassicContentCard'})
        row = row.find_all('li')
        for x in row:
            synonyms.append(x.get_text())
    except:
        pass
    for syn in wordnet.synsets(term):
        synonyms += syn.lemma_names()
    return set(synonyms)

# Pre-processing utilities
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
splitter = RegexpTokenizer(r'\w+')

# Load datasets
df_comb = pd.read_csv("dis_sym_dataset_comb.csv")
df_norm = pd.read_csv("dis_sym_dataset_norm.csv")

X = df_norm.iloc[:, 1:]
Y = df_norm.iloc[:, 0:1]
dataset_symptoms = list(X.columns)

with open('linear_regression_model.pkl', 'rb') as file:
     lr = pickle.load(file)


Xc = df_comb.iloc[:, 1:]
Yc = df_comb.iloc[:, 0:1]

scores = cross_val_score(lr, Xc, Yc, cv=5)


def transcribe_audio(file_path):
    try:
        config = DeepgramClientOptions(verbose=False)
        deepgram = DeepgramClient(DEEPGRAM_API_KEY, config)

        with open(file_path, "rb") as file:
            buffer_data = file.read()

        payload = {"buffer": buffer_data}
        options = PrerecordedOptions(model="nova-2", smart_format=True)

        response = deepgram.listen.rest.v("1").transcribe_file(
            payload, options, timeout=httpx.Timeout(300.0, connect=10.0)
        )

        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        return transcript

    except Exception as e:
        print(f"Transcription error: {e}")
        return None

# def extract_symptoms(text):
#     title_template = PromptTemplate(
#         input_variables=['conversation'],
#         template="""As a disease symptom analyzer, analyze the following conversation between a doctor and patient. List the symptoms that the patient has and return them in a Python list format:
#         {conversation}"""
#     )

#     title_chain = LLMChain(llm=llm, prompt=title_template, verbose=False, output_key='output')
#     response = title_chain.invoke({'conversation': text})
#     return eval(response["output"])  # Convert string representation of list to actual list
import re

def parse_code(code):
    pattern = re.compile(r"```(?:python)?(.*?)```", re.DOTALL)
    match = pattern.search(code)
    if match:
        return match.group(1).strip()
    else:
        return code.strip()

def extract_symptoms(text):
    title_template = PromptTemplate(
        input_variables=['conversation'],
        template="""As a disease symptom analyzer, analyze the following conversation between a doctor and patient. List the symptoms that the patient has and return them in a Python list format:
        {conversation}"""
    )

    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=False, output_key='output')
    response = title_chain.invoke({'conversation': text})
    
    # Extract the code from the response
    output = parse_code(response["output"])
    
    try:
        # Try to evaluate the output as a Python expression
        symptoms = eval(output)
        if isinstance(symptoms, list):
            return symptoms
    except:
        pass
    
    # If eval fails, try to extract symptoms line by line
    lines = output.split('\n')
    symptoms = []
    for line in lines:
        # Remove leading dashes, asterisks, or numbers (common in lists)
        line = re.sub(r'^[-*\d.]\s*', '', line.strip())
        if line:
            symptoms.append(line)
    
    if symptoms:
        return symptoms
    else:
        print("No symptom list found in the response. Raw output:", output)
        return []
def predict_disease(symptoms):
    sample_x = [0 for _ in range(len(dataset_symptoms))]
    for symptom in symptoms:
        if symptom in dataset_symptoms:
            sample_x[dataset_symptoms.index(symptom)] = 1

    prediction = lr.predict_proba([sample_x])
    diseases = list(set(Y['label_dis']))
    diseases.sort()
    topk = prediction[0].argsort()[-10:][::-1]

    results = []
    for t in topk:
        disease = diseases[t]
        match_sym = set()
        row = df_norm.loc[df_norm['label_dis'] == disease].values.tolist()[0][1:]
        for idx, val in enumerate(row):
            if val != 0:
                match_sym.add(dataset_symptoms[idx])
        #prob = (len(match_sym.intersection(set(symptoms))) + 1) / (len(set(symptoms)) + 1)
        #prob *= np.mean(scores)
        results.append((disease))

    return results

# def get_disease_details(disease):
#     # This function should be implemented to fetch disease details
#     # For now, we'll return a placeholder message
#     return f"Details for {disease}: Please consult a healthcare professional for accurate information and treatment options."

def main():
    audio_file = "RES0206.mp3"
    
    print("Transcribing audio...")
    transcript = transcribe_audio(audio_file)
    if not transcript:
        print("Failed to transcribe audio. Exiting.")
        return

    print("\nExtracting symptoms...")
    symptoms = extract_symptoms(transcript)
    print("Extracted symptoms:", symptoms)

    print("\nPredicting diseases...")
    predictions = predict_disease(symptoms)

    print("\nTop 10 predicted diseases:")
    for i, (disease, probability) in enumerate(predictions[:10], 1):
        print(f"{i}. {disease}: {probability:.2f}%")

    # while True:
    #     choice = input("\nEnter the number of the disease for more details, or 'q' to quit: ")
    #     if choice.lower() == 'q':
    #         break
    #     try:
    #         index = int(choice) - 1
    #         if 0 <= index < len(predictions):
    #             disease = predictions[index][0]
    #             details = get_disease_details(disease)
    #             print(f"\nDetails for {disease}:")
    #             print(details)
    #         else:
    #             print("Invalid number. Please try again.")
    #     except ValueError:
    #         print("Invalid input. Please enter a number or 'q'.")

if __name__ == "__main__":
    main()