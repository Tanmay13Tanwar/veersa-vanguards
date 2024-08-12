import streamlit as st
import numpy as np
import pandas as pd
import pickle
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from collections import Counter
import operator
import nltk
from itertools import combinations
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_val_score
from statistics import mean
from nltk.corpus import wordnet 
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from itertools import combinations
from time import time
from collections import Counter
import operator
import math
import pickle
#from Treatment import diseaseDetail
from sklearn.linear_model import LogisticRegression
from symptom import extract
from text2 import transcribe_audio

# Download NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

# Load utilities for text preprocessing
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
splitter = RegexpTokenizer(r'\w+')

# Load the pre-trained model and datasets
with open('linear_regression_model.pkl', 'rb') as file:
    lr = pickle.load(file)

df_comb = pd.read_csv("dis_sym_dataset_comb.csv")
df_norm = pd.read_csv("dis_sym_dataset_norm.csv")
Xc = df_comb.iloc[:, 1:]
Yc = df_comb.iloc[:, 0:1]

X = df_norm.iloc[:, 1:]
Y = df_norm.iloc[:, 0:1]
dataset_symptoms = list(X.columns)

# Define the synonyms function
def synonyms(term):
    synonyms = []
    response = requests.get('https://www.thesaurus.com/browse/{}'.format(term))
    soup = BeautifulSoup(response.content, "html.parser")
    try:
        container = soup.find('section', {'class': 'MainContentContainer'}) 
        row = container.find('div', {'class': 'css-191l5o0-ClassicContentCard'})
        row = row.find_all('li')
        for x in row:
            synonyms.append(x.get_text())
    except:
        None
    for syn in wordnet.synsets(term):
        synonyms += syn.lemma_names()
    return set(synonyms)

# Function to transcribe audio (assuming transcribe_audio is implemented elsewhere)
# def transcribe_audio(audio_path):
#     # Your implementation of transcribe_audio
#     return "transcribed text from audio"

# Function to extract symptoms from transcript (assuming extract is implemented elsewhere)
# def extract(transcript):
#     # Your implementation of extract
#     return ["symptom1", "symptom2"]

# Streamlit UI
st.title("Disease Prediction from Symptoms")

# Upload audio file
audio_file = st.file_uploader("Upload an audio file containing symptoms", type=["mp3", "wav"])
if audio_file is not None:
    audio_path = f"temp_audio.{audio_file.name.split('.')[-1]}"
    with open(audio_path, 'wb') as f:
        f.write(audio_file.read())
    
    st.write("Transcribing audio...")
    transcript = transcribe_audio(audio_path)
    st.write(f"Transcript: {transcript}")

    user_symptoms = extract(transcript)
    st.write("Extracted Symptoms: ", user_symptoms)
    
    # Preprocess user symptoms
    processed_user_symptoms = []
    for sym in user_symptoms:
        sym = sym.strip().replace('-', ' ').replace("'", '')
        sym = ' '.join([lemmatizer.lemmatize(word) for word in splitter.tokenize(sym)])
        processed_user_symptoms.append(sym)
    
    # Expand symptoms with synonyms
    user_symptoms = []
    for user_sym in processed_user_symptoms:
        user_sym = user_sym.split()
        str_sym = set()
        for comb in range(1, len(user_sym) + 1):
            for subset in combinations(user_sym, comb):
                subset = ' '.join(subset)
                subset = synonyms(subset) 
                str_sym.update(subset)
        str_sym.add(' '.join(user_sym))
        user_symptoms.append(' '.join(str_sym).replace('_', ' '))
    
    st.write("Expanded Symptoms with Synonyms: ", user_symptoms)
    
    # Find matching symptoms in the dataset
    found_symptoms = set()
    for data_sym in dataset_symptoms:
        data_sym_split = data_sym.split()
        for user_sym in user_symptoms:
            count = 0
            for symp in data_sym_split:
                if symp in user_sym.split():
                    count += 1
            if count / len(data_sym_split) > 0.5:
                found_symptoms.add(data_sym)
    found_symptoms = list(found_symptoms)

    if found_symptoms:
        st.write("Top matching symptoms from your search!")
        for idx, symp in enumerate(found_symptoms):
            st.write(f"{idx}: {symp}")
        
        select_list = st.text_input("Select relevant symptoms by entering indices (space-separated):").split()
        
        if select_list:
            dis_list = set()
            final_symp = []
            counter_list = []
            for idx in select_list:
                symp = found_symptoms[int(idx)]
                final_symp.append(symp)
                dis_list.update(set(df_norm[df_norm[symp] == 1]['label_dis']))
            
            for dis in dis_list:
                row = df_norm.loc[df_norm['label_dis'] == dis].values.tolist()
                row[0].pop(0)
                for idx, val in enumerate(row[0]):
                    if val != 0 and dataset_symptoms[idx] not in final_symp:
                        counter_list.append(dataset_symptoms[idx])
            
            dict_symp = dict(Counter(counter_list))
            dict_symp_tup = sorted(dict_symp.items(), key=operator.itemgetter(1), reverse=True)
            
            # Recommend co-occurring symptoms
            st.write("Now recommending co-occurring symptoms...")
            found_symptoms = []
            for idx, tup in enumerate(dict_symp_tup):
                found_symptoms.append(tup[0])
                if (idx + 1) % 5 == 0 or idx + 1 == len(dict_symp_tup):
                    st.write("\nCommon co-occurring symptoms:")
                    for i, ele in enumerate(found_symptoms):
                        st.write(f"{i}: {ele}")
                    select_list = st.text_input("Do you have any of these symptoms? Enter indices (space-separated), 'no' to stop, '-1' to skip:").lower().split()
                    if select_list[0] == 'no':
                        break
                    if select_list[0] == '-1':
                        found_symptoms = []
                        continue
                    for i in select_list:
                        final_symp.append(found_symptoms[int(i)])
                    found_symptoms = []

            st.write("\nFinal list of Symptoms that will be used for prediction:")
            sample_x = [0 for _ in range(len(dataset_symptoms))]
            for val in final_symp:
                st.write(val)
                sample_x[dataset_symptoms.index(val)] = 1

            # Predict disease
            lr = LogisticRegression()
            lr = lr.fit(X, Y)
            prediction = lr.predict_proba([sample_x])

            # Show top k diseases
            k = 10
            diseases = list(set(Y['label_dis']))
            diseases.sort()
            topk = prediction[0].argsort()[-k:][::-1]
            scores = cross_val_score(lr, Xc, Yc, cv=5)

            topk_dict = {}
            for idx, t in enumerate(topk):
                match_sym = set()
                row = df_norm.loc[df_norm['label_dis'] == diseases[t]].values.tolist()
                row[0].pop(0)
                for i, val in enumerate(row[0]):
                    if val != 0:
                        match_sym.add(dataset_symptoms[i])
                prob = (len(match_sym.intersection(set(final_symp))) + 1) / (len(set(final_symp)) + 1)
                prob *= np.mean(scores)
                topk_dict[t] = prob

            st.write(f"\nTop {k} diseases predicted based on symptoms")
            topk_sorted = dict(sorted(topk_dict.items(), key=lambda kv: kv[1], reverse=True))
            for j, key in enumerate(topk_sorted):
                prob = topk_sorted[key] * 100
                st.write(f"{j} Disease name: {diseases[key]} \tProbability: {round(prob, 2)}%")