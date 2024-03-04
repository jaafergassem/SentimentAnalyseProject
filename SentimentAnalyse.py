import PySimpleGUI as sg
import csv
import re
import nltk
from nltk.corpus import stopwords
from googletrans import Translator
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from transformers import pipeline

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stops = set(stopwords.words('english'))

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def listToString(s):
    str1 = ""
    for ele in s:
        str1 += ele + ' '
    return str1

def normalizeWord(string):
    lower_string = string.lower()
    no_number_string = re.sub(r'\d+', '', lower_string)
    no_punc_string = re.sub(r'[^\w\s]', '', no_number_string)
    no_wspace_string = no_punc_string.strip()
    
    lst_string = no_wspace_string.split()
    filtered_words = [word for word in lst_string if word not in stops]

    return listToString(filtered_words)

translator = Translator()

sg.theme("DarkBlue4")
sg.set_options(font=("Courier New", 13))
layout = [
    [sg.Text("Saisissez le texte à analyser")],
    [sg.Multiline(size=(50, 10), enable_events=True, key="-FOLDER-")],
    [sg.Text("", key="state")],
    [sg.Button("Analyser")],
]

window = sg.Window("Projet Big Data", layout, resizable=True)

while True:
    event, values = window.read()

    if event == "Analyser":
        translatedWord = translator.translate(values["-FOLDER-"], dest='en')
        y = normalizeWord(translatedWord.text)
        sentiment_pipeline = pipeline("sentiment-analysis")
        data = [y]
        x = sentiment_pipeline(data)
        header = ['originalText', 'translatedText', 'sentiment', 'score']
        data = [values["-FOLDER-"], translatedWord.text, x[0]['label'], x[0]['score']]

        with open('NewDataset.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(data)

        window['state'].update(x[0]['label'] + " | " + str(x[0]['score']))
        window['-FOLDER-'].update(translatedWord.text)

    if event == "Fermer" or event == sg.WIN_CLOSED:
        break

window.close()
