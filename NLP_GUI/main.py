import tkinter as tk
from tkinter import scrolledtext, messagebox
from googletrans import Translator
from nltk.corpus import stopwords
from nltk import FreqDist, word_tokenize, sent_tokenize
from string import punctuation
from nltk.stem import PorterStemmer
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import spacy

def translate_text():
    text_to_translate = entry.get()
    translator = Translator()
    try:
        translation = translator.translate(text_to_translate, src=source_language.get(), dest=target_language.get())
        translated_text.set(translation.text)
    except Exception as e:
        translated_text.set("Translation error: " + str(e))

def text_summarize(text, num_lines=3):
    stop_words = stopwords.words('english') + list(punctuation) + ['``']
    words = [word for word in word_tokenize(text.lower()) if word not in stop_words]
    original_sentences = [sentence for sentence in sent_tokenize(text)]
    sentences = []
    for sentence in sent_tokenize(text.lower()):
        op_sent = [word for word in word_tokenize(sentence) if word not in stop_words]
        sent = ' '.join(op_sent)
        sentences.append(sent)
    freq_words = dict(FreqDist(words))
    sent_dict = dict()
    for sentence in original_sentences:
        sent_weight = 0
        sent_words = word_tokenize(sentence)
        for sent_word in sent_words:
            if sent_word.lower() not in stop_words:
                weight = freq_words[sent_word.lower()]
                sent_weight += weight
        sent_dict[sent_weight] = sentence
    sorted_weight_list = list(sent_dict.keys())
    sorted_weight_list = sorted(sorted_weight_list, key=int, reverse=True)
    index_dict = dict()
    for index, weight in enumerate(sorted_weight_list):
        index_dict[index] = weight
    final_output = []
    for i in range(num_lines):
        final_output.append(sent_dict[index_dict[i]])
    final_output = '\n'.join(final_output)
    return final_output

def summarize_text():
    input_text = input_text_widget.get("1.0", tk.END)
    num_lines = int(num_lines_entry.get())
    try:
        summarized_text = text_summarize(input_text, num_lines)
        output_text_widget.delete("1.0", tk.END)
        output_text_widget.insert(tk.END, summarized_text)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def preprocess(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stop_words]
    review = ' '.join(review)
    return review

def predict_sentiment(review, model):
    review = preprocess(review)
    review_vector = cv.transform([review]).toarray()
    prediction = model.predict(review_vector)
    return prediction[0]

def predict_review_sentiment():
    review = entry_review.get()
    if review:
        nb_prediction = predict_sentiment(review, clf)
        lr_prediction = predict_sentiment(review, lr_model)
        nb_sentiment = "Positive" if nb_prediction == 1 else "Negative"
        lr_sentiment = "Positive" if lr_prediction == 1 else "Negative"
        messagebox.showinfo("Sentiment Prediction", f"Naive Bayes Prediction: {nb_prediction} ({nb_sentiment})\nLogistic Regression Prediction: {lr_prediction} ({lr_sentiment})")
    else:
        messagebox.showwarning("Warning", "Please enter a review.")

def ner():
    text = text_entry.get("1.0", "end").strip()
    if not text:
        messagebox.showinfo("Info", "Please enter some text.")
        return
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    if entities:
        result_text.config(state=tk.NORMAL)
        result_text.delete("1.0", "end")
        for entity, entity_type in entities:
            result_text.insert("end", f"{entity} - {entity_type}\n")
        result_text.config(state=tk.DISABLED)
    else:
        messagebox.showinfo("Info", "No named entities found in the text.")

# Load English language model for NER
nlp = spacy.load("en_core_web_sm")

# Load the dataset for sentiment analysis
try:
    data = pd.read_csv('Restaurant_Reviews.tsv', sep='\t')
except FileNotFoundError:
    messagebox.showerror("Error", "File not found. Please check the file path.")

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

if 'Review' in data.columns:
    corpus = [preprocess(review) for review in data['Review']]
else:
    messagebox.showerror("Error", "Dataset does not contain 'Review' column.")

cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = data['Liked']

clf = MultinomialNB()
clf.fit(X, y)

lr_model = LogisticRegression()
lr_model.fit(X, y)

# Create the main window
root = tk.Tk()
root.title("Text Analysis Tool")
root.resizable(False, False)

# Source language
source_label = tk.Label(root, text="Source Language:")
source_label.grid(row=0, column=0, pady=5, padx=5, sticky="w")
source_language = tk.StringVar()
source_language.set("en")  # Default to English
source_dropdown = tk.OptionMenu(root, source_language, "en", "ar")
source_dropdown.grid(row=0, column=1, pady=5, padx=5, sticky="ew")

# Target language
target_label = tk.Label(root, text="Target Language:")
target_label.grid(row=1, column=0, pady=5, padx=5, sticky="w")
target_language = tk.StringVar()
target_language.set("ar")  # Default to Arabic
target_dropdown = tk.OptionMenu(root, target_language, "en", "ar")
target_dropdown.grid(row=1, column=1, pady=5, padx=5, sticky="ew")

# Text entry for translation
entry_label = tk.Label(root, text="Enter text to translate:")
entry_label.grid(row=2, column=0, pady=5, padx=5, sticky="w")
entry = tk.Entry(root, width=50)
entry.grid(row=2, column=1, pady=5, padx=5, sticky="ew")

# Translate button
translate_button = tk.Button(root, text="Translate", command=translate_text)
translate_button.grid(row=3, column=0, pady=5, padx=5, columnspan=2, sticky="ew")

# Translated text display
translated_text = tk.StringVar()
translated_label = tk.Label(root, textvariable=translated_text, wraplength=300)
translated_label.grid(row=4, columnspan=2, pady=5, padx=5)

# Input text for summarization
input_text_widget = scrolledtext.ScrolledText(root, width=50, height=5, wrap=tk.WORD)
input_text_widget.grid(row=5, column=0, pady=5, padx=5, columnspan=2, sticky="nsew")

# Number of lines entry for summarization
num_lines_label = tk.Label(root, text="Number of lines:")
num_lines_label.grid(row=6, column=0, pady=5, padx=5, sticky="w")
num_lines_entry = tk.Entry(root, width=50)
num_lines_entry.grid(row=6, column=1, pady=5, padx=5, sticky="ew")

# Output text for summarization
output_text_widget = scrolledtext.ScrolledText(root, width=50, height=5, wrap=tk.WORD)
output_text_widget.grid(row=7, column=0, pady=5, padx=5, columnspan=2, sticky="nsew")

# Summarize button
summarize_button = tk.Button(root, text="Summarize", command=summarize_text, width=50)
summarize_button.grid(row=8, column=0, pady=5, padx=5, columnspan=2, sticky="ew")

# Input text for sentiment analysis
entry_review = tk.Entry(root, width=50)
entry_review.grid(row=9, column=0, pady=5, padx=5, columnspan=2, sticky="ew")

# Predict sentiment button
button_predict = tk.Button(root, text="Predict Sentiment", command=predict_review_sentiment, width=50)
button_predict.grid(row=10, column=0, pady=5, padx=5, columnspan=2, sticky="ew")

# Text entry for named entity recognition
text_entry = tk.Text(root, height=5, width=50)
text_entry.grid(row=11, column=0, pady=5, padx=5, columnspan=2, sticky="nsew")

# Perform NER button
ner_button = tk.Button(root, text="Perform NER", command=ner, width=50)
ner_button.grid(row=12, column=0, pady=5, padx=5, columnspan=2, sticky="ew")

# Result display for named entity recognition
result_text = tk.Text(root, height=5, width=50, state=tk.DISABLED)
result_text.grid(row=13, column=0, pady=5, padx=5, columnspan=2, sticky="nsew")

root.mainloop()
