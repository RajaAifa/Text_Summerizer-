import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk import word_tokenize
from rouge import Rouge
from nltk.translate import meteor_score
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.corpus import wordnet
import random
from wordcloud import WordCloud
from txtai.pipeline import Summary, Textractor
from PyPDF2 import PdfReader
import sqlite3
import bcrypt
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

st.set_page_config(layout="wide")
# Initialize SQLite database
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create users table if it doesn't exist
c.execute('''
          
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
''')
conn.commit()

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)

def signup(username, password):
    # Check if user already exists
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    if c.fetchone():
        return False
    else:
        # Hash the password and store the user
        hashed_password = hash_password(password)
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
        conn.commit()
        return True

def login(username, password):
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    if user and verify_password(password, user[1]):
        return True
    else:
        return False


class NeuralExtractiveSummarizer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    def summarize(self, text, percentage):
        sentences = sent_tokenize(text)
        num_sentences = len(sentences)
        num_sentences_to_select = max(1, int(num_sentences * percentage / 100))  # Ensure at least 1 sentence
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        logits = outputs.logits
        sentence_scores = torch.sigmoid(logits)

        # Select top sentences based on sentence scores and percentage
        top_indices = sentence_scores.squeeze().argsort(descending=True)[:num_sentences_to_select]
        summary = [sentences[i] for i in top_indices]
        return ' '.join(summary)

class AbstractiveCompressor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def compress(self, text):
        sentences = sent_tokenize(text)
        compressed_sentences = []

        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in self.stop_words]
            compressed_words = self.replace_with_synonyms(filtered_words)
            compressed_sentence = ' '.join(compressed_words)
            compressed_sentences.append(compressed_sentence)

        return ' '.join(compressed_sentences)

    def replace_with_synonyms(self, words):
        compressed_words = []

        for word in words:
            synonyms = self.get_synonyms(word)
            if synonyms:
                synonym = random.choice(synonyms)
                compressed_words.append(synonym)
            else:
                compressed_words.append(word)

        return compressed_words

    def get_synonyms(self, word):
        synonyms = []

        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                synonym = lemma.name()
                if '_' not in synonym:
                    synonyms.append(synonym)

        synonyms = list(set(synonyms))
        return synonyms

class InteractiveSummarizer:
    def __init__(self):
        pass

    def summarize_with_context(self, text, query):
        # Tokenize the query
        query_tokens = query.lower().split()

        # Tokenize the input text into sentences
        sentences = nltk.sent_tokenize(text)

        # Initialize a list to store sentences relevant to the query
        relevant_sentences = []

        # Iterate through each sentence in the input text
        for sentence in sentences:
            # Tokenize the sentence into words
            sentence_tokens = nltk.word_tokenize(sentence.lower())

            # Check if any query token is present in the sentence
            if any(token in sentence_tokens for token in query_tokens):
                # If the sentence contains a query token, add it to the list of relevant sentences
                relevant_sentences.append(sentence)

        # Combine the relevant sentences into a summary
        summary = ' '.join(relevant_sentences)

        return summary

class SummarizationSystem:
    def __init__(self):
        self.extractive_summarizer = NeuralExtractiveSummarizer()
        self.abstractive_compressor = AbstractiveCompressor()
        self.interactive_summarizer = InteractiveSummarizer()
        self.rouge = Rouge()
    
    def evaluate(self, hypothesis, reference):
        hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
        reference_tokens = nltk.word_tokenize(reference.lower())

        scores = self.rouge.get_scores(hypothesis, reference)
        # meteor_score_val = meteor_score.meteor_score(reference, hypothesis)
        meteor_score_val = meteor_score.meteor_score([reference_tokens], hypothesis_tokens)
        return scores, meteor_score_val
    
    def visualize_attention(self, text, summary):
        # Mock function to visualize attention
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(summary)

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 5)) 


        # Display wordcloud on axis
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    def run_summarization(self, text, reference_summary, percentage):
        summary = self.extractive_summarizer.summarize(text, percentage)
        compressed_summary = self.abstractive_compressor.compress(summary)
        interactive_summary = self.interactive_summarizer.summarize_with_context(compressed_summary, "Your query here...")

        st.write("Extractive summary:", summary)
        st.write("Compressed summary:", compressed_summary)
        st.write("Interactive summary:", interactive_summary)

        rouge_scores, meteor_score_val = self.evaluate(summary, reference_summary)
        st.write("ROUGE Scores:", rouge_scores)
        st.write("METEOR Score:", meteor_score_val)

        self.visualize_attention(text, summary)



@st.cache_resource
def text_summary(text, maxlength=None):
    #create summary instance
    summary = Summary()
    text = (text)
    result = summary(text)
    return result

def extract_text_from_pdf(file_path):
    # Open the PDF file using PyPDF2
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        page = reader.pages[0]
        text = page.extract_text()
    return text



# Create a Streamlit app
def main():
    st.title("Text Summarization System")
    auth_status = st.session_state.get('auth_status', None)
    if auth_status == "logged_in":  
        st.success(f"Welcome {st.session_state.username}!")
        st.write("This app summarizes text using various techniques.")
        choice = st.sidebar.selectbox("Select your choice", ["Summarize Text using Bert","Summarize Text using Txtai", "Summarize Document"])

        if choice == "Summarize Text using Bert":
            # Input text, reference summary, and percentage
            text = st.text_area("Input Text", "Your input text here...")
            reference_summary = st.text_input("Reference Summary", "Reference summary here...")
            percentage = st.slider("Percentage of Text to Include in Summary", 1, 100, 30)

            # Run summarization
            if st.button("Summarize"):
                summarization_system = SummarizationSystem()
                summarization_system.run_summarization(text, reference_summary, percentage)

        elif choice == "Summarize Text using Txtai":
            st.subheader("Summarize Text using txtai")
            input_text = st.text_area("Enter your text here")
            if input_text is not None:
                if st.button("Summarize Text"):
                    col1, col2 = st.columns([1,1])
                    with col1:
                        st.markdown("**Your Input Text**")
                        st.info(input_text)
                    with col2:
                        st.markdown("**Summary Result**")
                        result = text_summary(input_text)
                        st.success(result)

        elif choice == "Summarize Document":
            st.subheader("Summarize Document using txtai")
            input_file = st.file_uploader("Upload your document here", type=['pdf'])
            if input_file is not None:
                if st.button("Summarize Document"):
                    with open("doc_file.pdf", "wb") as f:
                        f.write(input_file.getbuffer())
                    col1, col2 = st.columns([1,1])
                    with col1:
                        st.info("File uploaded successfully")
                        extracted_text = extract_text_from_pdf("doc_file.pdf")
                        st.markdown("**Extracted Text is Below:**")
                        st.info(extracted_text)
                    with col2:
                        st.markdown("**Summary Result**")
                        text = extract_text_from_pdf("doc_file.pdf")
                        doc_summary = text_summary(text)
                        st.success(doc_summary)

    elif auth_status == "login_failed":
        st.error("Login failed. Please check your username and password.")
        auth_status = None
    elif auth_status == "signup_failed":
        st.error("Signup failed. Username already exists.")
        auth_status = None
    # Login/Signup form
    if auth_status is None or auth_status == "logged_out":
        form_type = st.radio("Choose form type:", ["Login", "Signup"])

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if form_type == "Login":
            if st.button("Login"):
                if login(username, password):
                    st.session_state.auth_status = "logged_in"
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.session_state.auth_status = "login_failed"
                    st.rerun()
        else:  # Signup
            if st.button("Signup"):
                if signup(username, password):
                    st.session_state.auth_status = "logged_in"
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.session_state.auth_status = "signup_failed"
                    st.rerun()

    # Logout button
    if auth_status == "logged_in":
        if st.button("Logout"):
            st.cache_resource.clear()
            st.session_state.auth_status = "logged_out"
            del st.session_state.username
            st.rerun()



if __name__ == "__main__":
    main()
