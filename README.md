# 📝 Text Summarizer

> An interactive NLP application that combines **extractive, abstractive, and query-driven text summarization** with automatic evaluation and visualization.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ML-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co/)
[![NLTK](https://img.shields.io/badge/NLP-NLTK-green)](https://www.nltk.org/)

---

## 🚀 Overview

**Text Summarizer** is an interactive Natural Language Processing (NLP) application built with **Python and Streamlit**.

The application provides **three complementary summarization techniques**:

- 🤖 **Extractive Summarization** using BERT
- ✨ **Abstractive Compression** using WordNet
- 🔎 **Query-Driven Summarization** based on a user-defined topic

It also provides automatic summary evaluation using **ROUGE and METEOR scores**, along with a **word cloud visualization** of the generated summary.

---

## ✨ Features

### 🤖 BERT-Based Extractive Summarization

Uses the pretrained `bert-base-uncased` model to analyze and score sentences.

The application:

1. Splits the input text into sentences.
2. Generates BERT representations.
3. Scores sentences according to their relevance.
4. Selects the highest-ranked sentences.
5. Generates a summary according to the user's selected percentage.

---

### ✨ WordNet-Based Abstractive Compression

The compression approach creates a shorter version of the text by:

- Removing stopwords
- Processing important words
- Replacing selected words with WordNet synonyms
- Producing a more compact representation of the original text

---

### 🔎 Interactive / Query-Driven Summarization

Users can provide a **topic or query**, and the system identifies sentences that are most relevant to that query.

This allows users to obtain a summary focused on a specific subject rather than the entire document.

---

### 📊 Summary Evaluation

The application evaluates generated summaries against a reference summary using:

- **ROUGE**
- **METEOR**

These metrics provide quantitative measurements of summary quality.

---

### ☁️ Word Cloud Visualization

A word cloud is generated from the resulting summary to provide a quick visual representation of the most prominent words.

---

## 🧠 Summarization Pipeline

```text
                    ┌────────────────────┐
                    │     Input Text     │
                    └─────────┬──────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
      ┌─────────────┐ ┌──────────────┐ ┌───────────────┐
      │  Extractive │ │  Abstractive  │ │ Query-Driven  │
      │  BERT Model │ │  Compression  │ │ Summarization │
      └──────┬──────┘ └───────┬──────┘ └───────┬───────┘
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                    ┌──────────────────┐
                    │ Generated        │
                    │ Summaries        │
                    └────────┬─────────┘
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
        ┌─────────────────┐     ┌────────────────┐
        │ ROUGE / METEOR  │     │  Word Cloud    │
        │   Evaluation    │     │ Visualization  │
        └─────────────────┘     └────────────────┘
```

---

## 🛠️ Technologies

| Category | Technologies |
|---|---|
| 🐍 Language | Python |
| 🌐 Web Framework | Streamlit |
| 🤖 NLP / Machine Learning | Hugging Face Transformers, BERT, PyTorch |
| 🧠 NLP Processing | NLTK, WordNet |
| 📊 Evaluation | ROUGE, METEOR |
| 📈 Visualization | Matplotlib, WordCloud |
| 🔢 Numerical Computing | NumPy |

---

## 📋 Prerequisites

Before running the project, make sure you have:

- **Python 3.8 or higher**
- **pip**
- Internet connection for downloading the pretrained BERT model

---

## ⚡ Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/TextSummarizer.git
```

### 2️⃣ Navigate to the Project

```bash
cd TextSummarizer
```

### 3️⃣ Create a Virtual Environment

It is recommended to use a virtual environment.

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` does not exist, install:

```bash
pip install streamlit torch transformers nltk rouge matplotlib numpy wordcloud
```

---

## 📚 NLTK Data

The required NLTK resources are downloaded automatically when the application starts.

You can also download them manually:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

---

## ▶️ Run the Application

Start Streamlit with:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 🎯 How to Use

### Step 1 — Enter Your Text

Paste or type the document you want to summarize into the **Input Text** field.

### Step 2 — Add a Reference Summary

Provide a reference summary if you want to calculate **ROUGE and METEOR scores**.

### Step 3 — Choose the Compression Percentage

Use the percentage slider to control how much of the original content is retained by the extractive summarizer.

### Step 4 — Generate the Summaries

Click **Summarize**.

The application generates:

- 🤖 Extractive summary
- ✨ Compressed summary
- 🔎 Query-relevant summary
- 📊 ROUGE scores
- 📈 METEOR score
- ☁️ Word cloud visualization

---

## 📊 Evaluation

The generated summaries can be compared against a reference summary using two commonly used NLP evaluation metrics.

### ROUGE

Measures the overlap between the generated summary and the reference summary.

### METEOR

Evaluates similarity between the generated and reference summaries while considering linguistic relationships such as stemming and synonyms.

---

## 🔬 Summarization Techniques

| Technique | Approach | Purpose |
|---|---|---|
| 🤖 Extractive | BERT sentence scoring | Select important sentences |
| ✨ Abstractive Compression | WordNet + stopword removal | Compress textual content |
| 🔎 Interactive | Query-based sentence selection | Focus on a specific topic |

---

## 💡 Use Cases

The application can be useful for:

- 📚 Academic research
- 📰 News summarization
- 📄 Document analysis
- 🔬 Research papers
- 📖 Educational content
- 🔎 Topic-focused information retrieval


## 📁 Project Structure

```text
TextSummarizer/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 👩‍💻 Author

**Raja Aifa**

Master's Degree in Web Service and Multimedia

---

## 📄 License

This project is intended for educational and research purposes.
