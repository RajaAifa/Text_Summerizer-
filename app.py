import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.translate import meteor_score
from rouge import Rouge
import matplotlib.pyplot as plt
import nltk
import random
from wordcloud import WordCloud


# ============================================================
# NLTK DATA
# ============================================================

@st.cache_resource
def download_nltk_data():
    """
    Download all NLTK resources required by the application.
    This fixes the punkt_tab error that can occur on Streamlit Cloud.
    """
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


download_nltk_data()


# ============================================================
# BERT MODEL
# ============================================================

@st.cache_resource
def load_bert_model():
    """
    Load the pretrained BERT tokenizer and model only once.
    Caching is important for Streamlit Cloud performance.
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    model.eval()

    return tokenizer, model


# ============================================================
# BERT EXTRACTIVE SUMMARIZER
# ============================================================

class NeuralExtractiveSummarizer:

    def __init__(self):
        self.tokenizer, self.model = load_bert_model()

    def get_embedding(self, text):
        """
        Generate a BERT embedding for a piece of text.
        """

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pooling over the token embeddings
        attention_mask = inputs["attention_mask"]

        token_embeddings = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()

        summed = torch.sum(token_embeddings * mask, dim=1)

        summed_mask = torch.clamp(mask.sum(dim=1), min=1e-9)

        embedding = summed / summed_mask

        return embedding.squeeze(0)

    def summarize(self, text, percentage):

        sentences = sent_tokenize(text)

        if not sentences:
            return ""

        # Number of sentences to keep
        num_sentences = len(sentences)

        num_sentences_to_select = max(
            1,
            int(num_sentences * percentage / 100)
        )

        # If the percentage is high enough, return all sentences
        if num_sentences_to_select >= num_sentences:
            return " ".join(sentences)

        # --------------------------------------------------------
        # Generate embedding for the complete document
        # --------------------------------------------------------

        document_embedding = self.get_embedding(text)

        # Normalize document embedding
        document_embedding = document_embedding / (
            torch.norm(document_embedding) + 1e-9
        )

        sentence_scores = []

        # --------------------------------------------------------
        # Score each sentence according to its similarity
        # to the complete document
        # --------------------------------------------------------

        for index, sentence in enumerate(sentences):

            sentence_embedding = self.get_embedding(sentence)

            sentence_embedding = sentence_embedding / (
                torch.norm(sentence_embedding) + 1e-9
            )

            similarity = torch.dot(
                document_embedding,
                sentence_embedding
            )

            sentence_scores.append(
                (index, similarity.item())
            )

        # Sort by relevance score
        sentence_scores.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # Select top sentences
        selected_indices = [
            index
            for index, score in sentence_scores[
                :num_sentences_to_select
            ]
        ]

        # Keep original document order
        selected_indices.sort()

        summary = [
            sentences[index]
            for index in selected_indices
        ]

        return " ".join(summary)


# ============================================================
# ABSTRACTIVE COMPRESSION
# ============================================================

class AbstractiveCompressor:

    def __init__(self):
        self.stop_words = set(
            stopwords.words("english")
        )

    def compress(self, text):

        if not text.strip():
            return ""

        sentences = sent_tokenize(text)

        compressed_sentences = []

        for sentence in sentences:

            words = word_tokenize(sentence)

            filtered_words = [
                word.lower()
                for word in words
                if word.isalnum()
                and word.lower() not in self.stop_words
            ]

            compressed_words = self.replace_with_synonyms(
                filtered_words
            )

            compressed_sentence = " ".join(
                compressed_words
            )

            compressed_sentences.append(
                compressed_sentence
            )

        return " ".join(compressed_sentences)

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

                if "_" not in synonym:
                    synonyms.append(synonym)

        # Remove duplicates
        synonyms = list(set(synonyms))

        return synonyms


# ============================================================
# INTERACTIVE SUMMARIZER
# ============================================================

class InteractiveSummarizer:

    def summarize_with_context(self, text, query):

        if not text.strip():
            return ""

        if not query.strip():
            return text

        query_tokens = [
            token.lower()
            for token in word_tokenize(query)
            if token.isalnum()
        ]

        sentences = sent_tokenize(text)

        relevant_sentences = []

        for sentence in sentences:

            sentence_tokens = [
                token.lower()
                for token in word_tokenize(sentence)
                if token.isalnum()
            ]

            # Count matching query words
            matching_words = sum(
                token in sentence_tokens
                for token in query_tokens
            )

            if matching_words > 0:
                relevant_sentences.append(
                    (sentence, matching_words)
                )

        # Sort by relevance
        relevant_sentences.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return " ".join(
            sentence
            for sentence, score
            in relevant_sentences
        )


# ============================================================
# MAIN SUMMARIZATION SYSTEM
# ============================================================

class SummarizationSystem:

    def __init__(self):

        self.extractive_summarizer = (
            NeuralExtractiveSummarizer()
        )

        self.abstractive_compressor = (
            AbstractiveCompressor()
        )

        self.interactive_summarizer = (
            InteractiveSummarizer()
        )

        self.rouge = Rouge()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    def evaluate(self, hypothesis, reference):

        if not hypothesis.strip() or not reference.strip():
            return None, None

        try:

            hypothesis_tokens = word_tokenize(
                hypothesis.lower()
            )

            reference_tokens = word_tokenize(
                reference.lower()
            )

            rouge_scores = self.rouge.get_scores(
                hypothesis,
                reference
            )

            meteor_score_value = (
                meteor_score.meteor_score(
                    [reference_tokens],
                    hypothesis_tokens
                )
            )

            return rouge_scores, meteor_score_value

        except Exception as error:

            st.warning(
                f"Evaluation could not be completed: {error}"
            )

            return None, None

    # --------------------------------------------------------
    # Word Cloud
    # --------------------------------------------------------

    def visualize_wordcloud(self, summary):

        if not summary.strip():
            st.warning(
                "No summary available for the word cloud."
            )
            return

        wordcloud = WordCloud(
            width=900,
            height=450,
            background_color="white"
        ).generate(summary)

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )

        ax.imshow(
            wordcloud,
            interpolation="bilinear"
        )

        ax.axis("off")

        st.pyplot(fig)

        plt.close(fig)

    # --------------------------------------------------------
    # Run complete summarization
    # --------------------------------------------------------

    def run_summarization(
        self,
        text,
        reference_summary,
        percentage,
        query
    ):

        if not text.strip():

            st.error(
                "Please enter some text to summarize."
            )

            return

        # ====================================================
        # EXTRACTIVE SUMMARY
        # ====================================================

        with st.spinner(
            "Generating extractive summary with BERT..."
        ):

            summary = (
                self.extractive_summarizer.summarize(
                    text,
                    percentage
                )
            )

        st.subheader(
            "🤖 Extractive Summary"
        )

        st.write(summary)

        # ====================================================
        # ABSTRACTIVE COMPRESSION
        # ====================================================

        with st.spinner(
            "Generating compressed summary..."
        ):

            compressed_summary = (
                self.abstractive_compressor.compress(
                    summary
                )
            )

        st.subheader(
            "✨ Abstractive Compression"
        )

        st.write(compressed_summary)

        # ====================================================
        # INTERACTIVE SUMMARY
        # ====================================================

        if query.strip():

            interactive_summary = (
                self.interactive_summarizer
                .summarize_with_context(
                    compressed_summary,
                    query
                )
            )

            st.subheader(
                "🔎 Query-Driven Summary"
            )

            if interactive_summary:

                st.write(
                    interactive_summary
                )

            else:

                st.info(
                    "No sentences matched your query."
                )

        # ====================================================
        # EVALUATION
        # ====================================================

        if reference_summary.strip():

            with st.spinner(
                "Calculating ROUGE and METEOR scores..."
            ):

                rouge_scores, meteor_value = (
                    self.evaluate(
                        summary,
                        reference_summary
                    )
                )

            if rouge_scores is not None:

                st.subheader(
                    "📊 Evaluation"
                )

                # ROUGE results
                rouge_result = rouge_scores[0]

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "ROUGE-1 F1",
                        f"{rouge_result['rouge-1']['f']:.4f}"
                    )

                with col2:

                    st.metric(
                        "ROUGE-2 F1",
                        f"{rouge_result['rouge-2']['f']:.4f}"
                    )

                with col3:

                    st.metric(
                        "ROUGE-L F1",
                        f"{rouge_result['rouge-l']['f']:.4f}"
                    )

                st.metric(
                    "METEOR",
                    f"{meteor_value:.4f}"
                )

        else:

            st.info(
                "Enter a reference summary to calculate "
                "ROUGE and METEOR scores."
            )

        # ====================================================
        # WORD CLOUD
        # ====================================================

        st.subheader(
            "☁️ Word Cloud"
        )

        self.visualize_wordcloud(summary)


# ============================================================
# STREAMLIT APPLICATION
# ============================================================

def main():

    st.set_page_config(
        page_title="Text Summarizer",
        page_icon="📝",
        layout="wide"
    )

    st.title(
        "📝 Text Summarization System"
    )

    st.write(
        "This app summarizes text using "
        "multiple NLP techniques."
    )

    st.markdown(
        """
        ### ✨ Available Techniques

        - 🤖 **BERT-based Extractive Summarization**
        - ✨ **WordNet-based Text Compression**
        - 🔎 **Query-Driven Interactive Summarization**
        - 📊 **ROUGE & METEOR Evaluation**
        - ☁️ **Word Cloud Visualization**
        """
    )

    st.divider()

    # ========================================================
    # INPUT TEXT
    # ========================================================

    text = st.text_area(
        "📄 Input Text",
        height=300,
        placeholder=(
            "Paste the text you want to summarize here..."
        )
    )

    # ========================================================
    # REFERENCE SUMMARY
    # ========================================================

    reference_summary = st.text_area(
        "📋 Reference Summary",
        height=150,
        placeholder=(
            "Enter a reference summary here "
            "to calculate ROUGE and METEOR scores..."
        )
    )

    # ========================================================
    # QUERY
    # ========================================================

    query = st.text_input(
        "🔎 Query / Topic",
        placeholder=(
            "Example: artificial intelligence, healthcare, education..."
        )
    )

    # ========================================================
    # PERCENTAGE
    # ========================================================

    percentage = st.slider(
        "📏 Percentage of Text to Include in Summary",
        min_value=1,
        max_value=100,
        value=30,
        step=1
    )

    st.caption(
        f"{percentage}% of the original sentences "
        "will be selected for the extractive summary."
    )

    # ========================================================
    # SUMMARIZE BUTTON
    # ========================================================

    if st.button(
        "🚀 Summarize",
        type="primary",
        use_container_width=True
    ):

        summarization_system = (
            SummarizationSystem()
        )

        summarization_system.run_summarization(
            text=text,
            reference_summary=reference_summary,
            percentage=percentage,
            query=query
        )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
