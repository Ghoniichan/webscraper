"""
lda_analysis.py

Run LDA topic modeling on the scraped LinkedIn job descriptions.

Features:
- Loads `linkedin_scraped_job_details_1600.csv` (fallbacks included)
- Text cleaning, tokenization, stop-word removal, lemmatization (spaCy)
- Bigrams/trigrams using Gensim Phrases
- Builds dictionary and corpus, fits Gensim LDA model
- Computes coherence score and prints top topics
- Saves pyLDAvis visualization to `lda_vis.html`

See README.md for run instructions and requirements.
"""

import os
import re
import logging
from typing import List, Optional

import pandas as pd

# NLP & topic modeling
import nltk
from nltk.corpus import stopwords

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

try:
    # If pyLDAvis not installed, script still runs but won't save interactive view
    import pyLDAvis.gensim_models as gensimvis
    import pyLDAvis
    HAVE_PYLDAVIS = True
except Exception:
    HAVE_PYLDAVIS = False

# Configure logging for gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load_data(csv_path: str) -> pd.DataFrame:
    """Load CSV and try common fallback columns for text."""
    df = pd.read_csv(csv_path)
    # prefer 'description', fallback to 'job_description' or combine title+description
    if 'description' in df.columns:
        df = df.dropna(subset=['description'])
    elif 'job_description' in df.columns:
        df = df.dropna(subset=['job_description'])
        df['description'] = df['job_description']
    else:
        # fallback to combine title + company or fail gracefully
        if 'title' in df.columns:
            df['description'] = df['title'].fillna('')
        else:
            raise ValueError('No description-like column found in CSV')
    return df


def clean_text(text: str) -> str:
    """Basic cleaning: remove URLs, html, non-alphanumeric, extra whitespace."""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def ensure_nltk() -> None:
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')


def load_spacy_model():
    """Load spaCy English model (en_core_web_sm). Provide friendly error if missing."""
    try:
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    except OSError:
        raise RuntimeError("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
    return nlp


def preprocess_texts(texts: List[str], nlp, extra_stopwords: Optional[set] = None) -> List[List[str]]:
    """Preprocess a list of raw texts into token lists ready for topic modeling.

    Steps:
    - clean
    - tokenize with spaCy
    - lemmatize
    - remove stopwords and short tokens / numbers
    """

    stop_words = set(stopwords.words('english'))
    # also include spaCy default stop words
    stop_words |= set([w.lower() for w in STOP_WORDS])
    if extra_stopwords:
        stop_words |= set(extra_stopwords)

    processed_texts = []
    for doc in nlp.pipe(map(clean_text, texts), batch_size=50):
        tokens = []
        for token in doc:
            # filter punctuation/stopwords/numeric/space
            if token.is_space or token.is_punct or token.is_stop:
                continue
            lemma = token.lemma_.lower().strip()
            if not lemma:
                continue
            if lemma in stop_words:
                continue
            if lemma.isdigit():
                continue
            if len(lemma) <= 2:
                continue
            tokens.append(lemma)
        processed_texts.append(tokens)
    return processed_texts


def build_bi_tri_grams(texts, min_count=5, threshold=100):
    """Build bigram and trigram models and return transformed texts."""
    bigram = gensim.models.Phrases(texts, min_count=min_count, threshold=threshold)
    trigram = gensim.models.Phrases(bigram[texts], threshold=threshold)

    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    texts_bigrams = [bigram_mod[doc] for doc in texts]
    texts_trigrams = [trigram_mod[bigram_mod[doc]] for doc in texts]
    return texts_bigrams, texts_trigrams, bigram_mod, trigram_mod


def run_lda(csv_path: str, num_topics: int = 8, passes: int = 10, save_vis: bool = True):
    df = load_data(csv_path)
    texts_raw = df['description'].astype(str).tolist()

    ensure_nltk()
    nlp = load_spacy_model()

    print("Preprocessing texts (tokenize, remove stopwords, lemmatize)...")
    processed_texts = preprocess_texts(texts_raw, nlp)

    print("Building bigram/trigram models...")
    _, texts_trigrams, bigram_mod, trigram_mod = build_bi_tri_grams(processed_texts)

    # Create dictionary and corpus
    id2word = corpora.Dictionary(texts_trigrams)
    id2word.filter_extremes(no_below=5, no_above=0.5)
    corpus = [id2word.doc2bow(text) for text in texts_trigrams]

    print(f"Training LDA model with {num_topics} topics and {passes} passes...")
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=passes,
        alpha='auto',
        per_word_topics=True
    )

    # Print the top topics
    print('\nTop topics:')
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic: {idx}\nWords: {topic}\n")

    # Compute coherence score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts_trigrams, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(f"Coherence Score (c_v): {coherence_lda:.4f}")

    if save_vis and HAVE_PYLDAVIS:
        print("Preparing pyLDAvis visualization (this may take a few seconds)...")
        vis = gensimvis.prepare(lda_model, corpus, id2word)
        outpath = os.path.join(os.getcwd(), 'lda_vis.html')
        pyLDAvis.save_html(vis, outpath)
        print(f"pyLDAvis saved to: {outpath}")
    elif save_vis:
        print("pyLDAvis not available; install pyLDAvis to save interactive visualization.")

    return lda_model, corpus, id2word


if __name__ == '__main__':
    # sensible defaults; change as needed
    CSV_PATH = os.path.join(os.getcwd(), 'linkedin_scraped_job_details_1600.csv')
    if not os.path.exists(CSV_PATH):
        # also try data folder
        alt = os.path.join(os.getcwd(), 'data', 'linkedin_all_tech_public_jobs.csv')
        if os.path.exists(alt):
            CSV_PATH = alt
        else:
            raise FileNotFoundError(f"Could not find CSV at expected locations. Looked at {CSV_PATH} and {alt}")

    lda, corpus, id2word = run_lda(CSV_PATH, num_topics=8, passes=12, save_vis=True)
