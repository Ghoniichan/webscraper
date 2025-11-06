import os
import streamlit as st

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Job Market Analysis",
    page_icon="📊",
    layout="wide"
)

import pandas as pd
import numpy as np
import streamlit.components.v1 as components
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    import lda_analysis
except ImportError:
    st.warning("lda_analysis module not found. Topic modeling functionality might be limited.")

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {str(e)}")
        return False

# Initialize NLTK data
nltk_ready = download_nltk_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Topic Modeling", "Job Classification"]
)

def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stop words and lemmatize
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    
    return processed_tokens

def classify_job():
    st.title("Job Classification")
    st.write("Analyze and classify job descriptions into market demand categories")
    
    # Check if NLTK data is available
    if not nltk_ready:
        st.error("NLTK data is not available. Classification may not work properly.")
        return
    
    try:
        # Load the classification model and components
        model_path = 'models/job_classifier_model.joblib'
        vectorizer_path = 'models/tfidf_vectorizer.joblib'
        mappings_path = 'models/category_mappings.joblib'
        
        if not all(os.path.exists(p) for p in [model_path, vectorizer_path, mappings_path]):
            st.error("""
            Some model files are missing. Please make sure you have the following files in the 'models' directory:
            - job_classifier_model.joblib
            - tfidf_vectorizer.joblib
            - category_mappings.joblib
            """)
            return
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        mappings = joblib.load(mappings_path)
        
        # Text input for classification
        job_description = st.text_area(
            "Enter job description:",
            height=200,
            placeholder="Paste job description here..."
        )
        
        if st.button("Classify"):
            if not job_description or job_description.isspace():
                st.warning("Please enter a job description.")
                return
                
            # Show processing message
            with st.spinner("Analyzing job description..."):
                try:
                    # Preprocess the input text
                    processed_text = preprocess_text(job_description)
                    if not processed_text:
                        st.warning("No valid words found in the input after preprocessing.")
                        return
                        
                    text_joined = ' '.join(processed_text)
                    
                    # Transform the text using the vectorizer
                    X = vectorizer.transform([text_joined])
                    
                    # Get the prediction and probability scores
                    prediction = model.predict(X)[0]
                    probabilities = model.predict_proba(X)[0]
                    confidence = probabilities[prediction]
                    
                    # Map the numerical prediction to category name
                    predicted_category = mappings['topic_to_category'][prediction]
                    
                    # Display results in columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info("**Predicted Category:**")
                        st.write(f"### {predicted_category}")
                    
                    with col2:
                        st.info("**Confidence Score:**")
                        st.write(f"### {confidence:.1%}")
                    
                    # Add confidence meter
                    st.progress(confidence)
                    
                    # Show detailed probability distribution
                    st.subheader("Probability Distribution")
                    prob_df = pd.DataFrame({
                        'Category': [mappings['topic_to_category'][i] for i in range(len(probabilities))],
                        'Probability': probabilities
                    }).sort_values('Probability', ascending=False)
                    
                    fig = px.bar(
                        prob_df,
                        x='Category',
                        y='Probability',
                        title='Confidence Scores by Category'
                    )
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during classification: {str(e)}")
    
    except Exception as e:
        st.error(f"Error loading model components: {str(e)}")
        if "Permission denied" in str(e):
            st.info("Tip: Make sure you have read permissions for the model files.")

def topic_modeling_page():
    st.title("LDA Topic Explorer")
    st.write("Analyze job descriptions using LDA Topic Modeling")

    # File upload
    uploaded = st.sidebar.file_uploader("Upload job CSV (optional)", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.sidebar.success("Loaded uploaded CSV")
    else:
        default_path = "linkedin_scraped_job_details_1600.csv"
        if os.path.exists(default_path):
            df = pd.read_csv(default_path)
            st.sidebar.write(f"Loaded {default_path}")
        else:
            st.warning("No CSV found. Upload a CSV or place linkedin_scraped_job_details_1600.csv in this folder.")
            df = pd.DataFrame(columns=["url", "title", "company", "location", "description"])

    st.sidebar.markdown("---")
    st.sidebar.write(f"Rows in dataset: {len(df)}")

    # LDA Controls
    st.sidebar.header("LDA Controls")
    num_topics = st.sidebar.number_input("Number of topics", min_value=2, max_value=50, value=6, step=1)
    passes = st.sidebar.number_input("Training passes", min_value=1, max_value=50, value=10, step=1)
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Topics")
        if st.button("Run LDA Analysis", key="run_lda"):
            with st.spinner("Training LDA model... This may take a few minutes."):
                try:
                    # Save uploaded file if necessary
                    csv_path = default_path
                    if uploaded is not None:
                        csv_path = "uploaded_jobs.csv"
                        df.to_csv(csv_path, index=False)
                    
                    # Run LDA (do not rely on pyLDAvis html embed)
                    lda_model, corpus, id2word = lda_analysis.run_lda(
                        csv_path,
                        num_topics=num_topics,
                        passes=passes,
                        save_vis=False
                    )

                    # Build topic word DataFrames and display modern Plotly charts
                    topic_word_dfs = []
                    for tid in range(num_topics):
                        words = lda_model.show_topic(tid, topn=15)
                        words_df = pd.DataFrame(words, columns=["Word", "Weight"])  # (word, prob)
                        topic_word_dfs.append(words_df)
                        st.subheader(f"Topic {tid + 1}")
                        st.plotly_chart(px.bar(words_df, x='Word', y='Weight', title=f'Topic {tid+1} top words'))

                    # Compute average topic probability across documents (prevalence)
                    num_docs = max(1, len(corpus))
                    topic_probs = np.zeros((num_docs, num_topics))
                    for i, doc in enumerate(corpus):
                        doc_dist = dict(lda_model.get_document_topics(doc, minimum_probability=0.0))
                        for t in range(num_topics):
                            topic_probs[i, t] = doc_dist.get(t, 0.0)
                    avg_topic_prob = topic_probs.mean(axis=0)
                    topic_prev_df = pd.DataFrame({
                        'Topic': [f'Topic {i+1}' for i in range(num_topics)],
                        'AvgProb': avg_topic_prob
                    })

                    # Save for interactive use in the right column
                    st.session_state['topic_word_dfs'] = topic_word_dfs
                    st.session_state['topic_prev_df'] = topic_prev_df
                    st.session_state['lda_model'] = lda_model

                    st.success("LDA analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"Error during LDA analysis: {str(e)}")
                finally:
                    # Cleanup if needed
                    if uploaded is not None and os.path.exists("uploaded_jobs.csv"):
                        os.remove("uploaded_jobs.csv")

    with col2:
        st.header("Topic Overview & Controls")

        if st.session_state.get('topic_prev_df') is not None:
            # Topic prevalence bar chart
            st.subheader("Topic prevalence (average probability across documents)")
            fig_prev = px.bar(st.session_state['topic_prev_df'], x='Topic', y='AvgProb', title='Topic prevalence')
            st.plotly_chart(fig_prev)

            # Topic selector to show top words
            topic_names = [f'Topic {i+1}' for i in range(len(st.session_state['topic_word_dfs']))]
            selected = st.selectbox("Select topic to inspect", topic_names)
            tid = int(selected.split()[1]) - 1
            words_df = st.session_state['topic_word_dfs'][tid]
            st.subheader(f"Top words for {selected}")
            st.plotly_chart(px.bar(words_df, x='Word', y='Weight', title=f'Top words - {selected}'))

            # Optional: show small dataset preview
            st.markdown("---")
            st.subheader("Dataset Preview")
            st.dataframe(df.head())
        else:
            st.info("Run the LDA analysis to generate interactive visualizations.")
            st.markdown("---")
            st.subheader("Dataset Preview")
            st.dataframe(df.head())

# Main app logic
if page == "Topic Modeling":
    topic_modeling_page()
else:
    classify_job()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This application provides tools for analyzing job descriptions "
    "using topic modeling and classification techniques."
)