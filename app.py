import os
import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

st.set_page_config(
    page_title="Job Market Analysis", 
    page_icon="📊",
    layout="wide"
)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Topic Modeling", "Job Classification"]
)

st.title("LDA Topic Explorer (Python / Streamlit)")

# --- Load data (upload or repo CSV) ---
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
        st.warning("No CSV found in repo. Upload a CSV or place linkedin_scraped_job_details_1600.csv in this folder.")
        df = pd.DataFrame(columns=["url", "title", "company", "location", "description"])

st.sidebar.markdown("---")
st.sidebar.write(f"Rows in dataset: {len(df)}")

# --- Model controls ---
st.sidebar.header("LDA Controls")
num_topics = st.sidebar.number_input("Number of topics", min_value=2, max_value=50, value=6, step=1)
passes = st.sidebar.number_input("Training passes", min_value=1, max_value=50, value=10, step=1)
run_remote = st.sidebar.checkbox("Use repo lda_analysis.run_lda (may train model)", value=False)
show_pyldavis = st.sidebar.checkbox("Embed lda_vis.html if available", value=True)

# persistent storage in session state
if "topics" not in st.session_state:
    st.session_state.topics = []  # list of dict: {id, top_terms, prevalence}
if "lda_meta" not in st.session_state:
    st.session_state.lda_meta = {}

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Topics")
    if run_remote:
        # try to import and run lda_analysis.run_lda (heavy)
        try:
            import lda_analysis
            csv_path = None
            if uploaded is not None:
                # save uploaded file temp to pass to run_lda
                csv_path = "uploaded_jobs.csv"
                df.to_csv(csv_path, index=False)
            else:
                csv_path = "linkedin_scraped_job_details_1600.csv"
            if st.button("Run LDA now (this may take minutes)"):
                with st.spinner("Training LDA..."):
                    try:
                        lda_model, corpus, id2word = lda_analysis.run_lda(csv_path, num_topics=num_topics, passes=int(passes), save_vis=False)
                        st.success("LDA training finished.")
                        # derive topics and per-document assignments
                        topics = []
                        for tid in range(num_topics):
                            words = lda_model.show_topic(tid, topn=10)
                            topics.append({
                                "id": tid,
                                "top_terms": [{"term": w, "weight": float(p)} for (w, p) in words],
                            })
                        # compute prevalence across corpus
                        topic_counts = {i: 0.0 for i in range(num_topics)}
                        doc_topics = []
                        for doc in corpus:
                            doc_dist = lda_model.get_document_topics(doc, minimum_probability=0.0)
                            # doc_dist is list of (topicid, prob)
                            dominant = max(doc_dist, key=lambda x: x[1])
                            topic_counts[dominant[0]] += 1
                            doc_topics.append(dominant)
                        total_docs = max(1, len(corpus))
                        for t in topics:
                            t["prevalence"] = topic_counts[t["id"]] / total_docs
                        st.session_state.topics = topics
                        st.session_state.lda_meta = {"model_present": True}
                        # cache doc-topic mapping for quick examples lookup
                        st.session_state.doc_topics = doc_topics
                        st.session_state.corpus = corpus
                        st.session_state.id2word = id2word
                        st.session_state.lda_model = lda_model
                    except Exception as e:
                        st.error(f"LDA training failed: {e}")
        except Exception as e:
            st.error("Could not import lda_analysis.py from repo. Make sure it exists and is importable.")
    else:
        # show topics from session_state or sample
        if not st.session_state.topics:
            # fallback sample topics if none
            st.session_state.topics = [
                {"id": 0, "top_terms": [{"term":"business","weight":0.014},{"term":"team","weight":0.009},{"term":"service","weight":0.009}], "prevalence":0.15},
                {"id": 1, "top_terms": [{"term":"developer","weight":0.012},{"term":"web","weight":0.010},{"term":"team","weight":0.011}], "prevalence":0.22},
                {"id": 2, "top_terms": [{"term":"data","weight":0.026},{"term":"team","weight":0.020},{"term":"solution","weight":0.012}], "prevalence":0.18},
            ]

        # render topic cards
        cols = st.columns(2)
        for i, topic in enumerate(st.session_state.topics):
            target_col = cols[i % len(cols)]
            with target_col:
                st.markdown(f"### Topic {topic['id']}")
                coh = topic.get("coherence")
                if coh:
                    st.caption(f"Coherence: {coh:.3f}")
                prevalence = topic.get("prevalence")
                if prevalence is not None:
                    st.caption(f"Prevalence: {prevalence*100:.1f}%")
                terms = ", ".join([t["term"] for t in topic["top_terms"][:10]])
                st.write(terms)
                if st.button(f"View examples (topic {topic['id']})", key=f"examples_{topic['id']}"):
                    # show top example rows for this topic
                    if st.session_state.get("lda_model") is not None and st.session_state.get("doc_topics") is not None:
                        # find docs with that dominant topic
                        matches = [i for i, d in enumerate(st.session_state.doc_topics) if d[0] == topic["id"]]
                        sample_idx = matches[:10]
                        if not sample_idx:
                            st.info("No example jobs found for this topic.")
                        else:
                            for idx in sample_idx:
                                row = df.iloc[idx] if idx < len(df) else None
                                if row is not None:
                                    st.markdown(f"**{row.get('title', 'No title')}** — {row.get('company','')}")
                                    st.write(row.get("description", "")[:400] + ("..." if len(str(row.get("description","")))>400 else ""))
                                    st.write(f"[Original URL]({row.get('url','')})")
                    else:
                        st.info("No trained LDA model in session. Either run the model or load a model that saved doc-topic mappings.")

with col2:
    st.header("Details & Visuals")
    if show_pyldavis and os.path.exists("lda_vis.html"):
        st.subheader("pyLDAvis interactive view")
        html = open("lda_vis.html", "r", encoding="utf-8").read()
        components.html(html, height=700, scrolling=True)
    else:
        if show_pyldavis:
            st.info("lda_vis.html not found in project root. Run the analysis script that saves pyLDAvis or place lda_vis.html here.")
    st.markdown("---")
    st.subheader("Dataset preview")
    st.dataframe(df.head(50))

st.markdown("---")
st.caption("This Streamlit app can call your repo's lda_analysis.run_lda to train a model. Training runs in-process and may take several minutes depending on data size.")