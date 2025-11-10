# Job Market Analysis with LDA Topic Modeling and Classification

This project analyzes job market demands using Topic Modeling (LDA) and Machine Learning classification on LinkedIn job descriptions.

## Quick Start

1. Create a virtual environment and install dependencies:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Install NLTK data and models:
```python
python -m nltk.downloader punkt stopwords wordnet
```

3. Run the Streamlit app:
```powershell
streamlit run app_new.py
```

## Project Components

### 1. Topic Modeling
- Implemented in `Labor_Market_Demand.ipynb`
- Uses LDA (Latent Dirichlet Allocation) for topic discovery
- Interactive visualization through Streamlit interface

### 2. Classification Model
- Built on top of topic modeling results
- Predicts job categories for new descriptions
- Provides confidence scores and probability distributions

## Implementation Process

### Part 1: Topic Modeling Pipeline

#### 1. Data Preprocessing
- Load and clean job descriptions
- Tokenization and lemmatization
- Stop words removal
- Text normalization

#### 2. Feature Engineering
- Document-Term Matrix creation
- TF-IDF vectorization
- Corpus preparation for LDA

#### 3. LDA Implementation
- Hyperparameter tuning:
  - Number of topics (optimal: 6)
  - Training passes (default: 10)
- Model training with optimal parameters
- Topic coherence evaluation

#### 4. Topic Interpretation
Discovered topics:
1. Business operations and accessibility
2. Web and software development
3. AI-driven hiring and HR tech
4. Digital business and customer engagement
5. Data and analytics
6. Software engineering and development teams

### Part 2: Classification Model

#### 1. Feature Processing
- TF-IDF vectorization of job descriptions
- Label encoding of topic categories

#### 2. Model Development
- Train/Test split (80/20)
- Multinomial Naive Bayes classifier
- Model evaluation and validation

#### 3. Model Deployment
Saved components in `models/` directory:
- `job_classifier_model.joblib`
- `tfidf_vectorizer.joblib`
- `category_mappings.joblib`

## Web Interface Features

### Topic Modeling Section
- Interactive topic number selection (2-50)
- Training passes adjustment (1-50)
- Topic visualization and analysis
- Coherence score display

### Classification Section
- Real-time job description classification
- Confidence score visualization
- Category probability distribution
- Interactive results display

## Model Parameters

### LDA Configuration
```python
LdaMulticore(
    num_topics=optimal_num_topics,  # Determined by coherence scores
    passes=10,                      # Number of training iterations
    random_state=100,              # For reproducibility
    workers=2                      # Parallel processing
)
```

### Classification Settings
```python
TfidfVectorizer(
    max_df=0.95,  # Ignore terms in >95% of docs
    min_df=2      # Ignore terms in <2 docs
)

MultinomialNB()  # Naive Bayes classifier
```

## Files Description

- `Labor_Market_Demand.ipynb`: Main notebook for model development
- `app_new.py`: Streamlit web application
- `requirements.txt`: Project dependencies
- `models/`: Saved model components
- `data/`: Dataset directory

## Results and Metrics

- Topic Coherence Score: [Latest score from your model]
- Classification Accuracy: [Latest accuracy from your model]
- Detailed performance metrics available in the notebook

## Usage Examples

1. **Topic Analysis:**
```python
from lda_analysis import run_lda
model, corpus, dictionary = run_lda('path_to_data.csv', num_topics=6, passes=10)
```

2. **Job Classification:**
```python
from app_new import classify_job_description
category, confidence = classify_job_description(job_text)
```

## Notes

- The models are optimized for tech industry job descriptions
- Regular retraining recommended as job market evolves
- Adjust parameters based on specific use cases
