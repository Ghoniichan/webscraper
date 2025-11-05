# LDA Topic Modeling for LinkedIn Scrapes

This workspace contains `visualization.py` (existing) and a new `lda_analysis.py` which performs LDA topic modeling on the scraped job descriptions.

Quick start

- Create a virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
# Then download the spaCy model:
python -m spacy download en_core_web_sm
```

- Run the LDA script from the repository root (it expects `linkedin_scraped_job_details_1600.csv` in the root or `data/`):

```powershell
python lda_analysis.py
```

Outputs

- Prints top topics and coherence score to stdout.
- If `pyLDAvis` is installed, saves `lda_vis.html` in the working directory for interactive exploration.

Notes

- The script uses spaCy for tokenization/lemmatization and NLTK stopwords. If `en_core_web_sm` is missing, run the download command above.
- Adjust `num_topics` and `passes` in `lda_analysis.py` or call `run_lda()` programatically for tuning.
