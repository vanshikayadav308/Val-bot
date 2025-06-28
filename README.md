#  valBot - LLM Comparison Dashboard

Compare responses from GPT-4 and Claude 3 to any prompt.

## Features
- Prompt input
- GPT-4 + Claude 3 API integration
- Side-by-side response display
- Aesthetic UI (dark theme)
  

## Run Locally
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m textblob.download_corpora
streamlit run app.py

