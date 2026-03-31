# Fraudulent Job Posting Classifier

A robust machine learning-based application designed to identify and flag fraudulent job postings. It evaluates the risk of a job description by combining statistical NLP models, semantic red-flag detection, personal data exposure risks, and employer credibility analysis.

## Features

* **Machine Learning Model**: Built with `scikit-learn`, the system utilizes a `LinearSVC` classifier combined with a `TfidfVectorizer` (for textual features) and `OneHotEncoder` (for categorical inputs). The dataset's class imbalance is handled using `RandomOverSampler`.
* **Automated Red-Flag Mining**: Uses chi-square ($\chi^2$) statistic selection to automatically mine n-grams (unigrams and bigrams) most associated with fraudulent behavior. This avoids reliance on hardcoded keyword lists.
* **Personal Data Risk Analysis**: Acts as a safeguard by checking descriptions for terms indicating unauthorized requests for personal sensitive data (e.g., CVV, Passports, Aadhaar/SSN).
* **Employer Credibility Heuristics**: Calculates a composite 0-100 credibility score based on offline metrics like the presence of a company logo, length of company profiles, known industry presence, and matches against a configurable trusted companies list.
* **Web Interface**: Offers a dynamic Flask-based frontend giving comprehensive reports on predictions, semantic fraud scores, discovered red-flags, and risk levels.

## Project Structure

```text
.
├── app.py                     # Main Flask application and server logic
├── model.py                   # Script for training the LinearSVC pipeline and mining red flags
├── fraud_job_model.pkl        # Serialized, pre-trained ML model (joblib)
├── fraud_job_meta.pkl         # Saved column metadata necessary for inferences
├── red_flags.json             # Auto-generated frequent fraudulent n-grams (via Text Chi2)
├── trusted_companies.json     # Configuration file for verified authentic employers
├── templates/                 # HTML templates for the Web UI
├── static/                    # Assorted CSS and JS statically served files
└── fake_job_posting.ipynb     # Interactive Jupyter Notebook to examine and test the workflow
```

> **Note:** The training dataset `fake_job_postings.csv` has been excluded from the repository. Ensure you place it in the application's root directly if you intend to execute the model-training script (`model.py`).

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. It is highly recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```
*(Optionally explicitly install: `flask`, `scikit-learn`, `pandas`, `imbalanced-learn`, `joblib`, `numpy`)*

### 1. Training the Model (Optional)

If you'd like to retrain the model and regenerate `red_flags.json`, ensure your dataset (`fake_job_postings.csv`) is in the base directory and execute:

```bash
python model.py
```

### 2. Running the Application

To launch the web interface, initialize the Flask server:

```bash
python app.py
```
Afterwards, navigate to `http://127.0.0.1:5000/` in your browser.

## Using the API

You can interact with the classifier via the application's `/predict_api` route explicitly utilizing JSON payloads. See `app.py` for acceptable schema values.

## Dataset Acknowledgement

The model relies on publicly available fake employment distributions in structural formats (e.g., EMSCAD dataset). Due to file size best practices, it is ignored via `.gitignore` and handled locally.
