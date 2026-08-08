# Sentiment Analysis of Player Reactions to Monetisation Systems in Gacha Games Using Natural Language Processing

**Author:** Samson Faluyi
**Programme:** MSc Computer Science with Data Analytics
**Institution:** Sheffield Hallam University
**Project:** Applied Research Project

---

## 1. Project Overview

This repository contains the computational artefact developed for an MSc Applied Research Project investigating sentiment towards monetisation systems in selected gacha games.

The proposed research focuses on player discussions relating to:

* *Genshin Impact*
* *Honkai: Star Rail*
* *Zenless Zone Zero*

The proposed data source is Reddit, with an observation period covering:

**1 January 2025 – 31 December 2025**

The research investigates whether Natural Language Processing (NLP) techniques can be used to classify player sentiment towards gacha-game monetisation and whether different NLP approaches provide different levels of classification performance.

Three NLP approaches are included in the research framework:

1. VADER
2. TF-IDF with Logistic Regression
3. DistilBERT

---

## 2. Research Questions

### RQ1

What sentiments are expressed in discussions relating to monetisation systems in selected gacha games?

### RQ2

How effectively can different NLP approaches classify player sentiment towards these monetisation systems?

---

## 3. Research Methodology

The proposed research uses a mixed-methods methodology combining quantitative sentiment classification with qualitative thematic analysis.

The quantitative component focuses on comparing different NLP approaches for sentiment classification.

The qualitative component uses thematic analysis to identify recurring issues within player discussions about monetisation.

The proposed computational workflow is:

```text
Data Input
    ↓
Data Validation
    ↓
Preprocessing
    ↓
Sentiment Labelling
    ↓
NLP Model Processing
    ↓
Model Evaluation
    ↓
Visualisation
    ↓
Qualitative Thematic Analysis
    ↓
Interpretation
```

---

## 4. Research Scope

The proposed study focuses on three gacha games:

* *Genshin Impact*
* *Honkai: Star Rail*
* *Zenless Zone Zero*

The proposed observation period is:

**1 January 2025 – 31 December 2025**

The intended target dataset is approximately **4,000–6,000 relevant Reddit comments**, subject to availability through an authorised data-access route and institutional ethical approval.

---

## 5. Monetisation Sampling Criteria

The proposed sampling strategy uses predefined monetisation-related keywords.

The initial keyword set includes:

```text
pull
pulls
pity
banner
banners
gacha
summon
wishes
wish
rates
drop rate
spending
spend
whale
whaling
F2P
free-to-play
top-up
battle pass
monetisation
monetization
```

The final keyword list would be documented before any approved empirical data collection.

---

## 6. Sentiment Categories

The proposed supervised classification task uses three sentiment categories:

* **Positive**
* **Neutral**
* **Negative**

The classification target is sentiment towards the **monetisation issue being discussed**, rather than the player's overall opinion of the game.

For example, a player may express positive feelings towards a character while simultaneously criticising the cost of obtaining that character. The annotation protocol therefore focuses specifically on the monetisation-related sentiment.

---

## 7. NLP Models

### 7.1 VADER

VADER is used as the rule-based baseline.

It provides a relatively lightweight approach designed for sentiment analysis of social-media text.

### 7.2 TF-IDF + Logistic Regression

TF-IDF is used to transform textual data into numerical features.

These features are then provided to a Logistic Regression classifier for supervised sentiment classification.

This provides a traditional machine-learning baseline for comparison with the transformer-based approach.

### 7.3 DistilBERT

DistilBERT provides the transformer-based component of the research framework.

The repository includes a DistilBERT-based inference framework.

Task-specific fine-tuning and empirical evaluation using the proposed Reddit dataset were not completed because the required research dataset was not collected.

---

## 8. Evaluation Framework

The proposed model evaluation uses:

* Precision
* Recall
* Macro-F1
* Confusion matrices
* Sentiment class distribution

Macro-F1 is particularly relevant because the three sentiment classes may not contain equal numbers of observations.

The repository contains evaluation utilities designed to support these measures.

---

## 9. Project Status

The project intentionally distinguishes between completed technical work and incomplete empirical research.

### 9.1 Completed

The project has developed the following components:

* research problem definition;
* research questions;
* literature review;
* research methodology;
* data-sampling design;
* sentiment-labelling protocol;
* Python project structure;
* text preprocessing framework;
* VADER sentiment-analysis component;
* TF-IDF and Logistic Regression component;
* DistilBERT-based inference component;
* evaluation framework;
* visualisation framework;
* testing framework; and
* ethics and data-management plan.

### 9.2 Not Completed

The following components were not completed:

* institutional ethical approval for Reddit data collection;
* approved Reddit data collection;
* final empirical Reddit dataset;
* final manual sentiment-labelled dataset;
* task-specific empirical model training where dependent on the unavailable research dataset;
* final empirical model evaluation;
* comparative model-performance results; and
* empirical conclusions regarding Reddit users' sentiment.

No fabricated empirical results are included in this repository.

---

## 10. Ethics and Data Management

The proposed empirical study involves the analysis of user-generated Reddit content.

Appropriate institutional ethical approval and an authorised data-access method are therefore required before collecting the proposed research dataset.

The required ethical approval was not obtained within the available project period. Consequently, the planned Reddit data collection was **not conducted**.

No Reddit research dataset is included in this repository.

The repository does not contain:

* Reddit usernames;
* personal information;
* API credentials;
* passwords;
* private research data; or
* restricted datasets.

The `data/sample_data.csv` file contains synthetic demonstration data only. It does not represent actual Reddit users or research participants.

Researchers intending to reproduce the proposed empirical study should obtain the necessary ethical approval and follow the applicable institutional, legal and platform requirements before collecting data.

---

## 11. Repository Structure

```text
gacha-sentiment-analysis/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── vader_sentiment.py
│   ├── tfidf_logistic_regression.py
│   ├── distilbert_sentiment.py
│   ├── evaluation.py
│   └── visualisation.py
│
├── notebooks/
│   ├── 01_preprocessing_demo.ipynb
│   ├── 02_vader_demo.ipynb
│   ├── 03_tfidf_logistic_regression_demo.ipynb
│   └── 04_distilbert_demo.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   └── test_evaluation.py
│
├── data/
│   ├── README.md
│   └── sample_data.csv
│
├── docs/
│   ├── methodology.md
│   ├── data_collection_protocol.md
│   ├── annotation_guidelines.md
│   └── testing.md
│
├── outputs/
│   ├── figures/
│   └── tables/
│
└── figures/
    └── research_pipeline.md
```

---

## 12. Installation

### Requirements

The project is designed for Python 3.10 or later.

It is recommended that the project is run inside a virtual environment.

### Create a virtual environment

On Windows:

```bash
python -m venv .venv
```

Activate the environment:

```bash
.venv\Scripts\activate
```

### Install dependencies

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

---

## 13. Running the Tests

The automated tests can be run from the repository root using:

```bash
pytest
```

The tests are intended to verify selected preprocessing and evaluation functionality.

Testing the computational components does not constitute empirical evaluation of the proposed research models.

---

## 14. Running the Demonstration

The repository contains synthetic demonstration data that can be used to test the computational pipeline.

The demonstration notebooks cover:

1. preprocessing;
2. VADER sentiment analysis;
3. TF-IDF with Logistic Regression; and
4. DistilBERT-based sentiment inference.

The demonstration results must not be interpreted as findings from the proposed Reddit research.

---

## 15. Reproducibility

The repository is designed to support reproducibility by providing:

* source code;
* dependency information;
* preprocessing procedures;
* model configuration;
* evaluation functions;
* testing code;
* proposed sampling criteria;
* sentiment annotation guidelines; and
* methodological documentation.

Exact Python package versions should be recorded before the final project submission.

No empirical Reddit dataset is included because the planned data collection was not conducted.

---

## 16. Limitations

The principal limitation of the current research artefact is the absence of the proposed empirical Reddit dataset.

Consequently, this repository does not provide:

* empirical player sentiment distributions;
* final model-performance scores;
* comparative empirical model results;
* empirical confusion matrices based on the research dataset; or
* conclusions about the sentiment of the wider Reddit player population.

The technical framework is intended to provide a foundation for future ethically approved empirical deployment.

---

## 17. Responsible Use

This repository is intended for academic research and demonstration.

Researchers intending to collect or analyse Reddit content should ensure compliance with:

* institutional research-ethics requirements;
* applicable data-protection requirements;
* relevant platform policies and terms;
* authorised data-access procedures; and
* applicable research-governance requirements.

The technical ability to access or process online content does not by itself constitute permission to collect or analyse that content for research purposes.

---

## 18. Academic Context

This repository accompanies the Applied Research Project submitted as part of the:

**MSc Computer Science with Data Analytics**
**Sheffield Hallam University**

The repository should be considered alongside the accompanying research paper.

---

## 19. Author

**Samson Faluyi**

MSc Computer Science with Data Analytics
Sheffield Hallam University

---

## 20. Project Disclaimer

This repository represents the computational artefact and methodological framework developed during the project.

The absence of empirical Reddit results is intentional and reflects the project's ethical and data-access constraints.

No empirical claims about player sentiment or comparative model performance should be inferred from the synthetic demonstration data included in this repository.
