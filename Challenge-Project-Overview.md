---

> ## Challenge Advisor: Update & Finalize Your Project Overview
>
> > 💡 **These grey text instructions are just for you, the team's Challenge Advisor; please delete them once you have completed the steps below.**
>
> We've pre-populated this Challenge Project Overview page — which is what will be shared with your Break Through Tech student team in August — using the details from your submission form. You should have received an email inviting you to join this repo as a Collaborator, enabling you to add files and make edits.
> 
> In order for your project to be finalized and assigned to a team, please:
> 1. **Review all sections below** and update or expand any content as needed, making sure to address the SME Feedback in the section immediately below. Look for square brackets to find the places below that require additional inputs from you (e.g., "About [Company / Org Name]").
> 2. **Add your dataset** to the [data folder](data) in this repo.
> 3. **Close the Issue assigned to you in this repo** to let us know that you have made your edits and the overview page is ready for final review. You can do this by going to the _Issues_ tab in the top left section of the menu above, add a comment that says "CA review complete", and click the button to Close the Issue. 
>
> If you're unfamiliar with how to edit a page like this in GitHub, check out [this tutorial](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/handson/edit-readme.html) for a quick overview (start with step 2 and only edit this page), and [this guide](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/markdown.html) on how to use Markdown to compose text.
>
>
> ❌ Remember that this is a public repo. Do NOT include: Proprietary data, PII, API keys, credentials, or anything confidential.

---

## 📋 BTT Internal Evaluation Notes
*(This section is for BTT staff and CAs only — remove before sharing with students)*

### Technical Vetting
| Check | Status | Notes |
| :--- | :--- | :--- |
| Python Compatibility | 🟢 | The tech stack is centered on Python, using Python-compatible libraries for machine learning and data processing. |
| Data Readiness | 🟢 | The IEEE-CIS Fraud Detection dataset is publicly available and appears ready for use, containing 590,000 transactions without significant preprocessing requirements indicated. |
| Resource Check | 🟢 | Users can utilize Google Colab, which offers sufficient computing resources in its free tier without specialized hardware requirements. |

### Internal Scores
- **Student Fit Score:** 7/10
- **Technical Depth Score:** 8/10
- **Overall Recommendation:** REVISE

### Advisor Feedback Draft
The project has a clear application and relevant industry context, encouraging student engagement. However, the diversity of techniques may introduce confusion. Simplifying modeling options or focusing on a few key methods could improve clarity and success. Additionally, ensure that students are equipped with the skills to interpret model results effectively, as this is essential for real-world applications. Recommend providing workshops or resources on the specific ML libraries to be used.

---

# Online Payment Fraud Detection Using High-Dimensional Transaction Data

**Company / Org:** Mastercard  
**Challenge Advisor:** Debasmita Das, debasmita.das@iiml.org  
**AI Studio Coach:** Harshini Donepudi, harshini.donepudi@breakthroughtech.org  
**Program:** Break Through Tech AI Studio – Fall 2026


## 🏢 About Mastercard
Mastercard is a global leader in the payment technology industry, connecting billions of consumers, financial institutions, and merchants through innovative digital transaction solutions. 

---

## 🎯 The Challenge
### Project Summary
In this project, you will use online payment transaction data and machine-learning and deep-learning techniques, including gradient-boosted trees, recurrent neural networks, and Transformer-based sequence models, to build a binary classification model that predicts whether an online payment transaction is fraudulent or legitimate. This will help payment companies address payment fraud losses, manual-review workload, customer friction, and the need to detect suspicious transactions accurately and quickly.

The project will use the publicly available IEEE-CIS Fraud Detection dataset, which contains approximately 590,000 transactions and more than 400 transaction, card, identity, device, and behavioral variables. The team will investigate whether information from a transaction’s temporal and behavioral context improves fraud detection compared with models that treat each transaction independently.

### Success Criteria

The primary evaluation metrics will be:

- Area under the precision-recall curve, or AUPRC
- Fraud-class recall
- Fraud-class precision
- Fraud-class F1-score
- False-positive rate
- Recall at a fixed review rate or fixed false-positive rate

ROC-AUC may be reported as a secondary metric, but AUPRC will be emphasized because it is more informative for highly imbalanced classification.

A successful outcome by December would include:

1. A reproducible data-processing and modelling pipeline.
2. A strong conventional baseline model.
3. At least one temporal or deep-learning model evaluated using a chronological split.
4. Evidence showing whether temporal context improves fraud detection over static transaction features.
5. An interpretable fraud-risk output that identifies the factors contributing to high-risk predictions.
6. A demonstration prototype capable of scoring sample transactions.
7. Clear documentation of model limitations, data leakage risks, and possible deployment considerations.
   
### Project Milestones
Use these milestones to guide your work. Your team will create a GitHub Projects board to track tasks within each milestone.

| Month | Milestone | Key Activities |
|---|---|---|
| September | Data understanding, preparation, and baseline development | Review the IEEE-CIS dataset, relevant research papers, and existing fraud-detection approaches. Define the binary target, business objective, assumptions, and evaluation framework. Clean and merge transaction and identity data. Analyze missing values, categorical variables, feature distributions, temporal patterns, and class imbalance. Create a chronological training, validation, and testing strategy.<br><br>Develop initial baseline models, such as logistic regression, random forest, LightGBM, or XGBoost. Establish baseline results using fraud-focused evaluation metrics. |
| October | Temporal feature engineering and deep-learning development | Create behavioral and temporal features using only information available before each transaction. Examples may include transaction velocity, time since the previous transaction, recent transaction amount statistics, device usage patterns, and changes in customer behavior. Construct transaction sequences where reliable entity or behavioral groupings are available. Develop and compare deep-learning models such as a multilayer perceptron, GRU, LSTM, CNN-GRU, or Transformer.<br><br>Investigate methods for handling class imbalance, including class weighting and focal loss. Perform feature-selection and hyperparameter experiments. Compare static transaction models with temporal or sequential models. |
| November | Evaluation, explainability, and prototype delivery | Finalize model selection using the chronological validation and test sets. Evaluate model performance across fraud recall, precision, false-positive rate, and operational review volume. Analyze errors and identify fraud cases missed by the model. Apply explainability techniques such as SHAP to identify important transaction and behavioral factors. Test model robustness across different time periods and transaction segments. Build a lightweight dashboard or API prototype that returns a fraud-risk score and supporting explanations. Prepare final technical documentation, presentation, and recommendations for future development. |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset
**Name and Source:** IEEE-CIS Fraud Detection Dataset (Kaggle)  
**Format:** CSV/TSV  
**Size:** 5gb to 10gb  
**Location:** https://www.kaggle.com/c/ieee-fraud-detection/data

### Key Details
- [Brief description of what's in the data]
- [Any known limitations or preprocessing needed]
- [Link to data dictionary or documentation, if available]
  
---
## 🛠️ Suggested Approach

**ML Problem Type:** [e.g., Classification, Regression, NLP, Computer Vision, LLM/RAG]

**Recommended Libraries:**
- [e.g., pandas, scikit-learn, TensorFlow, Hugging Face]

**Evaluation Metrics:**
- [e.g., Accuracy, Precision/Recall, RMSE, BLEU score]

---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- [e.g., Link to an article or blog post about the problem domain]
- [e.g., Link to an industry report or case study]

**Technical Tutorials:**
- [e.g., Link to a free tutorial on the ML technique(s) involved]
- [e.g., Link to documentation for a key library or tool]

**Code Examples:**
- [e.g., Link to a relevant GitHub repo]
- [e.g., Link to a sample implementation or starter code]

**Other:**
- [Links to any additional resources — e.g., papers, videos, podcasts, etc.]

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* [e.g., Your team's channel within Break Through Tech’s Discord space]
* [e.g., Email; please copy your teammates and AI Studio Coach]
* [e.g., Request a team check-in on Zoom]
* [Note: I will aim to respond within 48 hours. Please reach out to your AI Studio Coach with urgent questions.]

> 💡 **Challenge Advisor: Please update the above based on your availability and preference. If you are not able to answer questions or meet with fellows outside of the biweekly Lab Section check-ins, simply write in "N/A (only available during the official check-in times)"**

**Recommended free coding / collaboration tools**
* […]
* […]

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I’m excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
