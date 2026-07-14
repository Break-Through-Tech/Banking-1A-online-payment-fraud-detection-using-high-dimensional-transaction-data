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
**Program:** Break Through Tech AI Studio - Fall 2026  

---

## 🏢 About Mastercard
Mastercard is a global leader in the payment technology industry, connecting billions of consumers, financial institutions, and merchants through innovative digital transaction solutions. The team objective is to leverage advanced analytics and machine learning to safeguard the global financial ecosystem by proactively identifying and mitigating sophisticated online payment fraud.

---

## 🎯 The Challenge
### Project Summary
This project tasks students with developing a binary classification model to accurately distinguish between fraudulent and legitimate online transactions using high-dimensional data. By applying machine learning and deep learning techniques to the IEEE-CIS dataset, the team will aim to reduce financial losses and minimize customer friction in real-time. The final solution will demonstrate how data-driven insights can streamline manual review workloads for payment processors.

### Success Criteria
AUPRC (primary), Fraud-class recall, Fraud-class precision, Fraud-class F1-score, False-positive rate, Recall at fixed false-positive rate, and ROC-AUC.

### Project Milestones
Use these milestones to guide your work. Your team will create a GitHub Projects board to track tasks within each milestone.
| Month | Milestone | Key Activities |
|-------|-----------|----------------|
| **September** | Data Exploration & Preprocessing | Conducting exploratory data analysis (EDA), cleaning high-dimensional variables, and identifying anomalies/outliers. |
| **October** | Feature Engineering & Baseline Modeling | Engineering transaction-based features and deploying standard baselines like Logistic Regression and Random Forest. |
| **November** | Model Optimization & Evaluation | Tuning hyperparameters, implementing deep learning architectures, and conducting cross-validation assessments. |
| **December** | Insights, Deliverables & Presentation | Developing interpretable model outputs, building a final prototype dashboard, and packaging the technical documentation. |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset
**Name and Source:** IEEE-CIS Fraud Detection Dataset (Kaggle)  
**Format:** CSV/TSV  
**Size:** 5gb to 10gb  
**Location:** Accessible via Kaggle API or direct download.  

### Key Details
- Publicly available IEEE-CIS Fraud Detection dataset from Kaggle. It contains approximately 590,000 transactions and 400+ variables (card, identity, device, etc.). Data is Numerical/Quantitative and Time Series in CSV/TSV format.
- Teams must address extreme class imbalance and handle high-cardinality categorical variables while accounting for temporal data drift in the transaction logs.

---

## 🛠️ Suggested Approach
**ML Problem Type:** Classification  
**Recommended Libraries:**
- Gradient-boosted trees, recurrent neural networks, Transformer-based sequence models, logistic regression, random forest, LightGBM, XGBoost, multilayer perceptron, GRU, LSTM, CNN-GRU, SHAP
**Evaluation Metrics:** The primary metric is AUPRC, with secondary focus on Fraud-class recall and F1-score to balance false-positive rates effectively.

---

## 📚 Resources to Get Started
The following resources will help your team understand the problem space and potential technical approaches for this project:
**Background Reading:**
- Documentation on fraud detection systems and industry standards for PCI compliance.
**Technical Tutorials:**
- Tutorials on handling high-dimensional time-series data and implementing SHAP for interpretability.
**Code Examples:**
- Sample Jupyter notebooks demonstrating baseline classification pipelines on tabular transaction data.

---

## 🤝 How We'll Work Together
**Check-ins:** During our biweekly 60-min AI Studio Lab Section meeting block (2nd and 4th week of every month)  
**Communication:** Email and scheduled Slack channels.  
**Response time:** 48 hours for non-urgent technical queries.  
**Recommended Tools:**
- **Coding:** Google Colab Free Tier  
- **Collaboration:** GitHub, Notion  
- **Virtual Meetings:** Zoom, Google Meet  

---

## 🚀 Getting Started
1. **Review this overview document** and note any questions for our first meeting.
2. **Begin reviewing the dataset** using the link provided in the Dataset section.
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects).

I'm excited to work with you!

---

## ❓ Questions?
Please bring any questions to our first meeting during the week of August 24th (Break Through Tech's Bridge to Studio - Session B).
