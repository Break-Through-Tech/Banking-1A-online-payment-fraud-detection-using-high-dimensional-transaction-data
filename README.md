# AI Studio Challenge Project Title

> 💡 **Note for the team:** This is just a template. Update the above title with your AI Studio Challenge Project name. Remove all guidance notes and example text in this template and populate this README with your own content. You can work on this README throughout AI Studio, and get feedback from your AI Studio Coach and Challenge Advisor before finalizing it.  

---

### 👥 **Team Members**

**Example:**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Taylor Nguyen    | @taylornguyen | Data exploration, visualization, overall project coordination            |
| Jordan Ramirez   | @jramirez     | Data collection, exploratory data analysis (EDA), dataset documentation  |
| Amina Hassan     | @aminahassan  | Data preprocessing, feature engineering, data validation                 |
| Priya Mehta      | @pmehta       | Model selection, hyperparameter tuning, model training and optimization  |
| Chris Park       | @chrispark    | Model evaluation, performance analysis, results interpretation           |

---

## 🎯 **Project Highlights**

**Example:**

- Developed a machine learning model using `[model type/technique]` to address `[challenge project task]`.
- Achieved `[key metric or result]`, demonstrating `[value or impact]` for `[host company]`.
- Generated actionable insights to inform business decisions at `[host company or stakeholders]`.
- Implemented `[specific methodology]` to address industry constraints or expectations.

---

## 👩🏽‍💻 **Setup and Installation**

**Provide step-by-step instructions so someone else can run your code and reproduce your results. Depending on your setup, include:**

* How to clone the repository
* How to install dependencies
* How to set up the environment
* How to access the dataset(s)
* How to run the notebook or scripts

## Setup
1. Clone the repo and open the folder in VS Code.
2. Create and activate a venv: `python -m venv .venv`, then `.venv\Scripts\Activate.ps1` (Mac/Linux: `source .venv/bin/activate`).
3. `pip install -r requirements.txt` then `pip install -e .`
4. Set `KAGGLE_API_TOKEN` (see `.env.example`). Python version used: 3.13.5

## Repo layout
- `scripts/` runnable programs (data download, dataset build, split)
- `src/fraud/` shared importable code
- `splits/` frozen split ID files (never regenerate)
- `results/` experiment registry
- `notebooks/<name>/` personal exploration
- `tests/` automated checks
- `docs/` data documentation

## Team Ettique
- No data in the repo. The Parquet file will live in the shared Drive folder.
- Work on a branch, open a PR, one teammate review before merging.
- Shared code lives in `src/`, not notebooks.

---


## Getting the data
1. Accept the competition rules: https://www.kaggle.com/competitions/ieee-fraud-detection/rules
2. Generate a token at https://www.kaggle.com/settings/api and set it as KAGGLE_API_TOKEN
   (Windows: `setx KAGGLE_API_TOKEN "your_token"`, then fully restart VS Code).
3. Run `python scripts/setup_data.py`

Files land in `data/raw/` (ignored by Git). The script stops with an ERROR if anything is
missing or does not match `scripts/data_manifest.json`.

**Colab:** add the token in the Secrets panel (key icon) as KAGGLE_API_TOKEN, then run:
    import os
    from google.colab import userdata
    os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")

## 🏗️ **Project Overview**

**Describe:**

- How this project is connected to the Break Through Tech AI Program
- Your AI Studio host company and the project objective and scope
- The real-world significance of the problem and the potential impact of your work

---

## 📊 **Data Exploration**

**You might consider describing the following (as applicable):**

* The dataset(s) used: origin, format, size, type of data
* Data exploration and preprocessing approaches
* Insights from your Exploratory Data Analysis (EDA)
* Challenges and assumptions when working with the dataset(s)

**Potential visualizations to include:**

* Plots, charts, heatmaps, feature visualizations, sample dataset images

---

## 🧠 **Model Development**

**You might consider describing the following (as applicable):**

* Model(s) used (e.g., CNN with transfer learning, regression models)
* Feature selection and Hyperparameter tuning strategies
* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)


---

## 📈 **Results & Key Findings**

**You might consider describing the following (as applicable):**

* Performance metrics (e.g., Accuracy, F1 score, RMSE)
* How your model performed
* Insights from evaluating model fairness

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## 🚀 **Next Steps**

**You might consider addressing the following (as applicable):**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

---

## 📝 **License**

Specify how your project can be used by others. Choose an appropriate license and link it here (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type. 

**Example:**
This project is licensed under the MIT License.

---

## 📄 **References** (Optional but encouraged)

Cite relevant papers, articles, or resources that supported your project.

---

## 🙏 **Acknowledgements** (Optional but encouraged)

Thank your Challenge Advisor, host company representatives, TA, and others who supported your project.
