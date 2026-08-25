README
E-Learning Student Performance Prediction

A Machine Learning project based on the Open University Learning Analytics Dataset (OULAD). The project explores student performance prediction through two different Machine Learning approaches:

Milestone 1: Regression — Predicting student assessment scores. Milestone 2: Classification — Predicting student assessment performance classes. 📊 Dataset

This project uses the Open University Learning Analytics Dataset (OULAD), a well-known educational dataset containing information about students, courses, assessments, registrations, and Virtual Learning Environment (VLE) interactions.

Main Dataset Files assessments.csv courses.csv studentInfo.csv studentRegistration.csv StudentAssesments.csv studentVle.csv vle.csv

The datasets are merged using student, course, presentation, assessment, and VLE identifiers to construct a unified dataset.

VLE interactions are aggregated to calculate the total number of clicks made by each student:

total_clicks

The resulting merged dataset is saved as:

final_merged_dataset.csv 🏗️ Project Structure ML project/ │ ├── milestone 1/ │ ├── E-Learning Student Perfromance Prediction/ │ │ ├── OULAD.names │ │ ├── StudentAssesments.csv │ │ ├── assessments.csv │ │ ├── courses.csv │ │ ├── studentInfo.csv │ │ ├── studentRegistration.csv │ │ ├── studentVle.csv │ │ └── vle.csv │ │ │ └── E-Learning project.py │ ├── milestone 2/ │ └── Milestone 2 Updated Datasets/ │ └── Datasets train splits/ │ └── E-Learning Student Perfromance Prediction/ │ ├── OULAD.names │ ├── StudentAssesments.csv │ ├── assessments.csv │ ├── courses.csv │ ├── e-learning project.py │ ├── final_merged_dataset.csv │ ├── studentInfo.csv │ ├── studentRegistration.csv │ ├── studentVle.csv │ └── vle.csv │ └── README.md 🔹 Milestone 1 — Regression 🎯 Objective

The first milestone treats student performance as a regression problem.

The goal is to predict the numerical assessment score of a student based on academic, demographic, enrollment, assessment, and VLE activity information.

Target Variable score 🔄 Data Preparation

The following OULAD datasets were merged:

Student assessment data Assessment information Student demographic information Student registration information Course information VLE activity information

VLE data was aggregated by:

id_student code_module code_presentation

to calculate:

total_clicks

The resulting datasets were merged into a final unified dataset.

🧹 Data Preprocessing Missing Values

Missing values were investigated using:

data.isnull().sum()

Missing total_clicks values were replaced using the median, since the feature was strongly skewed.

Outlier Treatment

Numerical features were processed using Z-score based outlier capping.

Values outside:

mean ± 3 × standard deviation

were capped rather than removing the corresponding observations.

🔤 Categorical Encoding

Different encoding methods were used depending on the nature of the feature.

Binary Encoding

Binary features such as:

gender disability is_banked

were converted into numerical values.

Ordinal Encoding

Ordered categorical variables such as:

highest_education imd_band age_band

were encoded according to their natural order.

Assessment Type

Assessment types were mapped as:

CMA → 0 TMA → 1 Exam → 2 One-Hot Encoding

The region feature was transformed using OneHotEncoder.

🧠 Feature Engineering

Several features were engineered to capture student engagement, enrollment behavior, submission behavior, and historical performance.

Enrollment & Timing duration did_withdraw reg_time late_registration days_before_assessment days_until_dropout

These features describe how long students remained enrolled and their registration and assessment timing.

Submission Behavior is_late days_late

These features capture late submission behavior.

VLE Engagement click_intensity active_clicks_total active_click_ratio

click_intensity represents the student's VLE activity relative to their enrollment duration.

active_clicks_total measures activity on selected active learning materials such as quizzes, forums, and course content.

active_click_ratio represents the proportion of active clicks relative to total VLE activity.

Historical Performance prev_scores_avg score_momentum

prev_scores_avg represents the average of previous assessment scores.

The calculation uses shifted values so that the current assessment score is not directly included in its own historical feature.

score_momentum captures changes in the student's recent performance.

Workload & Risk credit_load_intensity at_risk_flag

These features attempt to capture study workload and potential risk based on registration timing and engagement.

🔎 Feature Selection

Correlation analysis was performed using a correlation matrix and heatmap.

Features with weak relationships to the target were removed, as well as features that were highly correlated with other variables.

Important features investigated included:

prev_scores_avg active_clicks_total active_click_ratio total_clicks weight click_intensity

The final dataset was then prepared for model training.

🤖 Regression Models

Six regression algorithms were implemented and compared.

Linear Regression
Used as a baseline model for establishing a simple linear relationship between the features and assessment score.

Ridge Regression
A regularized linear regression model using L2 regularization.

Random Forest Regressor
The final configuration used:

n_estimators = 600 max_depth = 8 min_samples_split = 10 min_samples_leaf = 5 max_features = "sqrt" bootstrap = True 4. XGBoost Regressor

The final configuration used:

n_estimators = 400 learning_rate = 0.05 max_depth = 4 subsample = 0.85 colsample_bytree = 0.85 min_child_weight = 3 reg_lambda = 3 reg_alpha = 0.1

GridSearchCV was also experimented with for XGBoost. However, manually selected hyperparameters performed better in the final experiments.

Gradient Boosting Regressor n_estimators = 300 learning_rate = 0.05 max_depth = 4 subsample = 0.8 max_features = "sqrt"
Decision Tree Regressor max_depth = 6 min_samples_split = 10 min_samples_leaf = 7 max_features = "sqrt" 📏 Regression Evaluation
The regression models were evaluated using:

R² Score Train R² Test R²

A model comparison plot was created to compare Train R² against Test R².

An Actual vs Predicted visualization was also generated for the best-performing model based on Test R².

🔹 Milestone 2 — Classification 🎯 Objective

The second milestone approaches the problem as a classification task.

Instead of predicting the exact numerical score, the goal is to predict the student's assessment performance class.

Target Variable assessmentClass 🔄 Data Preparation

The same OULAD data integration approach was used.

The following datasets were merged:

studentAssessments assessments studentInfo studentRegistration courses studentVle vle

VLE activity was aggregated into:

total_clicks

and incorporated into the final dataset.

🧹 Data Preprocessing

The classification pipeline included:

Missing value handling Numerical conversion Median imputation Z-score based outlier capping Binary encoding Ordinal encoding One-hot encoding Feature engineering Feature selection Standard scaling 🧠 Feature Engineering

The classification milestone reused the main feature engineering approach from Milestone 1.

Important features included:

Academic Features score weight num_of_prev_attempts VLE Engagement total_clicks click_intensity active_clicks_total active_click_ratio Enrollment Behavior did_withdraw Student Characteristics gender disability highest_education imd_band age_band

Additional engineered features were also created during experimentation, while weaker or redundant features were removed before training.

🤖 Classification Models

Four classification algorithms were implemented and compared.

Support Vector Machine kernel = linear C = 1
K-Nearest Neighbors n_neighbors = 5 algorithm = auto metric = euclidean
Logistic Regression penalty = l2 solver = lbfgs max_iter = 1000
Decision Tree criterion = gini max_depth = 5 📏 Classification Evaluation
The classification models were evaluated using:

Accuracy Precision Recall F1-score Confusion Matrix Classification Report

These metrics provide a more complete evaluation of classification performance than accuracy alone.

📌 Milestone Comparison Milestone 1 Milestone 2 Problem Type Regression Classification Target score assessmentClass Goal Predict numerical assessment score Predict assessment performance class Linear Model Linear Regression Logistic Regression Tree Model Decision Tree Regressor Decision Tree Classifier Ensemble Random Forest — Boosting XGBoost — Boosting Gradient Boosting — Distance-Based — KNN Kernel-Based — SVM Main Metrics R² Accuracy Train/Test R² Precision Recall F1-score Confusion Matrix 🔬 Machine Learning Workflow OULAD Dataset │ ▼ Data Loading │ ▼ Dataset Merging │ ▼ Exploratory Data Analysis │ ▼ Missing Value Handling │ ▼ Outlier Treatment │ ▼ Categorical Encoding │ ▼ Feature Engineering │ ▼ Feature Selection │ ▼ Train/Test Split │ ▼ Scaling │ ┌──────────┴──────────┐ ▼ ▼ Regression Classification │ │ ▼ ▼ Model Training Model Training │ │ ▼ ▼ Evaluation Evaluation 🛠️ Technologies Used Programming Language Python Data Processing Pandas NumPy Data Visualization Matplotlib Seaborn Statistical Analysis SciPy Machine Learning Scikit-learn XGBoost 📚 Machine Learning Concepts Demonstrated

This project demonstrates practical application of:

Exploratory Data Analysis Data Cleaning Missing Value Handling Outlier Treatment Categorical Encoding Feature Engineering Feature Selection Correlation Analysis Feature Scaling Train/Test Splitting Regression Classification Ensemble Learning Gradient Boosting Hyperparameter Tuning Model Comparison Model Evaluation Educational Data Mining Student Behavioral Analysis 🚀 How to Run

Clone the Repository git clone <YOUR_REPOSITORY_URL> cd "ML project"
Create a Virtual Environment python -m venv .venv Windows .venv\Scripts\activate
Install Dependencies pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost
Run Milestone 1
Navigate to the Milestone 1 directory and run:

python "E-Learning project.py" 5. Run Milestone 2

Navigate to the Milestone 2 project directory and run:

python "e-learning project.py" 📈 Results Milestone 1 — Regression

The regression models were compared primarily using Test R², while Train R² was used to analyze generalization and potential overfitting.

Milestone 2 — Classification

The classification models were evaluated using:

Accuracy Precision Recall F1-score Confusion Matrix

The exact numerical results are available from the model evaluation output generated by the respective Python scripts.

🎓 Project Goal

The overall goal of this project is to explore how Machine Learning can be applied to educational data to understand and predict student performance.

The two milestones approach the same dataset from different Machine Learning perspectives.

Regression

What numerical score is the student likely to achieve?

Classification

Which performance class does the student's assessment belong to?

Together, the two milestones demonstrate how the same educational dataset can be transformed and used to solve both regression and classification problems.

👨‍💻 Author

Bahboor

Computer Science Student | Machine Learning Enthusiast
