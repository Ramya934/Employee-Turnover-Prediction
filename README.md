# Employee-Turnover-Prediction

This project aims to predict employee turnover (churn) in a company using historical employee data. The model identifies employees likely to leave based on various features, enabling proactive strategies for employee retention. This predictive model is built using Random Forest and Logistic Regression algorithms, with a focus on achieving high accuracy and interpretability.

**Dataset**

The dataset contains several features that can influence an employee's decision to stay or leave, including:

  satisfaction_level: Employee satisfaction score (0 to 1)
  
  last_evaluation: Most recent evaluation score (0 to 1)
  
  number_project: Number of projects the employee has worked on
  
  average_monthly_hours: Average monthly hours worked
  
  time_spend_company: Years the employee has spent in the company
  
  Work_accident: Whether the employee has had a work accident (0 = No, 1 = Yes)
  
  promotion_last_5years: Whether the employee was promoted in the last 5 years (0 = No, 1 = Yes)
  
  sales: Department of the employee (e.g., sales, technical)
  
  salary: Employee's salary level (low, medium, high)
  
  left: Target variable indicating if the employee left the company (0 = No, 1 = Yes)

**Approach**

**Data Preprocessing:**

Handle missing values if any (not shown in the example but can be applied as needed).

Encode categorical variables (sales, salary) using label encoding to convert them into numerical format.

Standardize the dataset for optimal model performance.

**Model Selection:**

Logistic Regression: Chosen for its simplicity and interpretability.

Random Forest: Chosen for its robustness and ability to capture non-linear relationships.

**Training and Evaluation:**

Split the dataset into training and testing sets (80-20 split).

Train the models on the training data.

Evaluate the models on the test data using metrics like accuracy, precision, recall, and F1-score.

**Results:**

Analyze the model's performance and interpret the results.

Use evaluation metrics to compare the effectiveness of the two models.  
