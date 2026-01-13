# 🤖 MACHINE LEARNING - Complete Conceptual Guide

> **A comprehensive guide covering all Machine Learning concepts from fundamentals to advanced algorithms, with mathematical intuitions, practical implementations, and real-world applications.**

---

## Table of Contents

1. [Introduction to Machine Learning](#introduction-to-machine-learning)
2. [How to Start Any ML Project](#how-to-start-any-ml-project)
3. [Types of Machine Learning](#types-of-machine-learning)
4. [5-Step Data Analyst Process](#5-step-data-analyst-process)
5. [End-to-End ML Development Life Cycle](#end-to-end-ml-development-life-cycle)
6. [Data Assessment and Quality](#data-assessment-and-quality)
7. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
8. [Feature Engineering and Selection](#feature-engineering-and-selection)
9. [Regression Algorithms](#regression-algorithms)
10. [Classification Algorithms](#classification-algorithms)
11. [Ensemble Methods](#ensemble-methods)
12. [Clustering Algorithms](#clustering-algorithms)
13. [Dimensionality Reduction](#dimensionality-reduction)
14. [Model Evaluation Metrics](#model-evaluation-metrics)
15. [Gradient Descent and Optimization](#gradient-descent-and-optimization)
16. [Time Series Analysis](#time-series-analysis)
17. [Model Deployment](#model-deployment)
18. [Quick Reference and Cheat Sheets](#quick-reference-and-cheat-sheets)

---

## Introduction to Machine Learning

Machine Learning (ML) is a subset of Artificial Intelligence that enables computers to learn from data and make predictions or decisions without being explicitly programmed for every scenario.

### What Makes ML Different from Traditional Programming?

```
TRADITIONAL PROGRAMMING:
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│      Data       │ + → │     Rules       │ = → │     Output      │
│   (Input)       │     │   (Program)     │     │   (Answer)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘

MACHINE LEARNING:
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│      Data       │ + → │     Output      │ = → │     Rules       │
│   (Input)       │     │   (Answers)     │     │   (Model)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Why Machine Learning?

| Use Case | Why ML? |
|----------|---------|
| Spam Detection | Rules are too complex to write manually |
| Image Recognition | Patterns are too subtle for explicit rules |
| Recommendation Systems | User preferences are dynamic and personal |
| Fraud Detection | Patterns constantly evolve |
| Medical Diagnosis | Combines multiple factors in complex ways |

---

## How to Start Any ML Project

### 1. Business Problem to ML Problem

**Goal:** The very first step is to clearly define the business problem you're trying to solve. This involves understanding the core issue, the desired outcome, and the impact of a successful solution on the business.

**Translation:** Once the business problem is well-defined, the next critical step is to translate it into a specific, well-posed machine learning problem. This involves determining what kind of prediction or insight is needed.

**Examples:**

| Business Problem | ML Problem | Type |
|-----------------|------------|------|
| Increase customer retention | Predict which customers are at high risk of churn | Classification |
| Improve sales revenue | Forecast future product demand | Regression |
| Automate customer support | Identify the intent of customer inquiries | NLU/Classification |
| Reduce fraud losses | Detect anomalous transactions | Anomaly Detection |
| Personalize user experience | Recommend relevant products | Recommendation System |

### 2. Type of Problem

**Goal:** Understanding the nature of the ML problem is crucial for selecting appropriate algorithms, evaluation metrics, and data preprocessing techniques.

**Categorization:** This step involves identifying whether the problem falls into categories like:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TYPES OF MACHINE LEARNING                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SUPERVISED LEARNING                                                        │
│  ├── Classification (Discrete output)                                       │
│  │   ├── Binary Classification (2 classes)                                  │
│  │   └── Multi-class Classification (>2 classes)                           │
│  │                                                                          │
│  └── Regression (Continuous output)                                         │
│      ├── Linear Regression                                                  │
│      ├── Polynomial Regression                                              │
│      └── Multiple Regression                                                │
│                                                                             │
│  UNSUPERVISED LEARNING                                                      │
│  ├── Clustering                                                             │
│  │   ├── K-Means                                                            │
│  │   ├── Hierarchical                                                       │
│  │   └── DBSCAN                                                             │
│  │                                                                          │
│  └── Dimensionality Reduction                                               │
│      ├── PCA                                                                │
│      └── t-SNE                                                              │
│                                                                             │
│  REINFORCEMENT LEARNING                                                     │
│  └── Training an agent to make decisions to maximize reward                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. Current Solution

**Goal:** Before building a new ML solution, it's essential to understand the existing methods or processes used to address the business problem.

**Analysis:** This involves:
- Documenting the current workflow
- Understanding its effectiveness
- Identifying limitations (manual effort, scalability issues, accuracy)
- Calculating associated costs

**Baseline:** The performance of the current solution often serves as a baseline against which the ML model's performance will be compared. Understanding the current solution helps justify the need for an ML-based approach and sets realistic expectations.

### 4. Getting Data

**Goal:** Machine learning models are data-driven. This step focuses on identifying, sourcing, and acquiring the necessary data to train and evaluate the model.

**Considerations:**

| Aspect | Questions to Ask |
|--------|-----------------|
| **Data Sources** | Where does the relevant data reside (databases, APIs, logs, external datasets)? |
| **Data Collection** | How will the data be collected and accessed? |
| **Data Volume** | Is there enough data? How much is needed? |
| **Data Quality** | What is the quality (missing values, inconsistencies, errors)? |
| **Data Privacy** | Are there any regulations or ethical considerations? |

**Data Summary:** Provides a high-level overview of the dataset.

**Column Details:** Provides a more in-depth look at each individual column.

### 5. Metrics to Measure

**Goal:** Defining clear and relevant metrics is crucial for evaluating the performance of the ML model and determining if it effectively solves the business problem.

**Selection:** The choice of metrics depends on the type of ML problem:

| Problem Type | Metrics |
|-------------|---------|
| Classification | Accuracy, Precision, Recall, F1-Score, AUC-ROC |
| Regression | MSE, RMSE, MAE, R-squared |
| Clustering | Silhouette Score, Davies-Bouldin Index |

**Business Alignment:** The chosen metrics should align with the business objectives. For example, in a fraud detection system, recall (minimizing missed fraud cases) might be more critical than overall accuracy.

### 6. Online vs Batch Learning

**Goal:** This step considers how the ML model will be trained and how it will make predictions in a real-world setting.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ONLINE vs BATCH LEARNING                                 │
├────────────────────────────────┬────────────────────────────────────────────┤
│         BATCH LEARNING         │           ONLINE LEARNING                  │
├────────────────────────────────┼────────────────────────────────────────────┤
│ • Trained on entire dataset    │ • Learns incrementally                     │
│ • Periodic retraining needed   │ • Adapts to new data continuously          │
│ • Stable predictions           │ • Can adapt to changing patterns           │
│ • Resource intensive training  │ • Lower memory requirements                │
│ • Good for static data         │ • Good for streaming data                  │
│                                │                                            │
│ Example: Monthly sales forecast│ Example: Stock price prediction            │
└────────────────────────────────┴────────────────────────────────────────────┘
```

**Decision Factors:**
- Volume and velocity of data
- Need for real-time predictions
- Computational resources
- Stability of underlying data patterns

### 7. Check Assumptions

**Goal:** Machine learning models often rely on certain assumptions about the data and the underlying relationships. It's crucial to explicitly identify and check these assumptions.

**Examples:**

| Model | Assumptions |
|-------|-------------|
| **Linear Regression** | Linear relationship, independence of errors, homoscedasticity, normality of errors |
| **Naive Bayes** | Feature independence given the class |
| **Logistic Regression** | Linear relationship between features and log-odds |
| **K-Means** | Spherical clusters, similar cluster sizes |
| **Time Series** | Stationarity or specific patterns |

**Importance:** Violating these assumptions can lead to poorly performing or unreliable models. This step might involve statistical tests, visualizations, and domain expertise to validate the assumptions.

---

## Types of Machine Learning

### Supervised Learning

In supervised learning, we have labeled data - meaning we know the correct output for each input. The model learns to map inputs to outputs.

```
Training Data:
┌──────────────────┬─────────────┐
│ Features (X)     │ Label (y)   │
├──────────────────┼─────────────┤
│ [2.5, 1.2, 3.0]  │ Class A     │
│ [1.8, 2.5, 1.5]  │ Class B     │
│ [3.1, 0.8, 2.8]  │ Class A     │
│ [1.2, 3.0, 1.0]  │ Class B     │
└──────────────────┴─────────────┘

Model learns: f(X) → y
```

**Types:**
- **Classification:** Predict discrete categories
- **Regression:** Predict continuous values

### Unsupervised Learning

In unsupervised learning, we only have input data without labels. The model tries to find patterns or structure in the data.

```
Training Data:
┌──────────────────┐
│ Features (X)     │     → Model finds patterns/groups
├──────────────────┤
│ [2.5, 1.2, 3.0]  │
│ [1.8, 2.5, 1.5]  │
│ [3.1, 0.8, 2.8]  │
│ [1.2, 3.0, 1.0]  │
└──────────────────┘
```

**Types:**
- **Clustering:** Group similar data points
- **Dimensionality Reduction:** Reduce number of features while preserving information
- **Association:** Find rules that describe large portions of data

### Reinforcement Learning

The model (agent) learns by interacting with an environment and receiving rewards or penalties.

```
┌─────────┐    Action    ┌─────────────┐
│  Agent  │ ───────────→ │ Environment │
│         │ ←─────────── │             │
└─────────┘  State +     └─────────────┘
             Reward
```

**Applications:** Game playing, robotics, autonomous vehicles

---

## 5-Step Data Analyst Process

### 1. Asking for the Problem (Understanding the Business Question)

**What it is:** Before jumping into any data, the first step is to **understand the problem** you're trying to solve. This is where you communicate with stakeholders (like business leaders, managers, or clients) to know exactly what they need from the data.

**Why it matters:** If you don't understand the question, you might analyze the wrong data or go down the wrong path.

**Example:** If a company asks, "What's causing a drop in sales?" you'll need to figure out what kind of data might help answer that, like sales trends, customer feedback, or inventory data.

### 2. Data Wrangling (Data Gathering, Assessing, and Cleaning)

**What it is:** Data wrangling is about **collecting, preparing, and cleaning** data for analysis. It's the behind-the-scenes work that makes sure your data is in a usable form.

#### Key Steps in Data Wrangling:

**Data Gathering:**
- Collect data from various sources like databases, spreadsheets, APIs, etc.
- Example: Getting sales data from a database and customer feedback from an Excel file.

**Data Assessing:**
- Understand the data deeply before cleaning
- Check the quality and structure of the data (looking for missing values, duplicates, outliers, etc.)
- **Types:**
  - **Dirty data:** Has quality issues like missing, corrupt, or inaccurate data
  - **Messy data:** Has tidiness/structural issues. In tidy data: each variable forms a column, each observation forms a row, each observational unit forms a table

**Data Cleaning:**
- Fix issues found in the assessing stage
- Remove or fill missing data, correct errors, or standardize formats
- Example: If "USA" is written in different ways, standardize it to one format ("United States")

### 3. Exploratory Data Analysis (EDA)

**What it is:** This step is all about **understanding the data**. You explore and analyze it to look for patterns, trends, or relationships.

#### EDA includes:

**Descriptive Statistics:**
- Check things like the mean, median, and standard deviation
- Example: Calculate the average sales per month

**Visualizations:**
- Create charts like histograms, bar graphs, and scatter plots
- Example: A bar chart of sales over time to spot any dips or spikes

**Correlation:**
- Look for relationships between variables
- Example: See if there's a link between advertising spend and sales growth

#### Why do EDA:
- Model building
- Analysis and reporting
- Validate assumptions
- Handling missing values
- Feature engineering
- Detecting outliers

### 4. Conclusions (Drawing Insights and Analysis)

**What it is:** After exploring the data, it's time to **interpret the results**. This step involves making sense of the findings and drawing actionable insights.

**Here's what happens:**

- **Answer the business question:** Relate your findings back to the original problem
  - Example: "Sales are dropping in regions with lower ad spend."

- **Find patterns:** Identify any trends, anomalies, or correlations
  - Example: Higher customer satisfaction scores correlate with repeat purchases.

### 5. Presenting in Front of Others (Communicating Results)

**What it is:** The final step is **communicating your findings** clearly to the stakeholders.

#### Tips for Effective Presentation:

**Tell a story:**
- Present your analysis like a story with a clear beginning, middle, and end
- Example: "Here's how the advertising spend affected sales, and here's what we should do moving forward."

**Visualize findings:**
- Use charts, graphs, or dashboards to make your insights easier to digest
- Example: A line graph showing how sales grew with increased ad spend

**Provide actionable recommendations:**
- Based on your analysis, suggest next steps
- Example: "I recommend increasing ad spend by 10% in underperforming regions."

> **Pro Tip:** Use tools like [Graphy](https://graphy.app/) for data storytelling - it enables anyone to become a skilled data storyteller by radically simplifying the way data is presented and communicated.

### Summary Flow:
```
1. Ask the right question → Understand the problem
2. Wrangle the data → Gather, assess, and clean it up
3. Explore the data → Perform EDA (visuals and stats)
4. Draw conclusions → Analyze findings and make sense of them
5. Present results → Share insights clearly and take action
```

---

## End-to-End ML Development Life Cycle

### 1. Frame the Problem (Understand the Business Need)

**Define Objectives:**
- Clearly articulate the business problem you're trying to solve
- Define the goals of the ML project
- Identify desired outcomes and success metrics

**Identify Stakeholders:**
- Understand who will be using or affected by the ML system
- Document their requirements

**Formulate the ML Problem:**
- Translate the business problem into a specific ML task
- Examples: classification, regression, anomaly detection

**Determine Success Metrics:**
- Define quantitative metrics for evaluation
- Ensure they align with business objectives

**Consider Constraints:**
- Budget and time limitations
- Data availability
- Performance requirements
- Interpretability needs
- Legal or ethical considerations

### 2. Gathering Data

**Identify Data Sources:**
- Determine where the necessary data resides
- Sources: databases, APIs, files, external sources

**Collect Data:**
- Acquire data from identified sources
- Methods: scraping, downloading, database access, data pipelines

**Understand the Data:**
- Explore structure, format, size, and potential biases
- Document data sources and collection methods

**Data Governance:**
- Consider data privacy and security
- Ensure compliance with regulations

### 3. Data Preprocessing

**Data Cleaning:**
- Handle missing values (imputation or removal)
- Identify and treat outliers
- Correct inconsistencies
- Remove duplicates

**Data Transformation:**
- Scale numerical features
- Encode categorical variables
- Handle dates and times
- Create new features from existing ones

**Data Splitting:**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA SPLITTING                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Total Dataset (100%)                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Training Set (70%)  │ Validation Set (15%) │ Test Set (15%)       │   │
│  │                      │                      │                       │   │
│  │  Train the model     │ Tune hyperparameters │ Final evaluation     │   │
│  │                      │                      │                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Data Augmentation (Optional):**
- For image or text classification
- Generate synthetic data to increase training set size and diversity

### 4. Exploratory Data Analysis (EDA)

**Visualize Data:**
- Use charts, graphs, and visual techniques
- Understand data characteristics
- Identify patterns and relationships

**Statistical Analysis:**
- Calculate descriptive statistics
- Perform statistical tests
- Gain insights into the data

**Identify Potential Features:**
- Explore which features might be most relevant
- Document findings

**Uncover Data Quality Issues:**
- Identify remaining problems
- Address before modeling

### 5. Feature Engineering and Selection

**Feature Engineering:**
- Create new features from existing ones
- Use domain knowledge and EDA insights
- Techniques: combining features, polynomial features, transformations

**Feature Scaling:**
- Standardization: (x - mean) / std
- Normalization: (x - min) / (max - min)

**Feature Selection:**
- Identify and select most relevant features
- Reduce dimensionality
- Improve performance and interpretability

---

## Data Assessment and Quality

### Types of Assessment in Data Analytics

#### 1. Manual Assessment

**Description:** Done by visually inspecting data.

**Tools:** Excel, Google Sheets, raw text files

**Often used for:**
- Small datasets
- Quick checks
- Spot-checking data quality

**Examples:**
- Scanning a CSV file for missing values
- Using filters in Excel to look for outliers
- Manually documenting data column meanings

#### 2. Programmatic Assessment

**Description:** Uses code or automated tools to assess and profile data.

**Tools:** Python (pandas, pydeequ), R, SQL, data profiling tools, Power BI, Tableau

**Benefits:** Scalable and more accurate for large datasets

**Examples:**
```python
# Python example for programmatic assessment
import pandas as pd

# Check for missing values
df.isnull().sum()

# Check data types
df.dtypes

# Check for duplicates
df.duplicated().sum()

# Basic statistics
df.describe()
```

### Common Steps in Data Assessment Process

#### 1. Discover
**Goal:** Understand where the data is coming from and what data exists.

**Actions:**
- Understand data sources (source systems, APIs, databases)
- Identify what data exists and what's missing

#### 2. Document
**Goal:** Record metadata and create data dictionaries.

**Actions:**
- Record metadata: column names, data types, units, value ranges, business definitions
- Create data dictionaries or schema documentation

#### 3. Assess (Profile & Quality Check)
**Goal:** Run data profiling and check for data quality issues.

**Actions:**
- Run data profiling: Count of nulls, uniques, data types
- Calculate descriptive statistics (mean, median, mode)
- Check for: Missing values, Duplicates, Outliers, Inconsistencies

#### 4. Validate
**Goal:** Cross-check data against expected values and business rules.

**Actions:**
- Cross-check data with source systems or expected values
- Confirm that the data matches the business rules

#### 5. Report
**Goal:** Summarize findings and suggest improvements.

**Actions:**
- Summarize issues, observations, and risks
- Suggest fixes or improvements
- Present findings visually

### Data Quality Dimensions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DATA QUALITY DIMENSIONS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. COMPLETENESS → Is data missing?                                         │
│     Example: Phone numbers missing in 40% of rows                           │
│     Think of it like a pizza with missing slices 🍕                         │
│                                                                             │
│  2. VALIDITY → Is data invalid?                                             │
│     Example: Negative height, "banana" in date field                        │
│     Like putting a square peg in a round hole                               │
│                                                                             │
│  3. ACCURACY → Is data correct in the real world?                           │
│     Example: Salary says $5 instead of $50,000                              │
│     Like giving someone the wrong map                                       │
│                                                                             │
│  4. CONSISTENCY → Does data match across systems?                           │
│     Example: "USA" vs "United States" vs "U.S."                             │
│     Like telling two different stories                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Order of Severity

```
Completeness ← Validity ← Accuracy ← Consistency
(Most severe)                        (Least severe)
```

### Data Cleaning Order

1. Quality → Completeness
2. Tidiness
3. Quality → Validity
4. Quality → Accuracy
5. Quality → Consistency

> **Note:** This is the recommended pattern that most DA researchers and experts follow. First assign labels to each issue using quality dimensions, then clean following this order.

### Steps Involved in Data Cleaning

1. **Define** → Define the solution of each issue
   - Example: If column has "NYC" and "New York City", write a dictionary to map values

2. **Code** → Write the code for that solution

3. **Test** → Test the code and result

> **Important:** Always make sure to create a copy of your pandas dataframe before you start the cleaning process!

---

## Exploratory Data Analysis (EDA)

### Why Do EDA?

- Model building
- Analysis and reporting
- Validate assumptions
- Handling missing values
- Feature engineering
- Detecting outliers

### Column Type Classification

For convenience to perform EDA, assign each column with these labels:

| Type | Examples |
|------|----------|
| **Numerical** | Age, Fare, PassengerId |
| **Categorical** | Survived, Pclass, Sex, SibSp, Parch, Embarked |
| **Mixed** | Name, Ticket, Cabin |

> **Pro Tip:** Try to change the dtype of columns if possible as it reduces memory usage.

### Univariate Analysis

Univariate analysis focuses on analyzing each feature in the dataset independently.

**Distribution Analysis:**
- Examine the shape, central tendency, and dispersion of each feature

**Identifying Potential Issues:**
- Outliers, skewness, missing values

#### Common Distribution Shapes

| Shape | Description |
|-------|-------------|
| **Normal Distribution** | Symmetrical, bell-shaped; mean = median = mode |
| **Skewed Distribution** | Not symmetrical; one tail longer than other |
| **Bimodal Distribution** | Two peaks or modes |
| **Uniform Distribution** | All values have equal chance of occurring |

#### Measures of Dispersion

| Measure | Description |
|---------|-------------|
| **Range** | Difference between largest and smallest values |
| **Variance** | Average of squared deviations from mean |
| **Standard Deviation** | Square root of variance (same units as data) |
| **IQR** | Range between 25th and 75th percentile |

### Steps for Univariate Analysis on Numerical Columns

1. **Descriptive Statistics:** Compute mean, median, mode, standard deviation, range, quartiles

2. **Visualizations:** Histograms, box plots, density plots

3. **Identifying Outliers:** Use visualizations to identify outliers; determine if they're errors or legitimate

4. **Skewness:** Check for skewness; consider transforming data if necessary

5. **Conclusion:** Summarize findings and make decisions

### Steps for Univariate Analysis on Categorical Columns

1. **Descriptive Statistics:** Compute frequency distribution of categories

2. **Visualizations:** Count plots and pie charts

3. **Missing Values:** Check for missing values and decide how to handle them

4. **Conclusion:** Summarize findings

### Bivariate Analysis

**Step 1:** Select 2 columns

**Step 2:** Understand type of relationship

#### 1. Numerical - Numerical

```python
# Visualizations
- Scatterplot (regression plots)
- 2D histplot
- 2D KDEplots

# Check correlation coefficient for linear relationship
df[['col1', 'col2']].corr()
```

#### 2. Numerical - Categorical

```python
# Create visualizations comparing numerical distribution across categories
- Barplot
- Boxplot
- KDEplot
- Violinplot
- Scatterplots
```

#### 3. Categorical - Categorical

```python
# Create cross-tabulations or contingency tables
pd.crosstab(df['cat1'], df['cat2'])

# Visualizations
- Heatmap
- Stacked barplots
- Treemaps
```

**Step 3:** Write your conclusions

---

## Feature Engineering and Selection

### Understanding Variance, Covariance, and PCA

#### 1. Variance – Looking at a Single Feature (One Axis)

When we look at a **single feature** (one axis), **variance** tells us how much the data is spread out along that axis.

- If the data is really spread out, that feature likely holds **a lot of information**
- A highly variable feature can help us **distinguish between different data points**

👉 For selecting important individual features, a **high variance is often a good sign**.

Features with very low variance (almost constant) might be dropped — because they don't differentiate data points well.

#### 2. Covariance – Looking at Two Features (Two Axes)

When we consider **two features** (two axes), **covariance** comes into play.

- It tells us **how those two features change together**
- If the data spreads out **diagonally across the two axes**, the features are **related or correlated**

> If we had to pick just **one of these related features**, we'd look at their **variances** to decide **which one captures more of the overall spread**.

#### 3. PCA – Scaling Up to Many Features

**PCA (Principal Component Analysis)** handles **multiple features (dimensions)**.

It tries to find a **new set of axes**, called **principal components**, that align with the directions of **maximum variance** in the data.

#### 4. Principal Components – The New Directions

- The **first principal component** is the **single direction** that captures the **most variance**
- The **second principal component** is perpendicular to the first and captures the next most variance
- And so on...

```
Original Features:                    After PCA:
    y│                                   PC2│
     │    * *                              │    *  *
     │  *   *  *                           │  *     *
     │ *  *    *                      ─────┼───*──*───────→ PC1
     │* *   *                              │  *  *
     └───────────→ x                       │*    *
                                           
Features are correlated              Components are uncorrelated
```

### Feature Engineering Techniques

```python
# 1. Creating new features
df['age_squared'] = df['age'] ** 2
df['income_per_age'] = df['income'] / df['age']

# 2. Binning continuous variables
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100], 
                          labels=['Young', 'Adult', 'Middle', 'Senior'])

# 3. One-hot encoding
df = pd.get_dummies(df, columns=['category'])

# 4. Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])

# 5. Feature scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization
scaler = StandardScaler()
df['feature_scaled'] = scaler.fit_transform(df[['feature']])

# Normalization
normalizer = MinMaxScaler()
df['feature_normalized'] = normalizer.fit_transform(df[['feature']])
```

### Feature Selection Techniques

```python
# 1. Variance Threshold
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.1)
X_selected = selector.fit_transform(X)

# 2. Correlation-based selection
correlation_matrix = df.corr()
# Remove highly correlated features (> 0.95)

# 3. SelectKBest
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# 4. Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
selector = RFE(RandomForestClassifier(), n_features_to_select=10)
X_selected = selector.fit_transform(X, y)

# 5. Feature Importance from Tree-based models
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)
importances = model.feature_importances_
```

---

## Regression Algorithms

### Linear Regression

**What it does:** Finds a linear relationship between input features and a continuous output.

**Mathematical Intuition:**

```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

Where:
- ŷ = predicted value
- β₀ = intercept (bias)
- β₁, β₂, ..., βₙ = coefficients (weights)
- x₁, x₂, ..., xₙ = features
```

**Goal:** Minimize the Sum of Squared Errors (SSE):

```
SSE = Σ(yᵢ - ŷᵢ)²
```

**Geometric Intuition:**
- Finding the best-fit line (or hyperplane in higher dimensions)
- The line that minimizes the vertical distances from all points

**Implementation:**

```python
from sklearn.linear_model import LinearRegression

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Get coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
```

**Assumptions:**
1. Linear relationship between X and y
2. Independence of errors
3. Homoscedasticity (constant variance of errors)
4. Normality of errors

**Pros and Cons:**

| Pros | Cons |
|------|------|
| Simple and interpretable | Assumes linear relationship |
| Fast to train | Sensitive to outliers |
| Works well with small data | Cannot capture complex patterns |

### Polynomial Regression

**What it does:** Extends linear regression by adding polynomial terms to capture non-linear relationships.

```
ŷ = β₀ + β₁x + β₂x² + β₃x³ + ...
```

**Implementation:**

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Create polynomial features and fit
model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])
model.fit(X_train, y_train)
```

### Ridge Regression (L2 Regularization)

**What it does:** Linear regression with L2 penalty to prevent overfitting.

```
Loss = SSE + λ × Σβᵢ²

Where λ is the regularization strength
```

**Implementation:**

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)  # alpha = λ
model.fit(X_train, y_train)
```

### Lasso Regression (L1 Regularization)

**What it does:** Linear regression with L1 penalty - can perform feature selection by setting some coefficients to exactly zero.

```
Loss = SSE + λ × Σ|βᵢ|
```

**Implementation:**

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=1.0)
model.fit(X_train, y_train)
```

### Elastic Net

**What it does:** Combines L1 and L2 regularization.

```
Loss = SSE + λ₁ × Σ|βᵢ| + λ₂ × Σβᵢ²
```

**Implementation:**

```python
from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=1.0, l1_ratio=0.5)
model.fit(X_train, y_train)
```

---

## Classification Algorithms

### Logistic Regression

**What it does:** Despite its name, it's used for classification. Uses the sigmoid function to output probabilities.

**Mathematical Intuition:**

```
P(y=1|X) = 1 / (1 + e^(-(β₀ + β₁x₁ + ... + βₙxₙ)))

Or equivalently:
P(y=1|X) = σ(z) where z = β₀ + β₁x₁ + ... + βₙxₙ
```

**Sigmoid Function:**

```
        1 │      ___________
          │     /
      0.5 │    /
          │   /
        0 ┼──/─────────────
           -6    0    6
```

**Implementation:**

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# Predict classes
predictions = model.predict(X_test)

# Predict probabilities
probabilities = model.predict_proba(X_test)
```

### Decision Trees

**What it does:** Creates a tree-like model of decisions based on feature values.

**How it works:**
1. Select the best feature to split on (using Gini impurity or Information Gain)
2. Split the data based on that feature
3. Repeat recursively until stopping criteria met

**Gini Impurity:**
```
Gini = 1 - Σpᵢ²

Where pᵢ = proportion of class i in the node
```

**Information Gain:**
```
IG = Entropy(parent) - Σ(weighted average of Entropy(children))

Entropy = -Σpᵢ × log₂(pᵢ)
```

**Implementation:**

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
model.fit(X_train, y_train)

# Visualize the tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=feature_names)
plt.show()
```

**Pros and Cons:**

| Pros | Cons |
|------|------|
| Easy to interpret | Prone to overfitting |
| Handles non-linear relationships | Sensitive to small changes in data |
| No feature scaling needed | Can create biased trees with imbalanced data |

### Naive Bayes

**What it does:** Probabilistic classifier based on Bayes' theorem with independence assumption.

**Bayes' Theorem:**
```
P(y|X) = P(X|y) × P(y) / P(X)

Where:
- P(y|X) = Probability of class y given features X (posterior)
- P(X|y) = Probability of features X given class y (likelihood)
- P(y) = Prior probability of class y
- P(X) = Evidence
```

**"Naive" Assumption:**
Features are conditionally independent given the class.

**Implementation:**

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# For continuous features
model = GaussianNB()
model.fit(X_train, y_train)

# For discrete/count features (e.g., text classification)
model = MultinomialNB()
model.fit(X_train, y_train)
```

### Support Vector Machines (SVM)

**What it does:** Finds the hyperplane that best separates classes with maximum margin.

**Key Concepts:**
- **Hyperplane:** Decision boundary that separates classes
- **Support Vectors:** Data points closest to the hyperplane
- **Margin:** Distance between hyperplane and nearest support vectors
- **Kernel Trick:** Transform data to higher dimensions for non-linear separation

**Kernel Types:**

| Kernel | Use Case |
|--------|----------|
| Linear | Linearly separable data |
| RBF (Gaussian) | Most common, general purpose |
| Polynomial | When polynomial relationship exists |
| Sigmoid | Similar to neural networks |

**Implementation:**

```python
from sklearn.svm import SVC

# Linear kernel
model = SVC(kernel='linear', C=1.0)

# RBF kernel
model = SVC(kernel='rbf', C=1.0, gamma='scale')

model.fit(X_train, y_train)
```

### K-Nearest Neighbors (KNN)

**What it does:** Classifies based on the majority class of k nearest neighbors.

**How it works:**
1. Calculate distance from new point to all training points
2. Select k nearest neighbors
3. Assign class based on majority vote

**Distance Metrics:**

| Metric | Formula |
|--------|---------|
| Euclidean | √Σ(xᵢ - yᵢ)² |
| Manhattan | Σ|xᵢ - yᵢ| |
| Minkowski | (Σ|xᵢ - yᵢ|ᵖ)^(1/p) |

**Implementation:**

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
model.fit(X_train, y_train)
```

**Choosing k:**
- Small k: More sensitive to noise, complex boundary
- Large k: Smoother boundary, may miss local patterns
- Rule of thumb: k = √n (where n is number of samples)

---

## Ensemble Methods

### Why Ensemble Methods?

Single models have limitations:
- Decision trees overfit
- Linear models underfit complex patterns
- Each model has bias

**Solution:** Combine multiple models to get better predictions!

### Bagging (Bootstrap Aggregating)

**Concept:** Train multiple models on different bootstrap samples of the data, then aggregate predictions.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BAGGING                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Original Data                                                              │
│      │                                                                      │
│      ├──→ Bootstrap Sample 1 ──→ Model 1 ──┐                               │
│      ├──→ Bootstrap Sample 2 ──→ Model 2 ──┼──→ Aggregate ──→ Final       │
│      ├──→ Bootstrap Sample 3 ──→ Model 3 ──┤     (Vote/     Prediction    │
│      └──→ Bootstrap Sample n ──→ Model n ──┘     Average)                  │
│                                                                             │
│  Classification: Majority voting                                            │
│  Regression: Averaging                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Points:**
- Reduces variance
- Each model trained independently (can parallelize)
- Works well with high-variance models (like decision trees)

### Random Forest

**What it does:** Bagging with decision trees + random feature selection at each split.

**Key Innovations:**
1. Bootstrap samples for each tree
2. Random subset of features considered at each split
3. No pruning (let trees grow fully)

**Why it works:**
- Reduces correlation between trees
- Further reduces variance
- Creates diverse ensemble

**Implementation:**

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Classification
model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=None,        # Maximum depth of trees
    min_samples_split=2,   # Minimum samples to split
    max_features='sqrt',   # Features to consider at each split
    random_state=42
)
model.fit(X_train, y_train)

# Feature importance
importances = model.feature_importances_
```

### Boosting

**Concept:** Train models sequentially, each focusing on the mistakes of previous models.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BOOSTING                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Model 1 ──→ Errors ──→ Model 2 ──→ Errors ──→ Model 3 ──→ ... ──→ Final  │
│     ↑                      ↑                      ↑                         │
│  Train on              Train on               Train on                      │
│  original              reweighted             reweighted                    │
│  data                  data                   data                          │
│                                                                             │
│  Final = Weighted combination of all models                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### AdaBoost (Adaptive Boosting)

**How it works:**
1. Train a weak learner on the data
2. Increase weights of misclassified samples
3. Train next weak learner on reweighted data
4. Repeat and combine with weighted voting

**Implementation:**

```python
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
model.fit(X_train, y_train)
```

### Gradient Boosting

**How it works:**
1. Start with a simple prediction (e.g., mean)
2. Calculate residuals (errors)
3. Train a new model to predict the residuals
4. Add new model's predictions to previous predictions
5. Repeat

**Mathematical Intuition:**
```
F₀(x) = initial prediction (e.g., mean)
F₁(x) = F₀(x) + η × h₁(x)  where h₁ predicts residuals of F₀
F₂(x) = F₁(x) + η × h₂(x)  where h₂ predicts residuals of F₁
...
Fₘ(x) = Fₘ₋₁(x) + η × hₘ(x)

η = learning rate (shrinkage)
```

**Implementation:**

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)
```

### XGBoost (Extreme Gradient Boosting)

**What it does:** Optimized gradient boosting with regularization and efficiency improvements.

**Key Features:**
- L1 and L2 regularization
- Parallel processing
- Handles missing values
- Built-in cross-validation
- Tree pruning

**Implementation:**

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    reg_alpha=0,      # L1 regularization
    reg_lambda=1,     # L2 regularization
    random_state=42
)
model.fit(X_train, y_train)
```

### Comparison of Ensemble Methods

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENSEMBLE METHODS COMPARISON                               │
├─────────────────┬───────────────────┬───────────────────────────────────────┤
│ Method          │ Training          │ Focus                                 │
├─────────────────┼───────────────────┼───────────────────────────────────────┤
│ Bagging         │ Parallel          │ Reduce variance                       │
│ Random Forest   │ Parallel          │ Reduce variance + decorrelate trees  │
│ AdaBoost        │ Sequential        │ Focus on hard examples                │
│ Gradient Boost  │ Sequential        │ Minimize loss via gradient descent    │
│ XGBoost         │ Parallel trees    │ Optimized gradient boosting           │
└─────────────────┴───────────────────┴───────────────────────────────────────┘
```

---

## Clustering Algorithms

### K-Means Clustering

**What it does:** Partitions data into k clusters by minimizing within-cluster variance.

**Algorithm:**
1. Initialize k centroids randomly
2. Assign each point to nearest centroid
3. Recalculate centroids as mean of assigned points
4. Repeat until convergence

```
Iteration 1:         Iteration 2:         Final:
    * *                 * *                 * *
   *   * ○            *   * ○             *   * ●
    * *                 * *                 * *
                ──→                  ──→
  ○  * * *            ●  * * *            ●  * * *
    * * *                * * *                * * *

○ = centroid position changes until convergence
```

**Mathematical Objective:**
```
Minimize: Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²

Where:
- Cᵢ = cluster i
- μᵢ = centroid of cluster i
```

**Implementation:**

```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3, random_state=42)
model.fit(X)

# Get cluster labels
labels = model.labels_

# Get centroids
centroids = model.cluster_centers_

# Predict for new data
new_labels = model.predict(X_new)
```

**Choosing k (Number of Clusters):**

1. **Elbow Method:**
```python
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Plot and look for "elbow"
plt.plot(range(1, 11), inertias, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()
```

2. **Silhouette Score:**
```python
from sklearn.metrics import silhouette_score

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"k={k}, Silhouette Score: {score:.3f}")
```

### Hierarchical Clustering

**What it does:** Builds a hierarchy of clusters, either bottom-up (agglomerative) or top-down (divisive).

**Agglomerative (Bottom-up):**
1. Start with each point as its own cluster
2. Merge two closest clusters
3. Repeat until desired number of clusters

**Linkage Methods:**

| Method | Description |
|--------|-------------|
| Single | Minimum distance between clusters |
| Complete | Maximum distance between clusters |
| Average | Average distance between all pairs |
| Ward | Minimize variance when merging |

**Implementation:**

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Fit model
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(X)

# Create dendrogram
linked = linkage(X, method='ward')
dendrogram(linked)
plt.show()
```

### DBSCAN (Density-Based Spatial Clustering)

**What it does:** Groups points that are closely packed together, marks outliers.

**Key Parameters:**
- **eps:** Maximum distance between two samples to be considered neighbors
- **min_samples:** Minimum points required to form a dense region

**Point Types:**
- **Core points:** Have at least min_samples within eps
- **Border points:** Within eps of a core point but not core themselves
- **Noise points:** Neither core nor border (outliers)

**Implementation:**

```python
from sklearn.cluster import DBSCAN

model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X)

# -1 indicates noise/outliers
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
```

**Advantages over K-Means:**
- No need to specify number of clusters
- Can find arbitrarily shaped clusters
- Automatically identifies outliers
- Robust to outliers

---

## Dimensionality Reduction

### Principal Component Analysis (PCA)

**What it does:** Transforms data to a new coordinate system where the greatest variance lies on the first axis (first principal component), second greatest on second axis, etc.

**Mathematical Process:**
1. Standardize the data
2. Compute covariance matrix
3. Calculate eigenvalues and eigenvectors
4. Sort by eigenvalues (descending)
5. Select top k eigenvectors
6. Transform data

**Implementation:**

```python
from sklearn.decomposition import PCA

# Reduce to 2 dimensions
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)

# Explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:", sum(pca.explained_variance_ratio_))

# Cumulative explained variance
cumsum = np.cumsum(pca.explained_variance_ratio_)
# Choose n_components to retain 95% variance
n_components = np.argmax(cumsum >= 0.95) + 1
```

**Choosing Number of Components:**

```python
# Method 1: Retain 95% variance
pca = PCA(n_components=0.95)
X_transformed = pca.fit_transform(X)

# Method 2: Scree plot
pca = PCA()
pca.fit(X)
plt.plot(range(1, len(pca.explained_variance_ratio_)+1), 
         pca.explained_variance_ratio_, marker='o')
plt.xlabel('Component')
plt.ylabel('Explained Variance Ratio')
plt.show()
```

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

**What it does:** Non-linear dimensionality reduction for visualization, preserves local structure.

**Implementation:**

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X)

# Visualize
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels)
plt.show()
```

**Key Parameters:**
- **perplexity:** Balance between local and global aspects (typically 5-50)
- **n_iter:** Number of iterations

**Note:** t-SNE is mainly for visualization, not for preprocessing before ML models.

---

## Model Evaluation Metrics

### Regression Metrics

#### 1. MAE (Mean Absolute Error)

**What:** Average of absolute differences between actual and predicted values.

**Formula:**
```
MAE = (1/n) × Σ|yᵢ - ŷᵢ|
```

**Intuition:** "On average, how much is the model off?"

**Reason:** Useful when you want a simple, interpretable measure of average error **without penalizing outliers too heavily**.

#### 2. MSE (Mean Squared Error)

**What:** Average of squared differences between actual and predicted values.

**Formula:**
```
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
```

**Intuition:** Penalizes larger errors more.

**Reason:** Helps **catch and penalize outliers more harshly**. Ideal when large errors are more problematic.

#### 3. RMSE (Root Mean Squared Error)

**What:** Square root of the MSE.

**Formula:**
```
RMSE = √[(1/n) × Σ(yᵢ - ŷᵢ)²]
```

**Intuition:** Same units as the target variable.

**Reason:** Makes MSE more interpretable by putting it **back in the same units as the output**. Still emphasizes big errors.

#### 4. R² Score (Coefficient of Determination)

**What:** Proportion of variance in the target explained by the model.

**Formula:**
```
R² = 1 - [Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²]
```

**Intuition:** 1 = perfect, 0 = as good as mean prediction.

**Reason:** Tells you **how much better your model is than a baseline model that just predicts the mean**.

#### 5. Adjusted R² Score

**What:** Penalized R² that adjusts for number of features.

**Formula:**
```
Adjusted R² = 1 - [(1 - R²) × (n - 1) / (n - k - 1)]

Where:
- n = number of samples
- k = number of features
```

**Intuition:** Decreases if adding new features doesn't improve the model enough.

**Reason:** Prevents **overfitting** by penalizing unnecessary predictors.

#### Regression Metric Comparison

| Metric | Penalizes Big Errors? | Same Units as Target? | Range | Goal | Reason |
|--------|----------------------|----------------------|-------|------|--------|
| MAE | ❌ No | ✅ Yes | ≥ 0 | Lower is better | Simple average error |
| MSE | ✅ Yes | ❌ No | ≥ 0 | Lower is better | Highlights big errors |
| RMSE | ✅ Yes | ✅ Yes | ≥ 0 | Lower is better | Interpretable MSE |
| R² | ❌ No | ❌ Unitless | (-∞, 1] | Higher is better | Compares to baseline |
| Adjusted R² | ❌ No (adds penalty) | ❌ Unitless | (-∞, 1] | Higher is better | Prevents overfitting |

### Classification Metrics

#### Confusion Matrix

```
              │ Predicted ✅ │ Predicted ❌ │
──────────────┼──────────────┼──────────────┤
Actual ✅     │     TP       │     FN       │
              │ (True Pos)   │ (False Neg)  │
──────────────┼──────────────┼──────────────┤
Actual ❌     │     FP       │     TN       │
              │ (False Pos)  │ (True Neg)   │
```

#### 1. Accuracy

**What:** Overall proportion of correct predictions.

**Formula:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Reason:** Easy to understand and works well **only when classes are balanced**.

**Limitation:** Can be **misleading for imbalanced datasets**.

#### 2. Precision

**What:** Of predicted positives, how many were actually correct?

**Formula:**
```
Precision = TP / (TP + FP)
```

**Reason:** Important when **false positives are costly**, e.g., flagging legit users as fraud.

#### 3. Recall (Sensitivity / True Positive Rate)

**What:** Of actual positives, how many did the model correctly identify?

**Formula:**
```
Recall = TP / (TP + FN)
```

**Reason:** Important when **false negatives are costly**, e.g., missing cancer diagnoses.

#### 4. F1 Score

**What:** Harmonic mean of precision and recall.

**Formula:**
```
F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
```

**Reason:** Balances precision and recall, useful when you need **both false positives and false negatives controlled**.

#### Classification Metric Use-Cases

| Metric | Focus | Best When... | Reason |
|--------|-------|--------------|--------|
| Accuracy | Overall correctness | Data is balanced | Fast and intuitive |
| Precision | Quality of positives | False positives are costly (e.g., spam) | Focus on being right when predicting positive |
| Recall | Quantity of positives | False negatives are costly (e.g., cancer) | Don't miss true positives |
| F1 Score | Balance between both | You need both precision & recall | Compromise when both are important |

### Implementation

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

# Classification metrics
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Regression metrics
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
```

---

## Gradient Descent and Optimization

### What is Gradient Descent?

Gradient descent is an optimization algorithm used to minimize a function by iteratively moving in the direction of steepest descent.

**Mathematical Formula:**
```
θ_new = θ_old - η × ∇J(θ)

Where:
- θ = parameters (weights)
- η = learning rate
- ∇J(θ) = gradient of cost function
```

### Types of Gradient Descent

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      GRADIENT DESCENT TYPES                                  │
├─────────────────┬───────────────────────────────────────────────────────────┤
│                 │                                                           │
│  BATCH GD       │  • Uses ENTIRE dataset for each update                   │
│  (BGD)          │  • Weights updated ONCE per epoch                        │
│                 │  • Smooth convergence but SLOW                           │
│                 │  • Memory intensive for large datasets                    │
│                 │                                                           │
│                 │  θ = θ - η × (1/N) × Σ∇J(xᵢ, yᵢ)                        │
│                 │                                                           │
├─────────────────┼───────────────────────────────────────────────────────────┤
│                 │                                                           │
│  STOCHASTIC GD  │  • Uses ONE sample at a time                             │
│  (SGD)          │  • Weights updated after EACH sample                     │
│                 │  • Noisy but FAST                                        │
│                 │  • Can escape local minima                                │
│                 │                                                           │
│                 │  θ = θ - η × ∇J(xᵢ, yᵢ)                                  │
│                 │                                                           │
├─────────────────┼───────────────────────────────────────────────────────────┤
│                 │                                                           │
│  MINI-BATCH GD  │  • Uses BATCH of samples (e.g., 32, 64, 128)            │
│                 │  • Best of both worlds                                    │
│                 │  • Most commonly used in practice                         │
│                 │                                                           │
│                 │  θ = θ - η × (1/B) × Σ∇J(xᵢ, yᵢ)  [B = batch size]      │
│                 │                                                           │
└─────────────────┴───────────────────────────────────────────────────────────┘
```

### Learning Rate

**Too High:** Overshoots minimum, may diverge

**Too Low:** Very slow convergence

**Just Right:** Converges smoothly to minimum

```
Learning Rate Effect:

Too High:           Too Low:            Just Right:
    │                   │                   │
Loss│   ╱╲              │ ────────────      │ ╲
    │  ╱  ╲             │                   │  ╲
    │ ╱    ╲            │     ─────────     │   ╲
    │╱      ╲           │                   │    ╲_____
    └────────→          └─────────────→     └──────────→
      Epochs              Epochs              Epochs
```

### Advanced Optimizers

```python
# In deep learning frameworks:

# SGD with momentum
optimizer = SGD(learning_rate=0.01, momentum=0.9)

# Adam (Adaptive Moment Estimation)
optimizer = Adam(learning_rate=0.001)

# RMSprop
optimizer = RMSprop(learning_rate=0.001)
```

---

## Time Series Analysis

### What is Time Series Data?

Data collected over time, where the order and timing of observations matter.

**Components of Time Series:**
1. **Trend:** Long-term increase or decrease
2. **Seasonality:** Regular patterns that repeat
3. **Cyclical:** Irregular patterns over longer periods
4. **Noise/Residual:** Random variation

### Key Concepts

**Stationarity:**
- Mean and variance don't change over time
- Required for many time series models
- Test with ADF (Augmented Dickey-Fuller) test

**Autocorrelation:**
- Correlation of a signal with a delayed copy of itself
- ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function)

### Common Models

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TIME SERIES MODELS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ARIMA (AutoRegressive Integrated Moving Average)                           │
│  • AR(p): Past values influence future                                      │
│  • I(d): Differencing for stationarity                                      │
│  • MA(q): Past errors influence future                                      │
│                                                                             │
│  SARIMA: ARIMA with seasonality                                             │
│                                                                             │
│  Exponential Smoothing: Weighted average of past observations               │
│                                                                             │
│  Prophet: Facebook's tool for business forecasting                          │
│                                                                             │
│  LSTM: Deep learning for sequences                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation Example

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Test for stationarity
result = adfuller(time_series)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# Fit ARIMA model
model = ARIMA(time_series, order=(p, d, q))
fitted = model.fit()

# Forecast
forecast = fitted.forecast(steps=10)
```

---

## Model Deployment

### Deployment Options

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT OPTIONS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. REST API (Flask/FastAPI)                                                │
│     • Expose model as web service                                           │
│     • Easy to integrate with applications                                   │
│                                                                             │
│  2. Cloud Platforms                                                         │
│     • AWS SageMaker                                                         │
│     • Google Cloud AI Platform                                              │
│     • Azure Machine Learning                                                │
│                                                                             │
│  3. Containerization (Docker)                                               │
│     • Portable and reproducible                                             │
│     • Easy scaling with Kubernetes                                          │
│                                                                             │
│  4. Edge Deployment                                                         │
│     • Run on mobile devices or IoT                                          │
│     • TensorFlow Lite, ONNX                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Flask Deployment Example

```python
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = data['features']
    prediction = model.predict([features])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
```

### Model Serialization

```python
import pickle
import joblib

# Save model
# Method 1: pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Method 2: joblib (better for large numpy arrays)
joblib.dump(model, 'model.joblib')

# Load model
model = pickle.load(open('model.pkl', 'rb'))
model = joblib.load('model.joblib')
```

### Deployment to Heroku

1. **Prepare the Application:**

```python
# app.py
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    # Model prediction logic here
    return jsonify({'prediction': prediction})
```

2. **Requirements File (requirements.txt):**
```
flask
scikit-learn
gunicorn
```

3. **Procfile:**
```
web: gunicorn app:app
```

4. **Deploy to Heroku:**
```bash
git init
heroku create your-app-name
git add .
git commit -m "Initial commit"
git push heroku master
heroku open
```

---

## Quick Reference and Cheat Sheets

### Algorithm Selection Guide

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHICH ALGORITHM TO USE?                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  REGRESSION (Continuous output):                                            │
│  ├── Linear data → Linear Regression                                        │
│  ├── Non-linear data → Polynomial, Decision Tree, Random Forest             │
│  ├── Overfitting concern → Ridge, Lasso, Elastic Net                        │
│  └── Complex patterns → Gradient Boosting, XGBoost                          │
│                                                                             │
│  CLASSIFICATION (Categorical output):                                        │
│  ├── Linear separable → Logistic Regression, SVM (linear)                   │
│  ├── Non-linear → SVM (RBF), Decision Tree, Random Forest                   │
│  ├── Probabilistic → Naive Bayes                                            │
│  ├── Instance-based → KNN                                                   │
│  └── Complex patterns → Gradient Boosting, XGBoost                          │
│                                                                             │
│  CLUSTERING (No labels):                                                     │
│  ├── Known k, spherical clusters → K-Means                                  │
│  ├── Hierarchical structure → Hierarchical Clustering                        │
│  └── Arbitrary shapes, outlier detection → DBSCAN                           │
│                                                                             │
│  DIMENSIONALITY REDUCTION:                                                   │
│  ├── Linear, preprocessing → PCA                                            │
│  └── Visualization → t-SNE                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Common sklearn Imports

```python
# Data preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import silhouette_score
```

### Hyperparameter Tuning Template

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Random Search (for larger parameter spaces)
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20)
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(),
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)
```

### Cross-Validation Template

```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

# Simple cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

# K-Fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in kfold.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Train and evaluate model

# Stratified K-Fold (for imbalanced classification)
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in skfold.split(X, y):
    # Train and evaluate model
```

---

## Summary: The Complete ML Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE ML PIPELINE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. PROBLEM DEFINITION                                                      │
│     └── Business problem → ML problem → Success metrics                    │
│                                                                             │
│  2. DATA COLLECTION                                                         │
│     └── Identify sources → Collect → Document                              │
│                                                                             │
│  3. DATA ASSESSMENT                                                         │
│     └── Quality check → Discover issues → Document findings                │
│                                                                             │
│  4. DATA CLEANING                                                           │
│     └── Handle missing values → Fix inconsistencies → Remove duplicates    │
│                                                                             │
│  5. EDA                                                                     │
│     └── Univariate → Bivariate → Multivariate analysis                     │
│                                                                             │
│  6. FEATURE ENGINEERING                                                     │
│     └── Create features → Scale → Select relevant features                 │
│                                                                             │
│  7. MODEL SELECTION                                                         │
│     └── Choose algorithm(s) → Define baseline                              │
│                                                                             │
│  8. MODEL TRAINING                                                          │
│     └── Split data → Train → Validate                                      │
│                                                                             │
│  9. HYPERPARAMETER TUNING                                                   │
│     └── Grid/Random search → Cross-validation                              │
│                                                                             │
│  10. MODEL EVALUATION                                                       │
│      └── Test set evaluation → Compare metrics → Final selection           │
│                                                                             │
│  11. DEPLOYMENT                                                             │
│      └── Serialize model → Create API → Deploy → Monitor                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Conclusion

This guide covered the complete Machine Learning journey from understanding business problems to deploying models in production. Key takeaways:

1. **Start with the Problem** - Clearly define what you're trying to solve
2. **Data is King** - Invest time in understanding, cleaning, and preparing your data
3. **EDA is Essential** - Explore your data before modeling
4. **Feature Engineering Matters** - Good features often matter more than fancy algorithms
5. **Choose the Right Algorithm** - Match the algorithm to your problem type
6. **Evaluate Properly** - Use appropriate metrics and validation strategies
7. **Iterate** - ML is an iterative process, not a one-time task
8. **Deploy and Monitor** - Models need to be deployed and monitored in production

> **"The goal of machine learning is to find patterns in data that generalize well to new, unseen data."**

---

## Useful Resources

- [Full Complete Everything Roadmap for Data Science](https://github.com/SamBelkacem/AI-ML-cheatsheets)
- [100 Days of ML by CampusX](https://github.com/campusx-official/100-days-of-machine-learning)
- [Course for Absolute Beginners - Jovian](https://jovian.com/learn/data-analysis-with-python-zero-to-pandas)
- [ML Algorithms - Regression](https://www.youtube.com/watch?v=UZPfbG0jNec&list=PLKnIA16_Rmva-wY_HBh1gTH32ocu2SoTr)
- [Gradient Descent](https://www.youtube.com/watch?v=ORyfPJypKuU&list=PLKnIA16_RmvZvBbJex7T84XYRmor3IPK1)
- [Gradient Boosting](https://www.youtube.com/watch?v=fbKz7N92mhQ&list=PLKnIA16_RmvaMPgWfHnN4MXl3qQ1597Jw)
- [Logistic Regression](https://www.youtube.com/watch?v=XNXzVfItWGY&list=PLKnIA16_Rmvb-ZTsM1QS-tlwmlkeGSnru)
- [PCA](https://www.youtube.com/watch?v=ToGuhynu-No&list=PLKnIA16_RmvYHW62E_lGQa0EFsph2NquD)
- [Random Forest](https://www.youtube.com/watch?v=ToGuhynu-No&list=PLKnIA16_RmvYHW62E_lGQa0EFsph2NquD)
- [AdaBoost](https://www.youtube.com/watch?v=sFKnP0iP0K0&list=PLKnIA16_RmvZxriy68dPZhorB8LXP1PY6)
- [XGBoost](https://www.youtube.com/watch?v=BTLB-ppqBZc&list=PLKnIA16_RmvbXJbBW4zCy4Xbr81GRyaC4)
- [K-Means Clustering](https://www.youtube.com/watch?v=5shTLzwAdEc&list=PLKnIA16_RmvbA_hYXlRgdCg9bn8ZQK2z9)
- [Bagging Ensemble](https://www.youtube.com/watch?v=LUiBOAy7x6Y&list=PLKnIA16_RmvZ7iKIcJrLjUoFDEeSejRpn)
- [Time Series Analysis](https://www.youtube.com/watch?v=A3fowDMo8mM)
- [Graphy for Data Storytelling](https://graphy.app/)

---

*This guide was compiled from comprehensive Data Science course materials with detailed explanations, practical implementations, and real-world applications.*
