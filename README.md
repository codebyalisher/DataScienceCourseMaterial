# DataScienceCourseMaterial
   - **[Full Complete Everything Roadmap for Data Science](https://github.com/SamBelkacem/AI-ML-cheatsheets)**
   - **[Course for Absolute beginers Website](https://jovian.com/learn/data-analysis-with-python-zero-to-pandas), [YouTube Channel](https://www.youtube.com/@jovianhq/playlists)**
   - **[Overview of Data Science](https://www.linkedin.com/pulse/data-science-methodology-step-by-step-guide-uzair-azmat-5tekf/?trackingId=DOxr4vPTsiNgGbFTdDijoQ%3D%3D)**
   - **[General Concepts](https://www.linkedin.com/pulse/complete-data-analysis-guide-python-uzair-azmat-uavvf/?trackingId=QNtfgWzo5XW04hwg3EPwUQ%3D%3D)**
## How to Start the Project regarding any Domain's Problem
1. **Business Problem to ML Problem:**

   **Goal:** The very first step is to clearly define the business problem you're trying to solve. This involves understanding the core issue, the desired outcome, and the impact of a successful solution on the business. <br>
   **Translation:** Once the business problem is well-defined, the next critical step is to translate it into a specific, well-posed machine learning problem. This involves determining what kind of prediction or insight is needed.<br>
   **Examples:** <br>
        **Business Problem:** Increase customer retention.<br>
        **ML Problem:** Predict which customers are at high risk of churn (Classification).<br>
        **Business Problem:** Improve sales revenue.<br>
        **ML Problem:** Forecast future product demand (Regression).<br>
        **Business Problem:** Automate customer support.<br>
        **ML Problem:** Identify the intent of customer inquiries (Natural Language Understanding/Classification).<br>

2. **Type of Problem:**

   **Goal:** Understanding the nature of the ML problem is crucial for selecting appropriate algorithms, evaluation metrics, and data preprocessing techniques.<br>
  **Categorization:** This step involves identifying whether the problem falls into categories like:<br>
       **Supervised Learning:** Learning from labeled data (e.g., Classification, Regression).<br>
        **Unsupervised Learning:** Discovering patterns in unlabeled data (e.g., Clustering, Dimensionality Reduction).<br>
        **Reinforcement Learning:** Training an agent to make decisions in an environment to maximize a reward.<br>
   **Sub-Categorization:** Further specifying the type within these categories (e.g., Binary Classification vs. Multi-class Classification, Linear Regression vs. Polynomial Regression).<br>

3. **Current Solution:**

   **Goal:** Before building a new ML solution, it's essential to understand the existing methods or processes used to address the business problem.<br>
   **Analysis:** This involves documenting the current workflow, its effectiveness, its limitations (e.g., manual effort, scalability issues, accuracy), and its associated costs.<br>
   **Baseline:** The performance of the current solution often serves as a baseline against which the ML model's performance will be compared. Understanding the current solution helps justify the need for an ML-based approach and sets realistic expectations.<br>

4. **Getting Data:**

   **Goal:** Machine learning models are data-driven. This step focuses on identifying, sourcing, and acquiring the necessary data to train and evaluate the model.<br>
   **Considerations:** This includes:<br>
        **Data Sources:** Where does the relevant data reside (databases, APIs, logs, external datasets, etc.)?<br>
        **Data Collection:** How will the data be collected and accessed?<br>
        **Data Volume and Quality:** Is there enough data? What is the quality of the data (missing values, inconsistencies, errors)?<br>
        **Data Privacy and Security:** Are there any regulations or ethical considerations related to the data?<br>
   **Data Summary:** <br>
        **Purpose:** Provides a high-level overview of the dataset.<br>
   **Column Details:** <br>
        **Purpose:** Provides a more in-depth look at each individual column.<br>
5. **Metrics to Measure:**

     **Goal:** Defining clear and relevant metrics is crucial for evaluating the performance of the ML model and determining if it effectively solves the business problem.<br>
     **Selection:** The choice of metrics depends on the type of ML problem:<br>
           **Classification:** Accuracy, Precision, Recall, F1-Score, AUC-ROC.<br>
           **Regression:** Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R-squared.<br>
           **Unsupervised Learning:** Silhouette Score, Davies-Bouldin Index (though often more subjective).<br>
      **Business Alignment:** The chosen metrics should align with the business objectives defined in the first step. For example, in a fraud detection system, recall (minimizing missed fraud cases) might be more critical than overall accuracy.<br>
            **📈 Regression Metrics with**
      
      These metrics evaluate how well a regression model predicts continuous values.
      
      ---
      
      #### 1. MAE (Mean Absolute Error)
      
      - **What**: Average of absolute differences between actual and predicted values.  
      - **Formula**:  
        ```
        MAE = (1/n) * Σ |yᵢ - ŷᵢ|
        ```
      - **Intuition**: "On average, how much is the model off?"
      - **Reason**: Useful when you want a simple, interpretable measure of average error **without penalizing outliers too heavily**.
      
      ---
      
      #### 2. MSE (Mean Squared Error)
      
      - **What**: Average of squared differences between actual and predicted values.  
      - **Formula**:  
        ```
        MSE = (1/n) * Σ (yᵢ - ŷᵢ)²
        ```
      - **Intuition**: Penalizes larger errors more.
      - **Reason**: Helps **catch and penalize outliers more harshly**. Ideal when large errors are more problematic.
      
      ---
      
      #### 3. RMSE (Root Mean Squared Error)
      
      - **What**: Square root of the MSE.  
      - **Formula**:  
        ```
        RMSE = sqrt((1/n) * Σ (yᵢ - ŷᵢ)²)
        ```
      - **Intuition**: Same units as the target variable.
      - **Reason**: Makes MSE more interpretable by putting it **back in the same units as the output**. Still emphasizes big errors.
      
      ---
      
      #### 4. R² Score (Coefficient of Determination)
      
      - **What**: Proportion of variance in the target explained by the model.  
      - **Formula**:  
        ```
        R² = 1 - [Σ (yᵢ - ŷᵢ)² / Σ (yᵢ - ȳ)²]
        ```
      - **Intuition**: 1 = perfect, 0 = as good as mean prediction.
      - **Reason**: Tells you **how much better your model is than a baseline model that just predicts the mean**.
      
      ---
      
      #### 5. Adjusted R² Score
      
      - **What**: Penalized R² that adjusts for number of features.  
      - **Formula**:  
        ```
        Adjusted R² = 1 - [(1 - R²) * (n - 1) / (n - k - 1)]
        ```
        - `n` = number of samples  
        - `k` = number of features
      
      - **Intuition**: Decreases if adding new features doesn’t improve the model enough.
      - **Reason**: Prevents **overfitting** by penalizing unnecessary predictors.
      
      ---
      
      #### 📊 Regression Metric Comparison
      
      | Metric         | Penalizes Big Errors? | Same Units as Target? | Range           | Goal               | Reason |
      |----------------|-----------------------|------------------------|------------------|--------------------|--------|
      | MAE            | ❌ No                 | ✅ Yes                 | ≥ 0              | Lower is better    | Simple average error |
      | MSE            | ✅ Yes               | ❌ No                  | ≥ 0              | Lower is better    | Highlights big errors |
      | RMSE           | ✅ Yes               | ✅ Yes                 | ≥ 0              | Lower is better    | Interpretable MSE |
      | R²             | ❌ No                | ❌ Unitless            | (-∞, 1]          | Higher is better   | Compares to baseline |
      | Adjusted R²    | ❌ No (adds penalty)  | ❌ Unitless            | (-∞, 1]          | Higher is better   | Prevents overfitting |
      
      ---
      
      #### 🤖 Classification Metrics with Reasons
      
      These metrics help evaluate classification models, especially when **accuracy alone is misleading**.
      
      ---
      
      #### 🧩 Confusion Matrix
      
      |               | Predicted ✅ | Predicted ❌ |
      |---------------|-------------|--------------|
      | **Actual ✅**  | TP (True Positive)  | FN (False Negative) |
      | **Actual ❌**  | FP (False Positive) | TN (True Negative)  |
      
      ---
      
      #### 1. Accuracy
      
      - **What**: Overall proportion of correct predictions.  
      - **Formula**:  
        ```
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        ```
      - **Reason**: Easy to understand and works well **only when classes are balanced**.
      - **Limitation**: Can be **misleading for imbalanced datasets**.
      
      ---
      
      #### 2. Precision
      
      - **What**: Of predicted positives, how many were actually correct?  
      - **Formula**:  
        ```
        Precision = TP / (TP + FP)
        ```
      - **Reason**: Important when **false positives are costly**, e.g., flagging legit users as fraud.
      
      ---
      
      #### 3. Recall (Sensitivity / True Positive Rate)
      
      - **What**: Of actual positives, how many did the model correctly identify?  
      - **Formula**:  
        ```
        Recall = TP / (TP + FN)
        ```
      - **Reason**: Important when **false negatives are costly**, e.g., missing cancer diagnoses.
      
      ---
      
      #### 4. F1 Score
      
      - **What**: Harmonic mean of precision and recall.  
      - **Formula**:  
        ```
        F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        ```
      - **Reason**: Balances precision and recall, useful when you need **both false positives and false negatives controlled**.
      
      ---
      
      #### 🎯 Classification Metric Use-Cases
      
      | Metric     | Focus                    | Best When…                             | Reason |
      |------------|--------------------------|----------------------------------------|--------|
      | Accuracy   | Overall correctness       | Data is balanced                       | Fast and intuitive |
      | Precision  | Quality of positives      | False positives are costly (e.g. spam) | Focus on being right when predicting positive |
      | Recall     | Quantity of positives     | False negatives are costly (e.g. cancer) | Don't miss true positives |
      | F1 Score   | Balance between both      | You need both precision & recall       | Compromise when both are important |



6. **Online Vs Batch?**

      **Goal:** This step considers how the ML model will be trained and how it will make predictions in a real-world setting.<br>
      **Online Learning:** The model learns incrementally as new data arrives. Suitable for dynamic environments where data streams continuously.<br>
      **Batch Learning:** The model is trained on the entire available dataset at once. Requires retraining on the updated dataset when new data becomes available.<br>
      **Decision Factors:** The choice depends on factors like the volume and velocity of data, the need for real-time predictions, computational resources, and the stability of the underlying data patterns.<br>

7. **Check Assumptions:**

      **Goal:** Machine learning models often rely on certain assumptions about the data and the underlying relationships. It's crucial to explicitly identify and check these assumptions.<br>
      **Examples:** <br>
          **Linear Regression:** Assumes a linear relationship between features and the target variable, independence of errors, homoscedasticity (constant variance of errors), and normality of errors.<br>
          **Naive Bayes:** Assumes feature independence given the class.<br>
          **Time Series Forecasting:** Assumes stationarity or specific patterns in the time series data.<br>
      **Importance:** Violating these assumptions can lead to poorly performing or unreliable models. This step might involve statistical tests, visualizations, and domain expertise to validate the assumptions.

## 5-Step Data Analyst Process

## 1. **Asking for the Problem (Understanding the Business Question)**

- **What it is:** Before jumping into any data, the first step is to **understand the problem** you're trying to solve. This is where you communicate with stakeholders (like business leaders, managers, or clients) to know exactly what they need from the data.
- **Why it matters:** If you don’t understand the question, you might analyze the wrong data or go down the wrong path.
  
**Example:** If a company asks, "What’s causing a drop in sales?" you’ll need to figure out what kind of data might help answer that, like sales trends, customer feedback, or inventory data.

---

## 2. **Data Wrangling (Data Gathering, Assessing, and Cleaning)**

- **What it is:** Data wrangling is about **collecting, preparing, and cleaning** data for analysis. It’s the behind-the-scenes work that makes sure your data is in a usable form.
  
#### Key Steps in Data Wrangling:
  - **Data Gathering:** Collect data from various sources like databases, spreadsheets, APIs, etc. **[Youtube Link](https://youtu.be/dA6ZksRR6aw?si=9TAiQx9H0nVO-b-g)**
    - **Example:** Getting sales data from a database and customer feedback from an Excel file.
  - **Data Assessing:** Check the quality and structure of the data (looking for missing values, duplicates, outliers, etc.) **[Youtube Link](https://www.youtube.com/live/-HnN8GBINnc?si=FRvFLaTKdtKalwGK)**
    - **Example:** Reviewing if some values are missing, incorrect, or formatted wrong.
  - **Data Cleaning:** Fix issues found in the assessing stage. Remove or fill missing data, correct errors, or standardize formats.
    - **Example:** If "USA" is written in different ways, you’ll standardize it to one format ("United States").

---

## 3. **Exploratory Data Analysis (EDA) [Youtube Link](https://www.youtube.com/watch?v=PPEHpg2RixQ&list=PLKnIA16_RmvbAlyx4_rdtR66B7EHX5k3z&index=91)** 

- **What it is:** This step is all about **understanding the data**. You explore and analyze it to look for patterns, trends, or relationships.
  
#### EDA includes:
  - **Descriptive Statistics:** Check things like the mean, median, and standard deviation to understand the central tendencies of your data.
    - **Example:** Calculate the average sales per month.
  - **Visualizations:** Create charts like histograms, bar graphs, and scatter plots to get a quick feel for data trends.
    - **Example:** A bar chart of sales over time to spot any dips or spikes.
  - **Correlation:** Look for relationships between variables.
    - **Example:** See if there’s a link between the amount of advertising spend and sales growth.
#### Why do EDA
* Model building
* Analysis and reporting
* Validate assumptions
* Handling missing values
* Feature engineering
* Detecting outliers

`For convenience to Perform EDA ,assign the Each Column with these labels`
#### Column Types
   **Numerical** - Age,Fare,PassengerId <br>
   **Categorical** - Survived, Pclass, Sex, SibSp, Parch, Embarked <br>
   **Mixed** - Name, Ticket, Cabin
#### Try to change the dtype of columns if possible as it reduces the memory   
#### Univariate Analysis
   Univariate analysis focuses on analyzing each feature in the dataset independently.   
   **Distribution analysis:** The distribution of each feature is examined to identify its shape, central tendency, and dispersion. <br>
   **Identifying potential issues:** Univariate analysis helps in identifying potential problems with the data such as outliers, skewness, and missing values. <br>
The shape of a data distribution refers to its overall pattern or form as it is represented on a graph. Some common shapes of data distributions include:

   - **Normal Distribution:** A symmetrical and bell-shaped distribution where the mean, median, and mode are equal and the majority of the data falls in the middle of the distribution with gradually decreasing frequencies towards the tails.
   - **Skewed Distribution:** A distribution that is not symmetrical, with one tail being longer than the other. It can be either positively skewed (right-skewed) or negatively skewed (left-skewed).
   - **Bimodal Distribution:** A distribution with two peaks or modes.
   - **Uniform Distribution:** A distribution where all values have an equal chance of occurring. <br>
The shape of the data is important in identifying the presence of outliers,skewness and type of statistical tests and models that can be used for the furthere analysis.<br>
**Dispersion** is a statistical term used to describe the spread or variability of a set of data. It measures how far the values in a data set are spread out from the `central tendency (mean, median, or mode)` of the data.<br>
#### There are several measures of dispersion, including: 
   - **Range:** The difference between the largest and smallest values in a data set.
   - **Variance:** The average of the squared deviations of each value from the mean of the data set.
   - **Standard Deviation:** The square root of the variance. It provides a measure of the spread of the data that is in the same units as the original data.
   - **Interquartile range (IQR):** The range between the first quartile (25th percentile) and the third quartile (75th percentile) of the data. <br>
Dispersion helps to describe the spread of the data, which can help to identify the presence of outliers and skewness in the data.
#### Steps of doing Univariate Analysis on Numerical columns
   - **Descriptive Statistics:** Compute basic summary statistics for the column, such as mean, median, mode, standard deviation, range, and quartiles. These statistics give a general understanding of the distribution of the data and can help identify skewness or outliers.
   - **Visualizations:** Create visualizations to explore the distribution of the data. Some common visualizations for numerical data include histograms, box plots, and density plots. These visualizations provide a visual representation of the distribution of the data and can help identify skewness and outliers.
   - **Identifying Outliers:** Identify and examine any outliers in the data. Outliers can be identified using visualizations. It is important to determine whether the outliers are due to measurement errors, data entry errors, or legitimate differences in the data, and to decide whether to include or exclude them from the analysis.
   - **Skewness:** Check for skewness in the data and consider transforming the data or using robust statistical methods that are less sensitive to skewness, if necessary.
   - **Conclusion:** Summarize the findings of the EDA and make decisions about how to proceed with further analysis.
#### Steps of doing Univariate Analysis on Categorical columns
   - **Descriptive Statistics:** Compute the frequency distribution of the categories in the column. This will give a general understanding of the distribution of the categories and their relative frequencies.
   - **Visualizations:** Create visualizations to explore the distribution of the categories. Some common visualizations for categorical data include count plots and pie charts. These visualizations provide a visual representation of the distribution of the categories and can help identify any patterns or anomalies in the data.
   - **Missing Values:** Check for missing values in the data and decide how to handle them. Missing values can be imputed or excluded from the analysis, depending on the research question and the data set.
   - **Conclusion:** Summarize the findings of the EDA and make decisions about how to proceed with further analysis.
#### Steps of doing Bivariate Analysis
   **Select 2 cols**  <br>
   ***Understand type of relationship*** <br>
       1. **Numerical - Numerical** <br>
           a. You can plot graphs like scatterplot (regression plots), 2D histplot, 2D KDEplots <br>
           b. Check correlation coefficient to check linear relationship <br>
       2. **Numerical - Categorical** - create visualizations that compare the distribution of the numerical data across different categories of the categorical data. <br>
           a. You can plot graphs like barplot, boxplot, kdeplot violinplot even scatterplots<br>
       3. **Categorical - Categorical** <br>
           a. You can create cross-tabulations or contingency tables that show the distribution of values in one categorical column, grouped by the values in the other categorical column. <br>
           b. You can plot graphs like heatmap, stacked barplots, treemaps <br>
   **Write your conclusions**

---

## 4. **Conclusions (Drawing Insights and Analysis)**

- **What it is:** After exploring the data, it’s time to **interpret the results**. This step involves making sense of the findings and drawing actionable insights.
  
#### Here’s what happens:
  - **Answer the business question:** Relate your findings back to the original problem.
    - **Example:** "Sales are dropping in regions with lower ad spend."
  - **Find patterns:** Identify any trends, anomalies, or correlations that will help solve the problem.
    - **Example:** You notice that higher customer satisfaction scores correlate with repeat purchases.

---

## 5. **Presenting in Front of Others (Communicating Results)**

- **What it is:** The final step is **communicating your findings** clearly to the stakeholders (whether it’s in a meeting, report, or dashboard). It’s important to explain the data insights in a way that’s easy to understand for non-technical audiences.
  
#### Tips for Effective Presentation:
  - **Tell a story:** Present your analysis like a story with a clear beginning, middle, and end.
    - **Example:** "Here’s how the advertising spend affected sales, and here’s what we should do moving forward."
  - **Visualize findings:** Use charts, graphs, or dashboards to make your insights easier to digest.
    - **Example:** A line graph showing how sales grew with increased ad spend.
  - **Provide actionable recommendations:** Based on your analysis, suggest next steps.
    - **Example:** "I recommend increasing ad spend by 10% in underperforming regions."
    - **[Graphy for story telling of graphs](https://graphy.app/)** --> `Graphy enables anyone to become a skilled data storyteller, by radically simplifying the way data is presented and communicated.`

---

## **Putting It All Together**

Here’s a summary of how the steps flow:
1. **Ask the right question** – Understand the problem.
2. **Wrangle the data** – Gather, assess, and clean it up.
3. **Explore the data** – Perform EDA (visuals and stats).
4. **Draw conclusions** – Analyze findings and make sense of them.
5. **Present results** – Share insights clearly and take action.


## End-to-End Machine Learning Development Life Cycle (MLDLC)

**1. Frame the Problem (Understand the Business Need)**

   **Define Objectives:** Clearly articulate the business problem you're trying to solve and the goals of the ML project. What are the desired outcomes? What metrics will define success? <br>
   **Identify Stakeholders:** Understand who will be using or affected by the ML system and their requirements.<br>
   **Formulate the ML Problem:** Translate the business problem into a specific machine learning task (e.g., classification, regression, anomaly detection).<br>
   **Determine Success Metrics:** Define the quantitative metrics that will be used to evaluate the performance of the ML model and the overall success of the project.<br>
   **Consider Constraints:** Identify any limitations such as budget, time, data availability, performance requirements, interpretability needs, and legal or ethical considerations.

**2. Gathering Data**

   **Identify Data Sources:** Determine where the necessary data resides (databases, APIs, files, external sources, etc.).<br>
   **Collect Data:** Acquire the data from the identified sources. This might involve scraping, downloading, accessing databases, or setting up data pipelines.<br>
   **Understand the Data:** Explore the data's structure, format, size, and potential biases. Document data sources and collection methods.<br>
   **Data Governance:** Consider data privacy, security, and compliance requirements.<br>
   **🛠️ Types of Assessment(mean to find the issues in the data) in Data Analytics:** <br>
      **1. Manual Assessment**      <br>
          **Description:** Done by visually inspecting data.<br>
          **Tools:** Usually happens in tools like Excel, Google Sheets, or even raw text files.<br>
          **Often used for:** <br>
             - Small datasets
             - Quick checks
             - Spot-checking data quality
          **Examples:** <br>
             - Scanning a CSV file for missing values
             - Using filters in Excel to look for outliers
             - Manually documenting data column meanings    <br>      
      **2. Programmatic Assessment**      
          **Description:** Uses code or automated tools to assess and profile data.<br>
          **Tools:** Python (`pandas`, `pydeequ`), R, SQL, data profiling tools, or even Power BI/ Tableau data views.<br>
          **Benefits:** Scalable and more accurate for large datasets.<br>
          **Examples:** <br>
             - Writing a Python script to check null values, data types, unique values
             - SQL queries to detect duplicates or invalid data entries
             - Using `pandas profiling` or `Sweetviz` to generate automated data reports      <br>
      **🧭 Common Steps in a Data Assessment Process:**      
         These often follow this rough workflow:      <br>
         **🔍 1. Discover**      <br>
             **Goal:** Understand where the data is coming from and what data exists.<br>
             **Actions:** <br>
                - Understand data sources (source systems, APIs, databases)
                - Identify what data exists and what’s missing <br>
             **Tools:** Data catalog, SQL, API documentation      <br>
         **🗂️ 2. Document**      
             **Goal:** Record metadata and create data dictionaries.<br>
             **Actions:** <br>
                - Record metadata: column names, data types, units, value ranges, business definitions
                - Create data dictionaries or schema documentation <br>
             **Benefits:** Useful for sharing with stakeholders or future analysts     <br>  
         **🧪 3. Assess (Profile & Quality Check)**      <br>
             **Goal:** Run data profiling and check for data quality issues. <br>
             **Actions:** <br>
                - Run data profiling: Count of nulls, uniques, data types
                - Calculate descriptive statistics (mean, median, mode, etc.)
                - Check for:
                    - Missing values
                    - Duplicates
                    - Outliers
                    - Inconsistencies (e.g., "USA" vs "United States") <br>
                - **Tooling:** Use automated tools/scripts where possible      <br>
         **✅ 4. Validate**      <br>
            **Goal:** Cross-check data against expected values and business rules. <br>
            **Actions:**  <br>
                - Cross-check data with source systems or expected values
                - Confirm that the data matches the business rules     <br> 
         **📝 5. Report**      
            **Goal:** Summarize findings and suggest improvements. <br>
            **Actions:** <br>
                 - Summarize issues, observations, and risks
                 - Suggest fixes or improvements
                 - Present findings visually or in reports (e.g., using Excel, PowerPoint, or dashboards) <br>
         **✅ Data Quality Dimensions (in easy words):** <br>      
            * **Completeness** -> is data missing? <br>
            * **Validity** -> is data invalid -> negative height -> duplicate patient id <br>
            * **Accuracy** -> data is valid but not accurate -> weight -> 1kg <br>
            * **Consistency** -> both valid and accurate but written differently -> New Youk and NY <br>
            **Order of severity** <br>
               Completeness <- Validity <- Accuracy <- Consistency            <br>   
            **Data Cleaning Order** <br>
               1. Quality -> Completeness <br>
               2. Tidiness <br>
               3. Quality -> Validity <br>
               4. Quality -> Accuracy <br>
               5. Quality -> Consistency <br>
               This is the way or pattern that we can follow for our data as DA,mean first assign the label to each issue using quality dimensions and then do the cleaning by following the above order.This is not necessary but if we do wtih this we can do easily and fastly as most DA researcher and Expert do follow this. <br>
               **Steps involved in Data cleaning** <br>
                  * Define-->define the solution of each issue like if in col there is NYC and full name ,we will write the dictionary and map the value <br>
                  * Code-->now we will write the code of that solution <br>
                  * Test-->now we will test the code and result <br>
                  `Always make sure to create a copy of your pandas dataframe before you start the cleaning process` <br>
            **1. Completeness** <br>
             **What it means:** Are all the required pieces of data there? <br>
            * **Example:** If you need someone’s name, email, and phone number — but phone numbers are missing in 40% of the rows, your data isn’t complete.<br>
            * **Why it matters:** Incomplete data makes your analysis weak or misleading.<br>
            * **👉 Think of it like a pizza with missing slices 🍕 — it’s not whole.**  <br>          
            **2. Validity**
            * **What it means:** Does the data follow the rules it’s supposed to?<br>
            * **Example:** If a field says “Date of Birth” but some entries say "banana" or "13/45/2021", those are invalid.<br>
            * **Why it matters:** Invalid data messes up calculations, filters, and logic.<br>
            * **👉 It’s like putting a square peg in a round hole — it doesn’t fit the rule.**   <br>         
            **3. Accuracy**            
            * **What it means:** Is the data correct and true in the real world?<br>
            * **Example:** If someone’s email is spelled wrong or their salary says $5 instead of $50,000 — that’s inaccurate.<br>
            * **Why it matters:** Bad data = bad decisions.<br>
            * **👉 It’s like giving someone the wrong map — they’ll get lost.**  <br>          
            **4. Consistency**            
            * **What it means:** Does the data match across systems or entries?<br>
            * **Example:** If one table says someone's country is "USA" and another says "United States" or even "U.S.", that's inconsistent.<br>
            * **Or worse — their birthdate is different in two places.** <br>
            * **Why it matters:** Inconsistent data makes it hard to trust or join data from multiple places. <br>
            * **👉 Like telling two different stories — people won’t know which one is true.**

**3. Data Preprocessing**

   **Data Cleaning:** Handle missing values (imputation or removal), identify and treat outliers, correct inconsistencies, and remove duplicates.<br>
   **Data Transformation:** Convert data into a suitable format for ML algorithms. This can include scaling numerical features, encoding categorical variables, handling dates and times, and creating new features from existing ones.<br>
   **Data Splitting:** Divide the dataset into training, validation, and testing sets to train the model, tune hyperparameters, and evaluate its final performance on unseen data.<br>
   **Data Augmentation (Optional):** For tasks like image or text classification, generate synthetic data to increase the size and diversity of the training set.

**4. Exploratory Data Analysis (EDA)**

   **Visualize Data:** Use charts, graphs, and other visual techniques to understand the data's characteristics, identify patterns, relationships between variables, and potential issues.<br>
   **Statistical Analysis:** Calculate descriptive statistics (mean, median, standard deviation, etc.) and perform statistical tests to gain insights into the data.<br>
   **Identify Potential Features:** Explore which features might be most relevant for the ML task.<br>
   **Uncover Data Quality Issues:** Identify any remaining problems with the data that need to be addressed.

**5. Feature Engineering and Selection**

   **Feature Engineering:** Create new features from existing ones that might improve model performance based on domain knowledge and insights from EDA. This can involve combining features, creating polynomial features, or transforming features in other ways.<br>
   **Feature Scaling:** Apply scaling techniques (e.g., standardization, normalization) to ensure features have comparable ranges.<br>
   **Feature Selection:** Identify and select the most relevant features for the model, reducing dimensionality and potentially improving performance and interpretability. 
    Techniques include statistical methods, model-based selection, and dimensionality reduction techniques like PCA.
   #### 📊 Understanding Variance, Covariance, and PCA in Feature Selection – Step by Step

---

#### 🔹 1. Variance – Looking at a Single Feature (One Axis)

When we look at a **single feature** (one axis), **variance** tells us how much the data is spread out along that axis.

- If the data is really spread out, it means that feature likely holds **a lot of information**.
- A highly variable feature can help us **distinguish between different data points**.

👉 So, for selecting important individual features, a **high variance is often a good sign**.

---

#### 🔹 2. Covariance – Looking at Two Features (Two Axes)

When we consider **two features** (two axes), **covariance** comes into play.

- It tells us **how those two features change together**.
- If the data spreads out **diagonally across the two axes**, it suggests the features are **related or correlated**.

> If we had to pick just **one of these related features** for simplicity, we’d look at how much the data spreads along each individual axis (their **variances**) to decide **which one captures more of the overall spread**.

---

#### 🔹 3. PCA – Scaling Up to Many Features (Multiple Axes)

**PCA (Principal Component Analysis)** takes this concept further to handle **multiple features (dimensions)**.

It tries to find a **new set of axes**, called **principal components**, that align with the directions of **maximum variance** in the data.

---

#### 🔹 4. Principal Components – The New Directions

- The **first principal component** is the **single direction** that captures the **most spread** (maximum variance) in the entire dataset.
- The **second principal component** captures the **next most spread**, but is **independent (uncorrelated)** from the first.
- This process continues, with each new component being orthogonal (perpendicular) to the previous ones.

---

#### ✅ Summary

PCA builds on:
- **Variance** → to find the most informative directions.
- **Covariance** → to understand feature relationships and avoid redundancy.

And transforms your high-dimensional data into a **simpler, compact form** with minimal information loss.


#### 🧠 Key Idea of PCA   
   PCA does **not care about the original axes (X or Y)** — instead, it finds a **new axis (direction)** that:
   
   - ✅ Maximizes the **spread (variance)** of the data when projected onto it.
   - 📌 This new axis is called the **first principal component**.
#### Principal Component Analysis (PCA)

**Principal Component Analysis (PCA)** is a statistical technique used for:

#### 1. Dimensionality Reduction

- Reduces the number of features (dimensions) while preserving as much variance (information) as possible.
- Helps with:
  - Speeding up machine learning algorithms
  - Reducing overfitting
  - Simplifying complex datasets

#### Example:
High-dimensional data (e.g., 100 features) → PCA → Reduced to 10 features that still capture ~95% of the original variance.

---

#### 2. Visualization

- Projects high-dimensional data down to **2D or 3D** for easy visualization.
- Useful for:
  - Exploratory data analysis
  - Visualizing clusters or patterns
  - Understanding data structure

#### Example:
Reduce to 2 principal components → Plot as a 2D scatter plot to visualize data distribution or clustering.

---

#### Summary

| Purpose               | Goal                                      | Output                     |
|------------------------|---------------------------------------------|-----------------------------|
| Dimensionality Reduction | Fewer features, same variance               | Compressed data             |
| Visualization           | Understand high-dimensional structure       | 2D or 3D scatter plots      |

> ✅ PCA is often used as a **preprocessing step** before clustering or classification tasks.
#### Finding the Optimum Number of Principal Components in PCA

In Principal Component Analysis (PCA), it's important to choose the right number of principal components (PCs) to retain most of the data's variance while reducing dimensionality. Here's how you do it:

---

#### 🌟 Core Idea

# 📌 Choosing the "Best" Number of Principal Components in PCA

Okay, so imagine each of those eigenvalues (λ₁, λ₂, λ₃, ..., up to λ₇₈₄ in this example) represents the **'importance'** of its corresponding principal component. A **bigger eigenvalue** means that principal component captures **more of the overall spread or variance** in our original data.

---

#### 🧠 Visual Analogy: The Messy Room

Think of it like this:

- You walk into a **messy room**.
- The **first principal component** might be the **main direction** the mess is oriented – like a **big pile leaning one way**.
- The **second principal component** would be the **next most significant direction of mess**, maybe a scattering of items **perpendicular** to the main pile.
- And so on...

Each principal component finds a different **direction** of variation (or “mess”) in your high-dimensional data.

---

#### 📏 Raw Eigenvalues Aren't Percentages

Now, these eigenvalues themselves aren’t percentages. They’re just **raw values** showing how much variance each component explains.

To understand **how much of the total mess** each direction accounts for, we need to convert them into **percentages**.

---

#### 📐 Here's the Formula:

\[
\left( \frac{\lambda_i}{\lambda_1 + \lambda_2 + \lambda_3 + \cdots + \lambda_{784}} \right) \times 100
\]

---

#### 🔍 Let's Break It Down:

- **λᵢ (the numerator):**  
  This is the eigenvalue of a specific principal component (e.g., the first one, λ₁).  
  It tells us how much variance that one component captures.

- **λ₁ + λ₂ + λ₃ + ... + λ₇₈₄ (the denominator):**  
  This is the **sum of all the eigenvalues** — the total variance present in the original data.  
  Think of it as the **total amount of "mess"** in the entire room.

- **Dividing λᵢ by the sum:**  
  This gives the **proportion of total variance** that PCᵢ explains.  
  It’s like asking,  
  > “What fraction of the total mess does this specific direction account for?”

- **Multiplying by 100:**  
  Converts the proportion into a **percentage**, making it easier to interpret.  
  So we can say,  
  > “The first principal component explains X% of the total variance in the data.”

---

#### 🎯 Goal: Find the "Optimum" Number of Principal Components

The goal of PCA isn’t just to calculate these percentages — it’s to **reduce the number of dimensions** while keeping **most of the important information**.

Usually, we want to **capture a significant amount of the total variance**, like **90%**, while discarding the rest.

---

#### 📊 Example: Eigenvalues [30, 25, 15, 10, 5...]

Let’s say you have a sequence of eigenvalues:

λ₁ = 30
λ₂ = 25
λ₃ = 15
λ₄ = 10
λ₅ = 5 and so on upto n.

Assuming the total variance (sum of eigenvalues) is 100:

- **PC₁:** 30 / 100 = 30%
- **PC₁ + PC₂:** (30 + 25) / 100 = 55%
- **PC₁ + PC₂ + PC₃:** (30 + 25 + 15) / 100 = 70%
- Continue adding until you reach **90% cumulative variance**.

> In the image/example, it looks like **15 principal components (λ₁ to λ₁₅)** might be enough to reach 90%.  
> That means we can reduce from **784 dimensions** down to just **15**, while still keeping most of the important patterns in the data.

---

#### ✅ Summary — The Main Idea in a Nutshell:
`Here's the main idea in a nutshell:
Eigenvalues represent importance: Each eigenvalue (λ) associated with a principal component tells us how much variance (spread or information) that specific component captures from the original data. Larger eigenvalues mean more important components. 
Convert to percentage: To understand the proportion of the total variance explained by each PC, we convert the eigenvalues into percentages. This is done by dividing each individual eigenvalue by the sum of all eigenvalues and then multiplying by 100.   
Determine the "optimum" number: The goal is to select a smaller number of principal components that still retain a significant portion of the total variance (e.g., 90%). You do this by looking at the cumulative percentage of variance explained as you include more principal components. You stop when you reach a satisfactory level, achieving dimensionality reduction while preserving most of the important information in the data.`
- **Eigenvalues represent importance:**  
  Each eigenvalue (λ) associated with a principal component tells us how much **variance** that component captures. Bigger λ = more important component.

- **Convert to percentage:**  
  To understand how much **total variance** each PC explains, divide the eigenvalue by the **sum of all eigenvalues** and multiply by 100.

- **Determine the "optimum" number:**  
  Add up the percentages until the **cumulative variance** explained reaches your threshold (like **90%**).  
  Keep just those PCs — you’ve reduced the dimensions while keeping most of the useful information!

> 🎓 So in essence, we're just using eigenvalues to measure how much each PC explains the spread in the data, and then we pick the smallest set of PCs that explain enough of that spread to confidently move forward.


---

#### 3. **Determine the "Optimum" Number of PCs**
- Add up the variance percentages to get the **cumulative variance explained**.
- Choose the smallest number of PCs that together explain a **sufficient amount of variance** (commonly **90% or 95%**).
- This is where you **"cut off"** and keep those components for your reduced dataset.

---

#### 📈 Example Visualization (Scree Plot)

Plot the variance explained by each component:

- **X-axis**: Principal Component number  
- **Y-axis**: % of variance explained  
- Look for the **"elbow" point** or the point where cumulative variance crosses your target threshold (e.g., 90%).

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Fit PCA to your data
pca = PCA()
pca.fit(X)  # X is your input data

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Scree Plot
plt.figure(figsize=(8,5))
plt.plot(range(1, len(explained_variance)+1), cumulative_variance, marker='o', linestyle='--')
plt.axhline(y=0.9, color='r', linestyle='-')  # 90% threshold
plt.title('Cumulative Explained Variance by PCA')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained')
plt.grid(True)
plt.show()
```

   ---
   
   ## 🔍 What Does "Spread" Mean?
   
   When we talk about **spread** or **variance**, imagine projecting all your 2D data points onto a line (like casting shadows). PCA looks for the line where the **projected points are most spread out**.
   
   #### 🧭 Examples:
   
   - If most of the variation (distance between projected points) is **along the X-axis**,  
     → PCA chooses a direction **closer to the X-axis**.
   
   - If most of the variation is **along the Y-axis**,  
     → PCA chooses a direction **closer to the Y-axis**.
   
   - If the data is spread diagonally (e.g. both X and Y increase together),  
     → PCA finds a new axis **diagonal to both X and Y** (e.g. 45° line).
   
   > 📌 Note: PCA considers the **overall distribution of all data points**,  
   > not just the distance between the first and last point.
#### 📊 Example: Diagonal Spread in PCA

```
X: [1, 2, 3, 4, 5]  
Y: [1, 2, 3, 4, 5]
```

- ✅ Both X and Y **increase together**  
- 📈 The data forms a **perfect diagonal line (45°)**  
- 🔁 This means **maximum variance lies along the diagonal**, not strictly on X or Y  
- ✅ **PCA picks a new axis** (principal component) aligned with this diagonal direction  
  → A new rotated axis that captures the most spread

---

#### 📐 How PCA Decides the Direction (Step-by-Step)

1. **Compute the Covariance Matrix**  
   Captures how variables change with each other.

2. **Find Eigenvectors and Eigenvalues**  
   - **Eigenvectors** → Directions (possible new axes)  
   - **Eigenvalues** → Variance captured along each eigenvector

3. **Sort Eigenvectors by Eigenvalues (Descending Order)**  
   - Higher eigenvalue = more spread (information) in that direction
     `An eigenvector of a square n×n n × n matrix A is a non-zero vector x such that, when x is multiplied on the left by A , it yields a constant multiple of x . That is: Ax=λx. A x = λ x . The number λ is called the eigenvalue of A corresponding to the eigenvector x.this mean that matrix or a number which is most of the time said scalar ,so as this is applied linear transformation or matrix it changes the magnitude not direction with respect to spread of data points or values,so matrix applying is simple mean scalar which is changing the scale/magnitude of the eigen vectors.`
![image](https://github.com/user-attachments/assets/e8298f39-775b-4662-9f48-0012cb11dec3)

In the above image, a bunch of data points scattered in a **multi-dimensional space**, and **PCA** is like finding the most important **"directions"** in that space that capture the **most spread (variance)** of your data.

---

#### 🧩 Step-by-Step Breakdown

#### 🔹 **The Initial Data (Top Left)**

- You have a **3D coordinate system** with features `f1`, `f2`, and `f3`.
- Red '❌' marks represent **data points**, each located by its values for the 3 features.
- The **wiggly arrows** on axes show the **spread (variance)** along each feature axis.

➡️ You can **visually guess** where the data is more spread (more variance).

---

#### 🔹 **Step 1: Mean Centering (Top Right)**

- The dataset is shifted so its **center (mean)** is now at the origin (0,0,0).
- This is shown by `x̄` indicating the mean-centered data.

✅ **Why?**  
PCA focuses on **variance**, and mean-centering makes it easier to calculate and interpret **covariance**.

---

#### 🔹 **Step 2: Covariance Matrix (Middle Left)**

- From the mean-centered data, we compute the **3x3 covariance matrix**:

```
       f1     f2     f3
     ---------------------
f1 |  V(f1)  C(f1,f2) C(f1,f3)
f2 |  C(f2,f1) V(f2)  C(f2,f3)
f3 |  C(f3,f1) C(f3,f2) V(f3)
```

- **Diagonal** → Variance of each feature  
- **Off-diagonal** → Covariance between feature pairs

✅ **What it tells us:**  
How much features vary **with themselves** and **with each other**.

---

#### 🔹 **Step 3: Eigenvectors & Eigenvalues (Top & Bottom Right)**

- From the covariance matrix, we compute:

#### 📍 **Eigenvectors**
- New **directions (axes)** in space that represent the **strongest variance**.
- These are the **Principal Components (PC1, PC2, PC3)**.
- They are **orthogonal** (perpendicular to each other).

#### 📍 **Eigenvalues (λ₁, λ₂, λ₃)**
- Represent **how much variance** each eigenvector captures.
- Higher eigenvalue → more important that direction is.

---

#### 🔽 **Dimensionality Reduction (Bottom)**

You can reduce dimensions using the **principal components**:

- **(PC1)**: The direction (eigenvector) with the **largest eigenvalue (λ₁)**  
  → Projecting data onto PC1 gives a **1D representation** with most variance retained.

- **(PC1, PC2)**: Using top 2 eigenvectors  
  → Gives a **2D projection** that captures the **next most variance**.

- **(PC1, PC2, PC3)**: Use all → You’re back to **3D**, but now aligned with the principal directions.

---

#### 🧠 TL;DR – What PCA Does

1. 🎯 **Centers your data** so the average is zero.
2. 📊 **Calculates covariance** to understand how features vary together.
3. 🔁 **Finds eigenvectors & eigenvalues**:
   - Eigenvectors = new axes (principal components)
   - Eigenvalues = importance (variance captured)
4. 🔽 **Reduces dimensions** (if needed) by keeping only the top components with the most variance.

---

#### 🧩 Real-World Summary

| Step                | What It Means                                    |
|---------------------|--------------------------------------------------|
| Mean Centering      | Move the data cloud so the center is at origin   |
| Covariance Matrix   | Measure how features vary with each other        |
| Eigenvectors        | New axes (directions) showing max variance       |
| Eigenvalues         | How important each direction is (spread amount)  |
| Dim Reduction       | Drop less important directions (low variance)    |

---

🧠 **Insight**:  
> PCA is like rotating your view of the data to look down the most informative directions.
     
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Simulated 3D dataset (points lie roughly along a diagonal in 3D space)
np.random.seed(42)
n_points = 100
X = np.random.normal(0, 1, (n_points, 1))
Y = X * 0.8 + np.random.normal(0, 0.1, (n_points, 1))
Z = X * 0.6 + np.random.normal(0, 0.1, (n_points, 1))

data = np.hstack([X, Y, Z])

# Apply PCA
pca = PCA(n_components=3)
pca.fit(data)
components = pca.components_
mean = np.mean(data, axis=0)

# Plotting the original 3D data and the principal components
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.6, label='Data Points')

# Plot the principal components
for i in range(3):
    vector = components[i]
    ax.quiver(*mean, *(vector * 2), color='r', label=f'PC{i+1}')

ax.set_title('3D Data with Principal Components')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.legend()
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/8a4eec7e-d67a-4947-bde1-318bb280db4e)


4. **Select the Top Eigenvector**  
   - This becomes the **first principal component** (PC1)  
   - Captures the **maximum variance** in the data
#### 📐 Variance, Covariance & Correlation – Explained Simply

---

#### 🔹 Variance

- Measures **how much the data spreads out** from the mean **on a single axis**.
- In PCA, variance tells us how much info (spread) a principal component captures.
- **Projection**: It's like dropping shadows of points on a direction (unit vector) and measuring how spread out the shadows are.

#### Example:

```
X = [1, 2, 3, 4, 5] → Mean = 3
Variance(X) = Average of squared distance from mean = high
```

➡️ Data is spread widely on X → PCA may pick this direction.

---

#### 🔹 Covariance

- Measures **how two variables change together**.
  - Positive → both increase together
  - Negative → one increases, the other decreases
- Covariance is key in PCA — it's used to see how features interact.

#### Example:

```
X = [1, 2, 3, 4, 5]
Y = [2, 4, 6, 8, 10]
Cov(X, Y) = High positive (they rise together)
```

➡️ Indicates strong relationship → diagonal direction is important in PCA.

---

#### 🔹 Correlation

- Like covariance, but **normalized between -1 and 1**.
- Shows **strength and direction** of a linear relationship:
  - +1 → perfectly positive
  - -1 → perfectly negative
  - 0 → no linear relation

#### Example:

```
X = [1, 2, 3, 4, 5]
Z = [10, 8, 6, 4, 2]
Correlation(X, Z) = -1
```

➡️ As X increases, Z decreases → perfect negative relationship.

---

#### 📊 Visualization (Concept)

Imagine the points plotted in 2D:

- **X vs Y**: points rise diagonally → max variance along 45° line.
- **X vs Z**: points fall diagonally → max variance along -45° line.

In both cases, PCA finds the direction (not necessarily X or Y) where the data is **most spread out**, and that becomes the **principal component**.

> 💡 PCA uses **variance of projections** and **covariance between axes** to decide the best direction. Correlation helps understand relationships when variables are on different scales.

---

#### 🎯 Summary Table

| Concept     | Measures                     | Axes | Range       | Used in PCA? | Key Use                      |
|-------------|------------------------------|------|-------------|--------------|------------------------------|
| Variance    | Spread on one axis           | 1    | ≥ 0         | ✅ Yes       | Find spread along a vector   |
| Covariance  | Joint variability (X & Y)    | 2    | -∞ to ∞     | ✅ Yes       | Build covariance matrix      |
| Correlation | Strength + direction (scaled)| 2    | [-1, +1]    | ❌* (if unscaled) | Compare feature relationships |

---

*Note: PCA uses **correlation matrix instead of covariance matrix** when features are on different scales (standardized).




**6. Model Training, Evaluation, and Tuning**

   **Select Model(s):** Choose appropriate ML algorithms based on the problem type, data characteristics, and performance requirements.<br>

#### 🧠 How to Understand Any Math Formula Behind the Algorithm in Machine Learning — Step by Step

For Example if You're doing great by trying to **understand the logic and derivation** behind ML formulas of Linear Regression instead of just memorizing them. Here's a structured way to analyze **any equation**, just like how you did for linear regression.

---

#### ✅ Goal

> Build a step-by-step method to understand, break down, and interpret mathematical formulas and algorithms — especially those used in machine learning.

---

### 🔁 Math Deconstruction Strategy

#### 🔹 Step 1: **Identify the Real-World Context**

Ask: **What is this formula solving or predicting?**

- **Example**: In linear regression, we want to **predict a continuous variable (y)** from an input (x).

---

#### 🔹 Step 2: **Understand Each Variable**

Create a quick reference table for symbols:

| Symbol         | Meaning                                  |
|----------------|-------------------------------------------|
| \( x_i \)      | i-th input feature (independent variable) |
| \( y_i \)      | i-th actual output (dependent variable)   |
| \( \hat{y}_i \) | Predicted value \( \hat{y} = mx + b \)    |
| \( m \)        | Slope (rate of change of y with respect to x) |
| \( b \)        | Intercept (value of y when x = 0)         |

---

**Define the Error (Loss) Function**

Error = $\sum_{i=1}^{n} (y_i - \hat{y}_i)^2$

---

**Plug in the Prediction Formula**

Since $\hat{y}_i = mx_i + b$, substitute:

Error = $\sum_{i=1}^{n} (y_i - (mx_i + b))^2$
This is your **Loss Function (MSE)** — it measures how far off your predictions are.

---

#### 🔹 Step 3: **Break the Formula Down ,apply the Calculus to Minimize the error**

This is the **Mean Squared Error (MSE)** — it tells us how off our predictions are.

To minimize error, we take **partial derivatives** with respect to $m$ and $b$, set them to 0, and solve.

---

#### Final Formulas for Slope and Intercept

**Slope (m):**

$m = \dfrac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}$ <br>
**Intercept (b):**
$b = \bar{y} - m\bar{x}$

---

#### 🔹 Step 4: **Apply Calculus/Algebra Step-by-Step**

We want to **minimize the error** by adjusting **m** and **b**.
So we:
- Take partial derivatives of the error function w.r.t. \( m \) and \( b \)
- Set derivatives to 0 to find minima
- Solve equations to get:
This gives us the **best-fitting line** for the data.

---

#### 🔹 Step 5: **Understand the Mathematical Choices**

Ask: Why use squares instead of absolute values?

- Squares are **differentiable**, needed for calculus (gradient-based optimization)
- They **penalize large errors** more heavily
- Absolute values aren't smooth at 0 — makes differentiation harder

---

#### 🔹 Step 6: **Relate Back to the Real World**

Interpret the results:

- \( m \): How steep the line is (slope)
- \( b \): Where the line crosses the y-axis (intercept)
- These are **learned from real data** using math, not guessed

#### ✅ General Framework (like you did for linear regression)
| 🔢 Step | What to Ask | What You Did in Linear Regression |
|--------|-------------|-----------------------------------|
| 1️⃣ **Start from the main equation** | What is the base formula that defines the model or objective? | You started from `ŷ = mx + b` |
| 2️⃣ **Define each variable** | What does each symbol mean in real-world terms? | x = input, y = actual output, ŷ = predicted output |
| 3️⃣ **Write the goal** | What are we trying to optimize or calculate? | Minimize squared error between y and ŷ |
| 4️⃣ **Plug in what you know** | Use substitutions, e.g. put `ŷ = mx + b` into the error formula | Error = ∑(yi - (mxi + b))² |
| 5️⃣ **Apply calculus or algebra** | Use differentiation or manipulation to isolate the unknowns | Took derivative w.r.t m and b, solved for both |
| 6️⃣ **Interpret the final formula** | What do the resulting formulas mean in practical terms? | m = slope, b = intercept — model line of best fit |
| 7️⃣ **Understand why that method is used** | Why squares (instead of mod)? Why use derivatives? | Squared error is differentiable and penalizes larger errors more |

---

#### 🧠 Use This On Any New Formula

Ask yourself:
1. **What does this equation represent?**

2. **What does each variable mean in the real world?**

3. **What is the goal — minimize? maximize? classify?**

4. **How does it get computed? What’s being plugged in?**

5. **What math tools are being used — derivatives? sums? eigenvectors?**

6. **What do the final terms mean — slope? boundary? direction of variance?**

7. **Why is this method better — e.g. log-loss vs squared error?**


---

#### 🧠 Memory Trick: `D.E.C.O.D.E.R.`

| Letter | Meaning |
|--------|---------|
| **D** | **Define the formula** (what does it do?) |
| **E** | **Explain each variable** (real-world terms) |
| **C** | **Connect it to the goal** (minimizing error, etc.) |
| **O** | **Operate** the formula (substitute, expand) |
| **D** | **Differentiate or Derive** (if needed) |
| **E** | **Evaluate meaning of result** (interpret formula) |
| **R** | **Reason why this method** (why this technique is used) |

---

#### 🎯 General Checklist for Any Formula

| Step | What to Ask                            | Example (Linear Regression)        |
|------|----------------------------------------|------------------------------------|
| 1️⃣   | What's the purpose of this formula?     | Predict y from x                   |
| 2️⃣   | What does each symbol represent?       | x: input, y: output, m/b: parameters |
| 3️⃣   | Can I break it into smaller parts?     | Expand and simplify loss function  |
| 4️⃣   | What operation is being used? Why?     | Derivatives for optimization       |
| 5️⃣   | What does the final result mean?       | Slope/intercept of best-fit line   |

---

#### 🔄 Try This Strategy On:

- **Logistic Regression**: Understand sigmoid + cross-entropy loss
- **Gradient Descent**: See how weights update step-by-step
- **K-Means Clustering**: Interpret the distance + centroid update logic
- **Neural Networks**: Apply it to forward and backpropagation
---

   **Train Model(s):** Train the selected model(s) using the training data.<br>
   **Evaluate Model(s):** Assess the performance of the trained model(s) on the validation set using the chosen success metrics.<br>
   **Hyperparameter Tuning:** Optimize the model's hyperparameters using techniques like grid search, random search, or Bayesian optimization to achieve the best performance on the validation set.<br>
   **Model Selection:** Choose the best-performing model based on the evaluation results on the validation set.<br>[Hypothesis Testing](https://media.licdn.com/dms/document/media/v2/D4E1FAQG7DypLIylbmw/feedshare-document-pdf-analyzed/B4EZXSSaTuG0AY-/0/1742989815533?e=1745452800&v=beta&t=1Ab98o5m_sm6V1O7JMD24uxeK2R0NkqkL21B4O35AHA), 
   [Data Cleaning-Pipeline](https://media.licdn.com/dms/document/media/v2/D4E1FAQGRmsf_NRKd1g/feedshare-document-pdf-analyzed/B4EZYaAwW7HMAY-/0/1744193144485?e=1745452800&v=beta&t=Tfjy8CdfwDRSJb3sFhy4ctiQr6OQHGiVYdK6rjSYXOY), [Supervised Learning-Part-1](https://media.licdn.com/dms/document/media/v2/D4E1FAQEWkXpgbDTMUw/feedshare-document-pdf-analyzed/B4EZYkePEmHkAc-/0/1744368647089?e=1745452800&v=beta&t=V_sjpjyWvYhedxd0m0P3Pl_KXP7jv9f_0T_yPeOwijM), [Supervised Learning Models-Part-2](https://media.licdn.com/dms/document/media/v2/D4E1FAQHExMVYi9daXA/feedshare-document-pdf-analyzed/B4EZYz07T2HMAY-/0/1744626255012?e=1745452800&v=beta&t=H23uzJltP8seoNU-OOUhhr-IKUPaXos35mOam-mWO14), [SL-Part-3](https://media.licdn.com/dms/document/media/v2/D4E1FAQEoBnvJxqTbmQ/feedshare-document-pdf-analyzed/B4EZYVbaegHcAc-/0/1744116242941?e=1745452800&v=beta&t=naL83bHmWfB-6SBugtXmFfNgaLwfSOOas_kjPdYQYkk),<br> [UnSupervised Learning-Part-1](https://media.licdn.com/dms/document/media/v2/D4E1FAQFh63wlB63j6A/feedshare-document-pdf-analyzed/B4EZX7FS.FHcAo-/0/1743674242420?e=1745452800&v=beta&t=52lelWSXEci9xsJfdSJLu6q6BxyR5PUuzLj6MrWIVAQ) , 
   [Business Analytics-Part-1](https://media.licdn.com/dms/document/media/v2/D4E1FAQHwey8DoVmmyw/feedshare-document-pdf-analyzed/B4EZXHx22ZG0AY-/0/1742813498965?e=1745452800&v=beta&t=oMvu8J65rx-79PFSxpEty1axiI-B7vXFpJkN0RtDCu0), [BA-Part-2](https://media.licdn.com/dms/document/media/v2/D4E1FAQFNQ_trs718lA/feedshare-document-pdf-analyzed/B4EZW4BY_vHgAY-/0/1742549142065?e=1745452800&v=beta&t=gz42Ql7eVUwkB4NVtRoaHqt_3ei5OOzHBZqvVqbTB84), [BA-Part-3](https://media.licdn.com/dms/document/media/v2/D4E1FAQEGjm_yOKF1iQ/feedshare-document-pdf-analyzed/B4EZWtxhcuH0AY-/0/1742377206901?e=1745452800&v=beta&t=NMxWYnwgoVCjPurKGftx3oC5a6V7vEug2a8vjX-clIA), [BA-Part-4](https://media.licdn.com/dms/document/media/v2/D4E1FAQGEDljjcjSAHQ/feedshare-document-pdf-analyzed/B4EZW9bOpVH0AY-/0/1742639792784?e=1745452800&v=beta&t=PqJbggVe10JjYBTG13F_OQUJJlpBrV23X0EhSEoEz1A)

**7. Model Deployment**

   **Choose Deployment Strategy:** Decide how the model will be integrated into the existing system or made accessible (e.g., API, embedded system, batch processing).<br>
   **Build Deployment Infrastructure:** Set up the necessary hardware, software, and infrastructure to host and run the model.<br>
   **Deploy the Model:** Implement the chosen deployment strategy and make the model available for use.<br>
   **Monitoring and Logging:** Implement systems to track the model's performance, identify issues, and log predictions and relevant data.

**8. Testing (Model Validation and Acceptance)**

   **Evaluate on Test Set:** Evaluate the final selected model on the held-out test set to get an unbiased estimate of its performance on unseen data.<br>
   **User Acceptance Testing (UAT):** Involve end-users to test the deployed system and ensure it meets their needs and expectations.<br>
   **Performance Testing:** Evaluate the system's speed, scalability, and stability under realistic load conditions.

**9. Optimize (Maintenance and Improvement)**

   **Monitor Performance:** Continuously track the model's performance in the production environment.<br>
   **Retrain Model:** Periodically retrain the model with new data to maintain its accuracy and adapt to changing patterns.<br>
   **Model Updates:** Update the model if performance degrades significantly or if better models become available.<br>
   **Iterate and Improve:** Continuously look for ways to improve the model, the data pipeline, and the overall ML system based on monitoring, feedback, and new insights.

This detailed breakdown provides a more comprehensive understanding of the steps involved in an end-to-end machine learning development life cycle. Remember that this is an iterative process, and you might need to revisit earlier steps as you progress through the cycle.

## 1. Feature Engineering (Takes 30% of Project Time)
   **a) EDA**
     - i)   Analyze how many numerical features are present using histogram, pdf with seaborn, matplotlib.
     - ii)  Analyze how many categorical features are present. Is multiple categories present for each feature?
     - iii) Missing Values (Visualize all these graphs)
     - iv)  Outliers - Boxplot
     - v)   Cleaning
     
   **b) Handling the Missing Values**
     - i)   Mean/Median/Mode
     
   **c) Handling Imbalanced dataset 
   d) Treating the Outliers
   e) Scaling down the data - Standardization, Normalization
   f) Converting the categorical features into numerical features**

## 2. Feature Selection
   **a) Correlation
   b) KNeighbors
   c) ChiSquare
   d) Genetic Algorithm
   e) Feature Importance - Extra Tree Classifiers**
   
## 3. Model Creation
**4. Hyperparameter Tuning
5. Model Deployment
6. Incremental Learning** <br>
`Numerical features may be there, categorial features, missing values, visualise, outliers box plot, cleaning
Step 2 handling missing values by mean, box plot iqr remove, handling imbalance dataset, treating outliers, scaling data standarisation and normalisation, categorical to numerical features`
## Links for the Feature Engineering and EDA
- **Pre-Requesite to watch or to proceed ahead** : <br>
  - [Python](https://roadmap.sh/python)
  - [libraries pandas,numpy,seaborn,matplotlib](https://moonlighto2.medium.com/key-python-libraries-for-data-analysis-and-code-examples-f15c8a2349c1)
  - [Mathematics](https://www.youtube.com/watch?v=OmJ-4B-mS-Y&pp=ygULbWF0aGVtYXRpY3M%3D),[Must Know this Math if Coming From non-Tech Background](https://www.youtube.com/watch?v=0KQ1Vudz2GM&list=PLySt0K5r-RfWAuq2QijIAlkIGg5vtT4eW), [Math for Machine Learning](https://www.youtube.com/watch?v=yDzJ4tgaN7A&t=10s&pp=ygULbWF0aGVtYXRpY3M%3D),  [Linear Algebra](https://www.youtube.com/watch?v=mQewAJb8oJ8&pp=ygULbWF0aGVtYXRpY3M%3D)
  - [Probablity-Part-1](https://drive.google.com/file/d/1tH8hrTpU3SG-b6v3ecWxwSu1FRENeP6S/view), [Statistics](https://www.youtube.com/watch?v=L_OLifCqxCQ&pp=ygULbWF0aGVtYXRpY3M%3D),[Statistics Notes](https://media.licdn.com/dms/document/media/v2/D4D1FAQFuOm937gRjMw/feedshare-document-pdf-analyzed/B4DZY39UKZHIAg-/0/1744695981682?e=1745452800&v=beta&t=UQle9gc9EIFV_ENiMYyzd6Vz65K1_rS31SRYM0eMymI),  [Best to Know Statistics](https://www.youtube.com/watch?v=S7LvZZNq4ys&pp=ygULbWF0aGVtYXRpY3M%3D),  [Better to know Statistics](https://www.youtube.com/watch?v=bLZ-LSsQMCc&pp=ygULbWF0aGVtYXRpY3M%3D), [Learn More about Statistics](https://www.youtube.com/watch?v=LZzq1zSL1bs)
  - [Feature Engineering](https://www.youtube.com/watch?v=6WDFfaYtN6s&list=PLZoTAELRMXVPwYGE2PXD3x0bfKnR0cJjN) <br>
  - [EDA Playlist](https://www.youtube.com/watch?v=ioN1jcWxbv8&list=PLZoTAELRMXVPQyArDHyQVjQxjj_YmEuO9) And [Live session EDA playlist](https://www.youtube.com/watch?v=bTN-6VPe8c0&list=PLZoTAELRMXVPzj1D0i_6ajJ6gyD22b3jh) --- Know More about EDA Automatic using Libraries---> [EDA in Minutes](https://www.youtube.com/watch?v=AYalukmWroY), [EDA libraries at One Place](https://www.youtube.com/watch?v=BoKLMehRahw),[EDA using SweetViz Library](https://www.youtube.com/watch?v=D4fHn4lHCmI),[EDA using dtale library](https://www.youtube.com/watch?v=xSXGcuiEzUc), [Automatic EDA using Pandas-Profiling Library](https://www.influxdata.com/blog/pandas-profiling-tutorial/), [Dataprep for EDA](https://pypi.org/project/dataprep/), [Autoviz for EDA](https://pypi.org/project/autoviz/), [Pandas Visual analysis ](https://pypi.org/project/pandas-visual-analysis/)<br> 
  - [Feature Selection](https://www.youtube.com/watch?v=uMlU2JaiOd8&list=PLZoTAELRMXVPgjwJ8VyRoqmfNs2CJwhVH)
  - [Machine Learning](https://www.youtube.com/watch?v=z8sxaUw_f-M&list=PLZoTAELRMXVPjaAzURB77Kz0YXxj65tYz), [ML](https://www.youtube.com/watch?v=1ctqJCHMAmc),  [Blogs for Machine Learning](https://intellipaat.com/blog/tutorial/machine-learning-tutorial/)
  - [Deep Learning](https://www.youtube.com/watch?v=G1P2IaBcXx8&pp=ygUNZGVlcCBsZWFybmluZw%3D%3D), [DL](https://www.youtube.com/watch?v=d2kxUVwWWwU&pp=ygUPY29tcHV0ZXIgdmlzaW9u)
  - [NLP](https://www.youtube.com/watch?v=ENLEjGozrio)
  - [Computer vision or CV](https://www.youtube.com/watch?v=E-HSXRvL9Ik), [Blog for the CV](https://intellipaat.com/blog/what-is-computer-vision/)
  - [AI Course Blogs](https://intellipaat.com/blog/tutorial/artificial-intelligence-tutorial/)

### For Practical see the link
https://drive.google.com/drive/folders/1va6VQ7qCsqS5vqOAALAS5Y8QVys9nnwR?usp=sharing <br>
### 1. Understanding Your Data

#### Data Types and Summary Statistics
- **Numerical Variables**: These are quantitative and can be analyzed with mathematical operations. Examples include age, income, and temperature.
  - Use `df.describe()` to get summary statistics like mean, median, standard deviation, etc.
  
- **Categorical Variables**: These represent categories or groups. Examples include gender, country, and product type.
  - Use `df.info()` to understand the data types and identify categorical columns.
  - Use `df['column_name'].value_counts()` to see the distribution of categories.

- **Date/Time Variables**: These represent timestamps and can be used for time-series analysis.
  - Convert to datetime format using `pd.to_datetime(df['column_name'])`.
- **A random variable**: is not random itself, but rather a function that applies logic to events from the sample space and gives a numerical representation for further analysis.
#### Missing Data
- Identify missing values using `df.isnull().sum()`.
- Decide whether to fill missing values (e.g., with the mean or median), impute them (e.g., using forward fill), or drop rows/columns with missing data.

#### Visualization
- Use histograms to see the distribution of numerical data.
- Use boxplots to identify outliers and understand the spread of data.
- Use scatter plots to explore relationships between two numerical variables.
- Use bar plots for categorical data to compare frequencies.

### 2. Feature Engineering: What to Compare?

#### Exploratory Data Analysis (EDA)
- Ask questions about each column and how it might relate to other variables.
- Use pairwise scatter plots (pairplots) to visualize relationships between multiple variables.

#### Correlation
- Use `df.corr()` to compute pairwise correlation of columns.
- Visualize with a heatmap to quickly identify strong relationships.

### 3. What to Perform on the Data?

#### Cleaning
- **Handle Missing Values**: Decide on imputation or removal based on the context and importance of the data.
- **Handle Outliers**: Identify and decide whether to keep, transform, or remove them.
- **Handling the Imbalanced Data**: check whether data has equality in row or column wise and has the same scale of values.
- **Encoding Categorical Variables**: Convert categorical variables to numerical using techniques like one-hot encoding or label encoding.

#### Statistical Analysis
- **T-test**: Use to compare means between two groups.
- **Chi-square Test**: Use to test for associations between categorical variables.
- **Correlation Analysis**: Understand linear relationships between numerical variables.

#### Modeling
- **Regression**: Use for predicting continuous outcomes.
- **Classification**: Use for predicting categories.
- **Clustering**: Use for grouping similar data points.

### 4. Iterate and Refine

- **Iterate**: Data analysis is often a cyclic process. As you learn more, refine your approach.
- **Documentation**: Keep track of your steps and findings. This helps in reproducibility and sharing your work with others.
- **Seek Feedback**: Don't hesitate to ask for feedback from peers or mentors.

### Practical Steps to Start Analyzing

1. **Define the Problem**: Clearly outline what you're trying to solve or predict.
2. **Data Exploration**: Use the steps outlined above to understand the structure and content of your data.
3. **Feature Relationships**: Identify and visualize relationships between variables.
4. **Data Cleaning and Preparation**: Handle missing values, outliers, and encode categorical variables.
5. **Modeling**: Choose and apply appropriate models based on your problem.

### Additional Tips

- **Stay Curious**: Keep asking questions about the data and the relationships you observe.
- **Learn from Mistakes**: Don't be discouraged by initial setbacks. Each analysis is a learning opportunity.
- **Stay Updated**: The field of data science is constantly evolving. Keep learning new techniques and tools.

## Batch Learning vs. Online Learning in Machine Learning


| Feature           | Batch Learning                                      | Online Learning                                         |
|-------------------|-----------------------------------------------------|---------------------------------------------------------|
| **Data Processing** | Learns on the entire dataset at once.                | Learns on data instances sequentially (one by one or in small batches). |
| **Training Data** | Requires the entire dataset to be available upfront. | Can start learning with limited data and adapt as new data arrives. |
| **Computational Cost (Training)** | Can be computationally expensive, especially for large datasets. | Generally less computationally expensive per learning step. |
| **Memory Usage** | Requires enough memory to hold the entire dataset.     | Requires less memory as data is processed sequentially.    |
| **Learning Speed** | Training can be slow as it processes the entire dataset. | Learning is faster for each individual data point or small batch. |
| **Adaptability to New Data** | Requires retraining the entire model on the updated dataset. | Adapts to new data incrementally without full retraining. |
| **Use Cases** | Suitable for tasks where the dataset is relatively static and fits in memory (e.g., training a model on a fixed historical dataset). | Suitable for tasks with continuous data streams, large datasets that don't fit in memory, and models needing to adapt to changing patterns (e.g., stock price prediction, spam filtering, recommender systems). |
| **Model Updates** | Model is updated after processing the entire batch.   | Model is updated after processing each instance or small batch. |
| **Hyperparameter Tuning** | Typically done before training on the entire dataset. | Can be more complex to tune as the data distribution might change over time. |
| **Sensitivity to Data Order** | Less sensitive to the order of data in the training set. | Can be sensitive to the order of data arrival, especially with high learning rates.<br>[River library for online learning Model](https://riverml.xyz/latest/),  [VowPal Wabbit for online learning](https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/python_first_steps.html) . |

**[Link for Notes for ML Concepts](https://drive.google.com/file/d/1N_9GqQjWSWzX4BDXUJK-q4FwuEVuEKtn/view)**

## Two Fundamental Forms of Model Learning: Memorizing and Generalizing
 Learning can be broadly understood through two fundamental approaches: **memorizing** and **generalizing**.

**1. Memorizing (Rote Learning / Instance-Based Learning)**

* **Description:** Acquiring and retaining specific pieces of information or past experiences in detail. Applying this stored information directly to similar situations.
* **Analogy:** Memorizing vocabulary words, historical dates, specific sequences of steps.
* **Machine Learning Equivalent:** Instance-based learning (e.g., k-Nearest Neighbors), where the model "remembers" the training data and predicts based on similarity.

**Strengths:**

* Effective for recalling specific facts and procedures.
* Can be quick for familiar situations.

**Weaknesses:**

* Doesn't necessarily lead to understanding underlying principles.
* Poor at handling novel situations (struggles to generalize).
* Can be inefficient for large amounts of information.

**2. Generalizing (Understanding / Model-Based Learning)**

* **Description:** Identifying underlying patterns, rules, or principles from experience or data. Building an abstract understanding applicable to new, unseen situations.
* **Analogy:** Understanding grammar rules, physics laws, general problem-solving principles.
* **Machine Learning Equivalent:** Model-based learning (e.g., linear regression, decision trees, neural networks), where the model learns a function or parameters to map inputs to outputs.

**Strengths:**

* Allows application of knowledge to new and varied situations.
* Demonstrates a deeper understanding.
* More efficient for large and diverse datasets.

**Weaknesses:**

* Can be more time-consuming and effortful to develop.
* Generalizations can sometimes be incorrect or oversimplified.

**The Spectrum of Learning:**

In practice, learning often involves a blend of both memorization and generalization. Foundational facts might be memorized to aid understanding, and generalizations are often built upon past experiences. The goal of effective learning is usually to find a balance between recall and adaptability.

 ### Here's a step-by-step guide to help you through the process:

### 1. Understanding Your Data

**a. Data Collection**:
   - Ensure you have a complete dataset. This might include information like car make, model, year, price, mileage, engine type, etc.

**b. Data Inspection**:
   - Use basic functions (like `head()`, `tail()`, `info()`, and `describe()` in pandas) to understand the structure and summary statistics of your data.
   - Look for missing values, data types, and outliers.

### 2. Exploratory Data Analysis (EDA)

EDA helps you understand the data's underlying patterns, correlations, and distributions. Here are some common steps:

**a. Univariate Analysis**:
   - Look at one variable at a time.
   - Use histograms, box plots, and KDE plots for numerical data.
   - Use bar plots and pie charts for categorical data.

**b. Bivariate Analysis**:
   - Look at relationships between two variables.
   - Use scatter plots for numerical vs. numerical data.
   - Use bar plots or grouped bar plots for categorical vs. numerical data.
   - Use heatmaps for correlation matrices.

**c. Multivariate Analysis**:
   - Look at relationships involving more than two variables.
   - Use pair plots for visualizing multiple numerical variables.
   - Use faceting in seaborn to create multiple plots based on categorical variables.

### 3. Data Cleaning and Preprocessing
**0.Handling Imbalanced values**:
   **Techniques for Handling Imbalanced Data**<br>
         **Resampling Techniques**<br>
         **Oversampling (SMOTE)** <br>
         ***What it does:*** Synthetic Minority Over-sampling Technique (SMOTE) generates synthetic examples to balance the class distribution.<br>
         ***When to use:*** For highly imbalanced classes (e.g., fraud detection).<br>
         ***How to apply:*** `SMOTE()` from `imbalanced-learn`.<br> 
         **Undersampling**<br>
         ***What it does:*** Reduces the number of majority class instances to balance the dataset.<br>
         ***When to use:*** When the dataset is very large and removing some majority class instances won’t affect model performance.<br>
         ***How to apply:*** Use `RandomUnderSampler` from `imbalanced-learn`.<br> 
         **Class Weights**<br>
         ***What it does:*** Adjusts the weights of classes during model training to give more importance to minority classes.<br>
         ***When to use:*** When resampling is not preferred, but you still want to handle class imbalance.<br>
         ***How to apply:*** Set `class_weight='balanced'` in classifiers like `LogisticRegression` or `RandomForestClassifier`.<br> 
         **Anomaly Detection**<br>
         ***What it does:*** Treats the minority class as an anomaly or outlier and uses algorithms designed to detect outliers (e.g., Isolation Forest).<br>
         ***When to use:*** In cases where the minority class is rare but critical (e.g., fraud detection, disease diagnosis).<br>
         ***How to apply:*** Use algorithms like `IsolationForest` or `One-Class SVM`<br>
**a. Handling Missing Values**:
   - Decide whether to impute missing values (using mean, median, mode, or interpolation) or remove them.
     **Techniques for Handling Missing Values**<br>
         **Removing Missing Values**<br>
         ***What it does:*** Simply removes rows or columns that contain missing values.<br>
         ***When to use:*** When missing data is minimal and won’t affect the dataset significantly.<br>
         ***How to apply:*** `df.dropna()` in pandas.<br> 
         **Imputing Missing Values**<br>
         ***What it does:*** Fills missing values with a calculated value, such as mean, median, or mode.<br>
         ***When to use:*** When you can’t afford to lose rows or columns with missing data.<br>
         ***How to apply:*** Use `SimpleImputer()` in `sklearn` for mean/median/mode imputation.<br> 
         **K-Nearest Neighbors (KNN) Imputation**<br>
         ***What it does:*** Fills missing values based on the mean of the nearest neighbors.<br>
         ***When to use:*** When your data points are similar to each other and you want to use relationships to fill in missing values.<br>
         ***How to apply:*** `KNNImputer()` in `sklearn`.<br> 
         **Multiple Imputation**<br>
         ***What it does:*** Imputes missing values multiple times, producing several datasets, and then combines them for analysis.<br>
         ***When to use:*** When you want to account for uncertainty in the imputed values.<br>
         ***How to apply:*** Use the `IterativeImputer` or libraries like `fancyimpute`.<br>

**b. Encoding Categorical Variables**:
   - Convert categorical variables to numerical using techniques like one-hot encoding or label encoding.
     **Techniques for Handling Categorical Data**
         **Label Encoding**<br>
         ***What it does:*** Converts categorical labels (like "Red", "Blue", "Green") into numeric values (e.g., 0, 1, 2).<br>
         ***When to use:*** For ordinal categorical variables where the order matters (e.g., Low < Medium < High).<br>
         ***How to apply:*** `LabelEncoder()` in Python’s `sklearn` library.<br>         
         **One-Hot Encoding**<br>
         ***What it does:*** Converts categorical variables into a set of binary (0 or 1) columns, each representing a category.<br>
         ***When to use:*** For nominal variables (no inherent order, e.g., "Red", "Blue", "Green").<br>
         ***How to apply:*** Use `pd.get_dummies()` or `OneHotEncoder` in `sklearn`.<br>         
         **Ordinal Encoding**<br>
         ***What it does:*** Assigns each category an integer value based on some predefined order.<br>
         ***When to use:*** For categorical data with a clear order (e.g., "Low", "Medium", "High").<br>
         ***How to apply:*** Use `OrdinalEncoder` in `sklearn`.<br>
         **Target Encoding (Mean Encoding)** <br>
         ***What it does:*** Replaces categories with the mean of the target variable for each category.<br>
         ***When to use:*** For high-cardinality categorical variables.<br>
         ***How to apply:*** You can apply manually or use libraries like `category_encoders`.<br>

**c. Feature Scaling**:
   - Normalize or standardize numerical features if the scale differences are significant.
     **Techniques for Scaling**<br>
         **Min-Max Scaling**<br>
         ***What it does:*** Scales the data to a fixed range, usually between 0 and 1.<br>
         ***When to use:*** When features need to be on the same scale (e.g., neural networks).<br>
         ***How to apply:*** `MinMaxScaler()` in `sklearn`.<br> 
         **Standardization (Z-Score Scaling)** <br>
         ***What it does:*** Scales data so that it has a mean of 0 and a standard deviation of 1.<br>
         ***When to use:*** When you want the data to follow a normal distribution.<br>
         ***How to apply:*** `StandardScaler()` in `sklearn`.<br> 
         **Robust Scaling**<br>
         ***What it does:*** Scales the data using the median and the interquartile range (IQR), making it less sensitive to outliers.<br>
         ***When to use:*** When your data contains outliers.<br>
         ***How to apply:*** `RobustScaler()` in `sklearn`.<br> 
         **MaxAbs Scaling**<br>
         ***What it does:*** Scales each feature by its maximum absolute value, so all values are between -1 and 1.<br>
         ***When to use:*** For sparse data, or when you don’t want to shift/center the data.<br>
         ***How to apply:*** `MaxAbsScaler()` in `sklearn`.<br>

**d. Outlier Treatment**:
   - Decide whether to remove or transform outliers based on domain knowledge and their impact on the model. <br>
        **Outlier Detection and Handling Techniques**<br>
               **1. Z-Score Method (Standard Deviation Method)**   <br>            
               **What it does:** Measures how far a data point is from the mean in terms of standard deviations.<br>  
               **When to use:** If data follows a normal distribution.<br>  
               **How to apply:** Any point with a Z-score above 3 or below -3 is considered an outlier.<br>               
               **2. IQR (Interquartile Range) Method** <br>              
               **What it does:** Outliers are identified as those that lie below the lower bound (Q1 - 1.5 * IQR) or above the upper bound (Q3 + 1.5 * IQR).<br>  
               **When to use:** For non-normally distributed data or when you don’t assume data follows a bell curve.<br>  
               **How to apply:** Calculate Q1, Q3, and IQR, and flag data points outside the bounds.<br>               
               **3. Percentile-based Capping (Winsorization)** <br>               
               * **What it does:** Limits extreme values by setting them to a certain percentile value (usually the 1st and 99th percentiles).<br>  
               * **When to use:** When outliers are skewing the data and you want to make the dataset more robust.<br>  
               * **How to apply:** Cap the outliers at a certain percentile (e.g., replace values above the 99th percentile with the value at the 99th percentile). <br>                   **4. Isolation Forest**    <br>             
               * **What it does:** A machine learning algorithm that isolates outliers instead of profiling normal data points.<br>  
               * **When to use:** For high-dimensional datasets or complex relationships.<br>  
               * **How to apply:** Fit the Isolation Forest model to the data, then classify points as outliers based on isolation score.<br>               
               **5. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**<br>               
               * **What it does:** A clustering algorithm that identifies points as outliers if they do not belong to any cluster.<br>  
               * **When to use:** In spatial or clustering-based anomaly detection.<br>  
               * **How to apply:** Use DBSCAN to detect clusters, with points not belonging to any cluster marked as outliers.<br>               
               **6. Clipping or Truncation**<br>               
               * **What it does:** Replaces extreme values with the nearest value within a specified threshold (e.g., clip anything above the 95th percentile).<br>  
               * **When to use:** When you want to minimize the effect of outliers but still retain all data points.<br>  
               * **How to apply:** Set upper and lower bounds and replace outliers with these bounds.<br>               
               **7. Robust Regression** <br>               
               * **What it does:** Instead of removing outliers, this technique reduces their influence during modeling.<br>  
               * **When to use:** For regression models where outliers may influence the model parameters.<br>  
               * **How to apply:** Use algorithms like RANSAC or Theil-Sen estimator to build models that are less sensitive to outliers.<br>               
               **8. Log Transformation**<br>               
               * **What it does:** Compresses the data and helps reduce the impact of outliers by transforming the data using a logarithmic function.<br>  
               * **When to use:** When the data is highly skewed and contains extreme values.<br>  
               * **How to apply:** Apply log transformation to the entire feature to minimize the effect of large values.<br>               
               **9. Box-Cox Transformation**<br>               
               * **What it does:** Transforms data to make it more normal and handle extreme outliers by stabilizing variance.<br>  
               * **When to use:** When you want to transform non-normal data to normal.<br>  
               * **How to apply:** Apply Box-Cox transformation to adjust data distribution.

### 4. Feature Engineering

**a. Creating New Features**:
   - Derive new features from existing ones (e.g., calculating the age of the car from the year).

**b. Feature Selection**:
   - Use techniques like correlation matrix, feature importance from tree-based models, or PCA to select important features.<br>
     **Feature Selection Techniques**<br>

         **Chi-Square Test of Independence (χ²)** <br>
         ***What it does:*** Tests if there is a significant relationship between categorical variables.<br>
         ***When to use:*** For categorical independent and dependent variables.<br>
         ***How to apply:*** It calculates whether the observed frequency distribution differs significantly from the expected distribution.<br>
         ***How to apply in Python:*** `chi2()` from `sklearn.feature_selection`.<br>
         ***Use case:*** Selecting important categorical features by testing if they are independent of the target variable.<br> 

         **ANOVA F-Test (Analysis of Variance)** <br>
         ***What it does:*** Tests the means of different groups to check if they are significantly different.<br>
         ***When to use:*** For selecting categorical features when the dependent variable is continuous.<br>
         ***How to apply:*** It compares the variance between different groups to the variance within groups.<br>
         ***How to apply in Python:*** `f_classif()` from `sklearn.feature_selection` for classification tasks.<br>
         ***Use case:*** Checking whether continuous features differ significantly across the categories of the target variable.<br> 

         **Correlation Matrix**<br>
         ***What it does:*** Measures the linear relationship between features.<br>
         ***When to use:*** For continuous variables, it helps identify multicollinearity.<br>
         ***How to apply:*** High correlations between independent features can cause redundancy, and we often drop one of the correlated features.<br>
         ***How to apply in Python:*** `df.corr()` in pandas to calculate Pearson’s correlation coefficient.<br>
         ***Use case:*** Identifying highly correlated features and removing one to reduce multicollinearity.<br> 

         **Mutual Information**<br>
         ***What it does:*** Measures the dependency between variables. It computes the amount of information gained about one feature by knowing another feature.<br>
         ***When to use:*** Works for both categorical and continuous variables.<br>
         ***How to apply:*** It is a non-parametric method, and it can be used to measure how much information the independent variables provide about the dependent variable.<br>
         ***How to apply in Python:*** `mutual_info_classif()` for classification and `mutual_info_regression()` for regression from `sklearn.feature_selection`.<br>
         ***Use case:*** Feature selection when you need to measure the statistical dependence between variables.<br> 

         **Recursive Feature Elimination (RFE)** <br>
         ***What it does:*** Recursively removes features and builds a model on the remaining features, and ranks them based on performance.<br>
         ***When to use:*** When you want an automated process to select the most significant features.<br>
         ***How to apply:*** It uses cross-validation and model accuracy to eliminate the least important features.<br>
         ***How to apply in Python:*** `RFE()` from `sklearn.feature_selection`.<br>
         ***Use case:*** Reducing the number of features in models like linear regression, decision trees, and support vector machines (SVM).<br> 

         **L1 Regularization (Lasso Regression)** <br>
         ***What it does:*** Regularization technique that penalizes the absolute size of the coefficients, effectively shrinking some coefficients to zero.<br>
         ***When to use:*** For regression tasks when you want to enforce sparsity in your feature selection.<br>
         ***How to apply:*** The features with non-zero coefficients are considered important.<br>
         ***How to apply in Python:*** `Lasso()` from `sklearn.linear_model`.<br>
         ***Use case:*** Feature selection in high-dimensional datasets or when you want to reduce the model complexity.<br> 

         **Random Forest Feature Importance**<br>
         ***What it does:*** Measures the importance of each feature in predicting the target by observing how much each feature decreases the impurity in the tree.<br>
         ***When to use:*** For both classification and regression problems, particularly when dealing with a large number of features.<br>
         ***How to apply:*** Random forests create decision trees and rank features based on their contribution to reducing entropy.<br>
         ***How to apply in Python:*** `RandomForestClassifier()` or `RandomForestRegressor()` and check the `feature_importances_` attribute.<br>
         ***Use case:*** Ranking features based on their importance in a decision tree model.<br> 

         **Boruta Algorithm**<br>
         ***What it does:*** A wrapper method that uses a Random Forest classifier to perform feature selection, and it works by comparing the importance of each feature with that of a random feature.<br>
         ***When to use:*** To ensure that you only keep features that carry relevant information.<br>
         ***How to apply:*** It runs a Random Forest multiple times to determine the relevance of each feature.<br>
         ***How to apply in Python:*** `BorutaPy()` from the `Boruta` package.<br>
         ***Use case:*** For selecting important features from a large number of potential predictors in high-dimensional data.<br> 

         **Principal Component Analysis (PCA)**<br>
         ***What it does:*** Reduces dimensionality by transforming the features into a new set of features (principal components), which capture the most variance in the data.<br>
         ***When to use:*** For continuous features, particularly when you want to reduce the feature space.<br>
         ***How to apply:*** The principal components are linear combinations of the original features.<br>
         ***How to apply in Python:*** `PCA()` from `sklearn.decomposition`.<br>
         ***Use case:*** Dimensionality reduction and identifying important features that capture most of the variance.<br> 

         **Variance Threshold**<br>
         ***What it does:*** Removes all features that have a variance below a certain threshold. Features with low variance don’t carry much information.<br>
         ***When to use:*** When you want to remove features that are almost constant across all samples.<br>
         ***How to apply:*** Set a threshold below which the features are discarded.<br>
         ***How to apply in Python:*** `VarianceThreshold()` from `sklearn.feature_selection`.<br>
         ***Use case:*** Eliminating features with little to no variation across samples, typically useful in high-dimensional datasets.<br>
     
**c. Why Transformation of Features Are Required?**
   1. Linear Regression --- Gradient Descent --- Global Minima
   2. Algorithms like KNN, K Means, Hierarchical Clustering --- Euclidean Distance <br>
   ***Every Point has some vectors and Direction*** <br>
   ***Deep Learning Techniques (Standardization, Scaling)***<br>
   1. ANN ---> Global Minima, Gradient
   3. RNN
   4. 0-255 pixels
   
**d. Types Of Feature Transformation**:
   - Normalization And Standardization
   - Scaling to Minimum And Maximum values
   - Scaling To Median And Quantiles
   - Gaussian Transformation
   - Logarithmic Transformation
   - Reciprocal Transformation
   - Square Root Transformation
   - Exponential Transformation
   - Box Cox Transformation

### 5. Insight and Problem Identification

**a. Business Understanding**:
   - Understand the problem you're trying to solve (e.g., predicting car prices, classifying car types).

**b. Hypothesis Formation**:
   - Form hypotheses based on your EDA (e.g., cars with lower mileage might have higher prices).

**c. Insight Generation**:
   - Use your findings from EDA to generate insights and guide your modeling efforts.

### 6. Model Training Readiness

**a. Data Splitting**:
   - Split your data into training, validation, and test sets.

**b. Validation Strategy**:
   - Decide on a validation strategy (e.g., cross-validation).

### 7. Model Selection and Training

**a. Choosing the Right Model**:
   - Choose a model based on the problem type (regression for continuous outcomes, classification for categorical outcomes).

**b. Training the Model**:
   - Train your model using the training set and validate it using the validation set.

**c. Evaluation Metrics**:
   - Use appropriate evaluation metrics (e.g., RMSE for regression, accuracy for classification).

### 8. Iteration and Refinement

**a. Hyperparameter Tuning**:
   - Use techniques like grid search or random search to find the best model parameters.

**b. Model Ensembling**:
   - Combine multiple models to improve performance (e.g., bagging, boosting, stacking).

### 9. Final Evaluation

**a. Test Set Evaluation**:
   - Evaluate your final model on the test set to get an unbiased estimate of its performance.

**b. Deployment**:
   - Once satisfied with the model's performance, prepare it for deployment.

### Tips for Beginners

- **Start Simple**: Begin with basic visualizations and models. As you gain confidence, explore more complex techniques.
- **Ask Questions**: Always ask questions about your data and try to answer them through EDA.
- **Practice**: The more you practice, the more comfortable you'll become with the process.
- **Seek Help**: Don't hesitate to ask for help from colleagues, online forums, or tutorials.

By following these steps, you'll be able to systematically explore your car data, prepare it for modeling, and gain insights that can guide your decision-making process. 
Data Analysis and Visulization:

#### common Graphs
| Graph Type       | Data Type                  | Purpose                                     | Use Case                                          |
|------------------|----------------------------|---------------------------------------------|-------------------------------------------------|
| Histogram        | Numerical                  | Show distribution of a numerical variable   | Checking the distribution of Age or Salary        |
| Boxplot          | Numerical or Categorical (with groups) | Show distribution, detect outliers     | Comparing salary distribution across departments  |
| Density Plot     | Numerical                  | Show smoothed distribution                  | Checking distribution shape (e.g., normal distribution) |
| Bar Chart        | Categorical                | Show frequency of categories                | Showing department distribution in a company      |
| Pie Chart        | Categorical                | Show proportions of categories              | Showing market share of different products        |
| Scatter Plot     | Numerical vs Numerical     | Show relationship between two numerical variables | Correlation between Height and Weight         |
| Heatmap          | Numerical (Correlation Matrix) | Show correlation between variables        | Visualizing correlations between Salary, Age, and Experience |
| Stacked Bar Chart | Categorical vs Categorical | Show relationship between two categorical variables | Comparing Gender within Department            |
| Violin Plot      | Numerical vs Categorical   | Show distribution of a numerical variable across categories | Comparing Income distribution across Region categories |
| Pair Plot        | Numerical                  | Show pairwise relationships between variables | Exploring relationships between multiple numerical features |

**In Summary**:
- **Distribution graphs** (histograms, boxplots, density plots) are used to visualize the spread and central tendency of single variables.
- **Relationship graphs** (scatter plots, boxplots, pair plots) help you understand how two variables interact, whether they are numeric or categorical.
- **Correlation graphs** (heatmaps, scatter plots) help visualize the degree to which numerical variables are related.
- **Single variable graphs** like bar charts and pie charts are best for visualizing the distribution of categorical variables.

## Categories and Attributes in Data Variables

When working with data, various terms describe the structure, storage, and representation of information. Below is a list of common terms used in data science, databases, and analytics.

### **1. Dataset Terminology**
In the context of data science, a dataset is often represented as a table where data is organized into rows and columns.

#### **Columns (Feature/Variable)**
- **Description**: A column represents a specific type of data, usually corresponding to one attribute or feature. Each column contains values of a particular variable.
- **Example**: In a dataset of employee information, columns might include Age, Salary, Department, etc.
- **Alternative Names**:
  - **Feature**: Refers to columns in machine learning datasets.
  - **Variable**: A more general term for any column in a dataset.
  - **Attribute**: Often used to describe a column that represents a characteristic of an entity.

#### **Rows (Record/Observation)**
- **Description**: A row represents an individual record or observation. Each row contains values for all the columns in the dataset.
- **Example**: A row in a dataset could represent a single employee with all their information (e.g., age, salary, department, etc.).
- **Alternative Names**:
  - **Record**: Refers to an individual entry in the dataset (usually one complete observation).
  - **Observation**: Each row is considered an observation in statistical analysis or machine learning.
  - **Instance**: In machine learning, each row is often called an instance of the data.

#### **Values (Data Point)**
- **Description**: A value refers to the individual piece of data inside a cell (intersection of a row and column).
- **Example**: In the column Age, a value might be 25. In the column Department, a value could be HR.

### **2. Categories of Data Variables**
- **Categorical Variable**: Represents distinct groups or categories.
  - Example: Colors (Red, Green, Blue), Gender (Male, Female, Other)
  - **Nominal**: Categories that have no particular order (e.g., Red, Blue, Green).
  - **Ordinal**: Categories that have a specific order or ranking (e.g., Low, Medium, High).
- **Numerical Variable**: Represents numerical values.
  - Example: Age (25, 30, 40), Salary (50000, 70000)
  - **Discrete**: Numeric values that are countable and finite (e.g., Number of students in a class).
  - **Continuous**: Numeric values that can take any value in a range (e.g., Height, Temperature).
- **Binary Variable**: A variable with only two possible values.
  - Example: Yes/No, 0/1, True/False

### **3. Data Terms for Specific Analysis**
- **Target Variable (Dependent Variable)**: The variable that is predicted or explained.
  - Example: In a sales prediction model, Sales Revenue is the target variable.
- **Predictor Variable (Independent Variable)**: The variable used to predict the target variable.
  - Example: In a sales prediction model, Advertising Spend is an independent variable.
- **Constant**: A variable that doesn’t change its value in the dataset.
  - Example: If the dataset has a column Country with all values being USA, that column is a constant.

### **4. More Terminology Related to Rows and Columns**
- **Index**: A label for identifying rows in the dataset.
  - Example: In a pandas DataFrame, the index can be a column like ID or automatically generated (0, 1, 2, 3,...).
- **Key**: A column (or set of columns) that uniquely identifies each row in a dataset.
  - Example: In a customer dataset, CustomerID could be the primary key.
- **Foreign Key**: A column that creates a relationship between two tables by referencing the primary key of another table.
  - Example: In a Sales table, CustomerID references CustomerID in the Customers table.
- **Missing Data**: When a value is not available for a specific row and column.
  - Example: If a customer hasn't provided their Phone Number, the value in the Phone Number column could be missing (NaN in pandas).

### **5. Statistics-Related Terms**
- **Mode**: The most frequent value in a dataset.
  - Example: If the Department column has values HR, Finance, HR, Marketing, HR, then the mode is HR.
- **Median**: The middle value of a sorted dataset.
  - Example: For the dataset 1, 2, 3, 4, 5, the median is 3.
- **Mean**: The sum of all values divided by the number of values.
  - Example: For the dataset 2, 4, 6, 8, the mean is (2 + 4 + 6 + 8) / 4 = 5.

### **6. Data Quality-Related Terms**
- **Outlier**: A data point that is significantly different from the majority of other data points.
  - Example: In a dataset of house prices, a house with a price of $1 million might be an outlier if most houses are priced between $100k to $500k.
- **Skewness**: Describes the asymmetry of the data distribution.
  - Example: Income data is often right-skewed because a few people earn much more than the majority.
- **Kurtosis**: Measures the "tailedness" of the data distribution.
  - Example: Financial returns often show high kurtosis, meaning extreme values (outliers) happen more often than a normal distribution would predict.

### **7. Types of Data in a Dataset**
- **Structured Data**: Data that is organized in a fixed schema, like tables in a relational database.
- **Unstructured Data**: Data without a predefined schema, like images, videos, or raw text.
- **Semi-structured Data**: Data that does not follow a strict table format but has some structure, like JSON or XML files.
- **Time-Series Data**: Data points indexed by time (e.g., Stock prices recorded every minute).
- **Cross-Sectional Data**: Data collected at a single point in time (e.g., Census data).
- **Panel Data (Longitudinal Data)**: Data collected over time for the same entities (e.g., GDP of countries recorded every year for a decade).

### **8. Data Representation in Different Formats**
- **Tabular Data**: Data arranged in rows and columns (e.g., CSV, SQL tables).
- **Relational Data**: Data stored with relations between multiple tables (e.g., SQL databases).
- **Hierarchical Data**: Data with parent-child relationships (e.g., XML, JSON, Tree structures).
- **Graph Data**: Data represented as nodes and edges (e.g., Social Network graphs).

This should help clarify the different terms and their meanings in data science and databases!


#### 1. Distribution
The distribution of data refers to how the values of a variable are spread out or arranged. In other words, it tells you how often each value or range of values occurs in your dataset.

**A distribution can be visualized in different ways**, such as:

**Histograms:** Show the frequency of data points within certain intervals (bins). <br>
**Boxplots:** Provide a summary of the distribution, showing the median, quartiles, and potential outliers. <br>
**Probability Distribution:** This is the theoretical model describing how the values of a variable are distributed, like normal distribution or exponential distribution. <br>
**Example:** If you have a dataset of people's ages, the distribution tells you how many people fall into certain age groups (e.g., 20-30 years old, 30-40 years old, etc.).
#### 2. Spread or Dispersion
The spread or dispersion of the data refers to the extent to which the values in your dataset differ from the average or central value. It answers the question: How much do the values vary?

**There are a few key measures of dispersion (spread):**

**a. Range**
The range is the simplest measure of spread, which is the difference between the maximum and minimum values in the dataset. <br>
Formula: <br>
Range=Maximum Value−Minimum Value <br>
Example: If your data points are ages of people: 18, 25, 30, 35, 60, then the range is 60−18=42

**b. Variance**
Variance measures how far each data point is from the mean (average) and gives us an idea of how spread out the values are. The higher the variance, the more spread out the data is. <br>
Formula:

Variance=1/𝑛∑(𝑋𝑖−𝜇)square <br>
where: <br>
𝑋𝑖​  <br>
  is each individual data point <br>
𝜇 <br>
μ is the mean of the data <br>
𝑛 <br>
n is the number of data points <br>
Example: If most data points are close to the mean, variance will be small; if data points are spread out, variance will be large.

**c. Standard Deviation**  <br>
Standard deviation is the square root of the variance. It’s more interpretable than variance because it’s in the same units as the data. <br>
`Small std = Less variation.` <br>
`Large std = More variation.` ,sometime it is useful as it has more information or variablity. <br>
A **std of 52** indicates moderate spread, suggesting you have a `mix of low and high horsepower` cars. <br>
It's useful for understanding the spread and variability of your data.<br>
Formula: <br>

Standard Deviation= squre root of Variance​ <br>
 
**Example:** If the standard deviation is small, the data points are close to the mean. If it's large, the data points are spread out across a wider range.i.e.<br>
**Real-Life Example:**<br>
Horsepower in cars: If the mean horsepower is 143 and the standard deviation is 52, it means most cars have horsepower between:<br>
`143−52=91(lower bound)` <br>
`143+52=195(upper bound)`<br>
So, about 68% of cars in the dataset have horsepower between 91 and 195.<br>
on average, the horsepower variable of cars in car dataset deviates by 52 units from the mean.If the mean is 143 hp and the standard deviation is 52 hp, this means that most of the cars in the dataset will have horsepower values that are between 91 hp and 195 hp or (1 standard deviation away from the mean).
For More Comprehensive you can visualize through the graphs using boxplot etc.
**Other Best Example** suppose we have the missing values in a column and we impute these missing values with central tendency ,then to check the difference before applying the central tendency to that column and after applying  using standard deviation ,it will show the difference ,this is how through std we can apply or check the difference for central tendency after applying.

**d. Interquartile Range (IQR)**  <br>
IQR measures the range within which the central 50% of data points lie. It's calculated as the difference between the third quartile (Q3) and the first quartile (Q1). <br>

Formula:<br>
IQR=𝑄3−𝑄1<br>
Example: If the first quartile is 20 and the third quartile is 40, the IQR is <br>
40−20=20. This tells you that the middle 50% of the data falls within a range of 20 units.

### Key Terms:
1. **Distribution**: Refers to how data points are spread out or arranged in a dataset.<br>
2. **Spread/Dispersion**: Indicates how much the data points differ from the central value, such as the mean.<br>
3. **Range**: The difference between the maximum and minimum values in a dataset.<br>
4. **Variance**: A measure of how much each data point differs from the mean. It's calculated by averaging the squared differences from the mean.<br>
5. **Standard Deviation**: The square root of variance, providing a measure of spread in the same units as the data.<br>
6. **Interquartile Range (IQR)**: The range between the first quartile (Q1) and third quartile (Q3), representing the middle 50% of the data.

### Visualizing Distribution and Spread:
- **Histograms**: These plots display the distribution of data across different intervals or bins, showing the frequency of data points within each bin.<br>
- **Boxplots**: These charts illustrate the spread of data by showing the median, quartiles, and potential outliers. They are useful for comparing distributions across different groups.<br>
- **Density Plots**: Similar to histograms but provide a smooth curve representing the distribution of the data, which can be more insightful for understanding the shape of the data.

### Examples of Distributions:
1. **Uniform Distribution**: All values occur with equal frequency. An example is rolling a fair die where each outcome (1 through 6) has an equal chance of appearing.<br>
2. **Normal Distribution**: Data follows a bell-shaped curve. Most data points cluster around the mean, with fewer points as you move away from the mean. This is often referred to as a "bell curve."<br>
3. **Skewed Distribution**: The data not symmetric. <br>
   - **Right-Skewed (Positive Skew)**: The tail of the distribution extends to the right, meaning there are a few very large values.<br>
   - **Left-Skewed (Negative Skew)**: The tail extends to the left, indicating a few very small values.<br>
#### Why are these Important?
**Distribution** helps us understand how our data behaves, and what patterns or trends might exist.<br>
**Spread/Dispersion** helps us gauge how variable the data is, and whether the data is tightly packed or widely spread.

#### What is the Analysis?   A detailed examination of anything complex in order to understand its nature or to determine its essential features

Data Analysis is the process of cleaning, transforming, visualizing, and analyzing the data to gain valuable insights to make more effective business decisions is known as Data Analysis.
## 1-Advanced Pandas
1-https://www.javatpoint.com/pandas-cheat-sheet,
2-https://python.plainenglish.io/a-comprehensive-guide-to-pandas-cheat-sheet-for-data-science-enthusiasts-b6f131ab5284
#### 1.1-Working with Multi-index DataFrames
Multi-index allows you to select more than one row and column in your index.Mean we can combine/add/append multiple arrays,tuples,frames with different values as columns or rows to make one dataset.
We can use various methods of multi-index such as MultiIndex.from_arrays(), MultiIndex.from_tuples(), MultiIndex.from_product(), MultiIndex.from_frame, etc., which helps us to create multiple indexes from arrays, tuples, DataFrame,
## 2- Numpy Applications
#### Eigenvalues & Eigenvector
**Scalar:** 0-dimensional, just a single number.Example: 5, 3.14, -42<br>
**Vector:** 1-dimensional, an array of numbers. Example: [1, 2, 3], [0, 3.5, 6, 9]<br>
**Matrix:** 2-dimensional, an array of numbers arranged in rows and columns.Example: [[1, 2], [3, 4]], [[5, 6, 7], [8, 9, 10]] <br>
Here basically we have matrixlets say A=[1 2 3] and vector which is also a matrix X=[4 5 6 ],& when we compute them by performing the dot product ,at the end we have some value which is either 0 or some number that's what we can say our   
Compute the product AX
 for
AX=kX
  where  k
  is some scalar,When this equation holds for some  X
  and  k
 , we call the scalar  k
  an eigenvalue of  A
 . We often use the special symbol  λ
  instead of  k
  when referring to eigenvalues. In Example  7.1.1
 , the values  10
  and  0
  are eigenvalues for the matrix  A
  and we can label these as  λ1=10
  and  λ2=0
 
When  AX=λX
  for some  X≠0
 , we call such an  X
  an eigenvector of the matrix  A
 . The eigenvectors of  A
  are associated to an eigenvalue. Hence, if  λ1
  is an eigenvalue of  A
  and  AX=λ1X
 , we can label this eigenvector as  X1
 . Note again that in order to be an eigenvector,  X
  must be nonzero.
Let’s say you have a matrix 
𝐴
A representing some transformation in space. If you multiply this matrix by a vector 
𝑋
X, it will generally rotate and scale 
𝑋
X. But if 
𝑋
X is an eigenvector of 
𝐴
A, then multiplying 
𝐴
A by 
𝑋
X will only scale 
𝑋
X by a factor 
𝜆
λ, without changing its direction. In other words:

𝐴
𝑋
=
𝜆
𝑋
AX=λX
So, the eigenvalue 
𝜆
λ represents how much the vector is stretched or compressed, and the eigenvector 
𝑋
X represents the direction that stays unchanged under the transformation by matrix 
𝐴
This equation means that when we apply the matrix 
𝐴
A to the vector 
𝑋
X, the vector 
𝑋
X only gets scaled by 
𝜆
λ (but not rotated or changed direction). The eigenvector 
𝑋
X remains in the same direction after being transformed by 
𝐴
A, but its length is scaled by the corresponding eigenvalue 
𝜆
λ.
  for furthere understanding you can look at here this link    https://byjus.com/maths/eigen-values/
  ##### Why/Purpose of eigenvalues and vector
**1. Simplifying Matrix Operations:**
Eigenvalues and eigenvectors allow us to diagonalize a matrix (if possible), which simplifies matrix operations. When a matrix is diagonalized, it is represented as a product of its eigenvectors and eigenvalues, making many matrix calculations, like exponentiation, much easier.

Example: In physics or in systems analysis, diagonalization helps reduce complex systems of linear equations to simpler forms.

**2. Data Reduction (PCA in Machine Learning):**
Eigenvalues and eigenvectors are used in Principal Component Analysis (PCA), a common technique in machine learning and data science for dimensionality reduction. PCA helps identify the "principal components" of the data, which are the directions (eigenvectors) that explain the most variance in the data. The eigenvalues tell you how much variance each principal component explains.

***Why it matters:*** In large datasets, many features (dimensions) might be redundant. PCA reduces the dimensions of the data while retaining most of the information, which makes the data easier to work with (e.g., faster training in machine learning).
## 3- Data Visualization with matplotlib and seaborn
**Seaborn vs Matplotlib**
**Matplotlib** is a low-level plotting library that provides a high degree of control over individual elements. Even for basic functionalities, it requires more code.

Whereas **seaborn** is a high level library for visualization and requires less coding compared to matplotli.

**Matplotlib** lets users customize the appearances of plots, including color and styles.

**Seaborn** has in-built themes and color palettes making it easier for users to create visually appealing plots.

**Matplotlib** can work with pandas but users may need to manipulate data for certain type of plots.

**Seaborn** is very much flexible with pandas and it doesn’t require as much manipulation as matplotlib.

**Annotations** allow users to label data points and indicate trends or add descriptions to different parts of a plot.

**Both seaborn and matplotlib have style themes.**

There are **five themes** in seaborn:
`white,
dark,
whitegrid,
darkgrid,
ticks`
We will use style theme from matplotlib. Matplotlib gives 26 styles which you can be seen with the `plt.style.available` method.
**Output:**
`['Solarize_Light2',
 '_classic_test_patch',
 'bmh',
 'classic',
 'dark_background',
 'fast',
 'fivethirtyeight',
 'ggplot',
 'grayscale',
 'seaborn',
 'seaborn-bright',
 'seaborn-colorblind',
 'seaborn-dark',
 'seaborn-dark-palette',
 'seaborn-darkgrid',
 'seaborn-deep',
 'seaborn-muted',
 'seaborn-notebook',
 'seaborn-paper',
 'seaborn-pastel',
 'seaborn-poster',
 'seaborn-talk',
 'seaborn-ticks',
 'seaborn-white',
 'seaborn-whitegrid',
 'tableau-colorblind10']`
 ### What is Visualization?
Visualization refers to techniques used to communicate both abstract and concrete,behaviour,information, ideas by creating images, diagrams, or animations of objects
### 3.1- Data Visualization Explanation
Data visualization uses graphical representations like graphs, charts, and maps to simplify complex data, making it easier to understand and analyze. Visuals allow data scientists to summarize thousands of rows and columns of complex data and put it in an understandable and accessible format. It helps identify patterns, correlations, and outliers, providing a more effective analysis than tables or descriptive statistics. Data visualization is crucial in decision-making, enabling data analysts, scientists, and engineers to communicate insights to non-technical stakeholders and inform actions, such as A/B testing or addressing bias in models like ChatGPT.  
Link to understand more about the data visualization: [What is Data Visualization?](https://www.couchbase.com/blog/what-is-data-analysis/)

#### A comprehensive breakdown of Data Analysis and Data Visualization concepts, organized into a structured tree format to clarify the relationship between different elements:

#### 1. Data Types
- **Quantitative Data**: Numerical values, used for statistical analysis.
- **Qualitative Data**: Descriptive data, typically non-numerical and used for interpretive analysis.

#### 2. Types of Data Analysis
- **Univariate Analysis**: Univariate Analysis for Continuous Variables and Categorical Variables.
- **Bivariate Analysis**: Bivariate Analysis for Continuous Variable vs Continuous Variable, Categorical Variable vs Categorical Variable.
- **Multivariate Analysis**: Multivariate Analysis for Numerical-Numerical,Numerical-Categorical Variables.

#### 3. Data Analysis Techniques
- **Descriptive Analysis**: Summarizes main features (e.g., mean, median, mode).  
  **Tools**: Histograms, Bar Charts.  
- **Diagnostic Analysis**: Investigates causes and patterns (e.g., regression, correlation).  
  **Tools**: Scatter plots, Heatmaps.  
- **Predictive Analysis**: Forecasts future outcomes using historical data.  
  **Tools**: Line plots, Forecasting models.  
- **Prescriptive Analysis**: Provides recommendations based on predictive insights.  
  **Tools**: Optimization models, Decision trees.  

#### 4. Types of Data Analysis Methods
- **Statistical Analysis**: Applies statistical techniques (e.g., regression, hypothesis testing).  
  **Tools**: R, Python (scikit-learn, NumPy).  
- **Data Mining**: Discovers patterns from large datasets.  
  **Tools**: Clustering, Classification algorithms.  
- **Text Mining**: Analyzes unstructured textual data.  
  **Tools**: Natural Language Processing (NLP), sentiment analysis.  
- **Time Series Analysis**: Focuses on data points collected over time.  
  **Tools**: Forecasting, Decomposition.

#### 5. Key Data Visualization Techniques
- **Distribution**: Analyzing how data points are spread.  
  **Visualizations**: Histograms, Boxplots.  
- **Correlation**: Identifying relationships between variables.  
  **Visualizations**: Scatter plots, Heatmaps.  
- **Ranking**: Comparing quantities in order.  
  **Visualizations**: Bar charts, Dot plots.  
- **Part-of-Whole**: Showing proportions of a total.  
  **Visualizations**: Pie charts, Stacked Bar charts.  
- **Evolution**: Analyzing changes over time.  
  **Visualizations**: Line charts, Area charts.  
- **Map**: Geospatial data visualization.  
  **Visualizations**: Choropleth maps, Heat maps.  
- **Networks**: Analyzing connections or relationships.  
  **Visualizations**: Network graphs, Force-directed graphs.

#### 6. Deciding the Right Technique
**Question to Ask**:  
- How many variables are you analyzing?  
  - **Univariate Analysis**: Analyzing a single variable to understand its distribution, central tendency, and variability.  
  - **Bivariate Analysis**: Analyzing the relationship between two variables, often looking for correlations or associations.  
  - **Multivariate Analysis**: Analyzing the relationships between three or more variables simultaneously.

**What do you want to analyze?**  
- Use suitable techniques for distribution, correlation, ranking, etc.  
- Choose appropriate visualizations depending on analysis goals.

#### 7. Data Analysis Process
- **Define Objective**: Understand what you want to achieve.
- **Prepare and Explore Data**: Clean data, check for outliers.
- **Apply Analysis Techniques**: Choose methods (statistical, machine learning, etc.).
- **Interpret Results**: Derive meaningful insights.
- **Communicate Findings**: Use visualizations and reports to convey results effectively.

#### 8. Common Data Analysis Tools
- **Spreadsheet Software**: Basic analysis (Excel, Google Sheets).
- **Business Intelligence Platforms**: Interactive dashboards (Power BI, Tableau).
- **Programming Languages**: Advanced analysis (Python, R).
- **Cloud-Based Platforms**: Scalable environments for large datasets (Google Cloud, AWS).
- **Text Analytics Tools**: Analyzing unstructured data (NLP libraries, sentiment analysis tools).

#### 9. Importance of Data Analysis
- **Decision Making**: Provides insights for informed choices.
- **Problem Solving**: Identifies root causes and optimizes processes.
- **Performance Evaluation**: Measures success and evaluates KPIs.
- **Risk Management**: Identifies and mitigates potential risks.
- **Gathering Insights**: Drives innovation and strategic actions.

This tree structure organizes data analysis and visualization concepts systematically, making it easier to understand how to approach data analysis, choose the right techniques, and select appropriate visualizations based on your data and goals.
### 3.2- Structured Flow for Implementing Data Analysis

The process of implementing data analysis can be broken down into clear steps, starting from problem identification and leading up to decision-making based on the analysis. Below is a structured approach for implementing data analysis:

#### Step 1: Define the Objective
- **Action**: Clearly identify the question or problem you want to solve with data.
- **Example**: "What are the key factors influencing customer churn?"

#### Step 2: Collect and Prepare the Data
- **Action**: Gather data relevant to the analysis.  
  Ensure data quality (handle missing values, duplicates, or irrelevant information).
- **Example**: Collect customer demographic data, transaction history, and support interaction logs.

#### Step 3: Identify the Type of Data
- **Action**: Classify your data into Quantitative or Qualitative:
  - **Quantitative**: Numerical, measurable (e.g., age, sales, etc.).
  - **Qualitative**: Descriptive, non-numerical (e.g., customer feedback, reviews).
- **Example**: Customer age (quantitative), customer feedback (qualitative).

#### Step 4: Choose the Type of Data Analysis
- **Action**: Select the type of analysis based on your objective:
  - **Univariate Analysis** (for one variable): Summarize basic statistics (mean, median, etc.).
  - **Bivariate Analysis** (for two variables): Identify relationships (e.g., correlation).
  - **Multivariate Analysis** (for multiple variables): Explore interactions among many variables.
- **Example**: Analyze customer churn using Bivariate analysis (e.g., age vs. churn, tenure vs. churn).

#### Step 5: Select the Appropriate Data Analysis Technique
- **Action**: Based on the data type and analysis, choose the most suitable technique:
  - **Descriptive**: Summarize key statistics.
  - **Diagnostic**: Identify reasons or relationships.
  - **Predictive**: Forecast future behavior or outcomes.
  - **Prescriptive**: Provide recommendations for optimal action.
- **Example**: For customer churn prediction, Predictive analysis using a regression model or machine learning.

#### Step 6: Visualize the Data
- **Action**: Choose appropriate visualizations based on your analysis type:
  - **Distribution**: Histograms, Boxplots.
  - **Correlation**: Scatter Plots, Heatmaps.
  - **Ranking**: Bar Charts, Dot Plots.
  - **Part-of-Whole**: Pie Charts, Stacked Bars.
  - **Evolution**: Line Charts, Area Charts.
  - **Map**: Geospatial Visualization.
  - **Networks**: Network Graphs.
- **Example**: Use scatter plots to visualize the relationship between age and churn rate, or line charts to track churn over time.

#### Step 7: Apply the Chosen Analysis Method
- **Action**: Apply the selected technique using appropriate tools:
  - **Statistical Methods** (e.g., hypothesis testing, regression).
  - **Data Mining** (e.g., clustering, classification).
  - **Text Mining** (e.g., sentiment analysis).
  - **Time Series** (e.g., forecasting, decomposition).
- **Example**: Apply predictive analysis using regression or machine learning algorithms to predict churn.

#### Step 8: Interpret the Results
- **Action**: Review the output of your analysis and interpret the results:
  - What patterns, trends, or correlations have emerged?
  - What are the key insights, and what do they mean in the context of your problem?
- **Example**: Identify that age and customer tenure are strong predictors of churn, with younger customers and those with shorter tenure more likely to leave.

#### Step 9: Make Data-Driven Decisions
- **Action**: Use the insights to make informed decisions or recommendations:
  - Suggest improvements, strategies, or actions based on the analysis.
- **Example**: Develop a retention strategy focusing on younger customers and offering incentives to long-term customers to reduce churn.

#### Step 10: Communicate the Findings
- **Action**: Present your findings clearly to stakeholders:
  - Use visualizations to highlight key points (charts, graphs).
  - Provide a clear narrative of the insights and actionable recommendations.
- **Example**: Present a report and dashboard to executives, showcasing the key churn drivers, visualizing trends, and suggesting actions for retention.

#### Step 11: Take Action & Monitor
- **Action**: Implement the suggested actions and monitor their effectiveness over time:
  - Use **Prescriptive Analysis** to optimize strategies and actions.
  - Monitor outcomes to see if the decisions lead to improved results.
- **Example**: After implementing retention strategies, monitor churn rates and evaluate the success of your interventions.

#### Step 12: Iteration and Refinement
- **Action**: Data analysis is an iterative process:
  - As new data comes in, revisit the analysis to refine insights or improve predictions.
- **Example**: Track churn metrics monthly, refining your predictive model with new customer data.

#### Tools & Resources Needed:
- **Basic Tools**: Excel, Google Sheets (for simple analysis).
- **Advanced Tools**: R, Python (for machine learning and deep analysis).
- **Visualization Tools**: Tableau, Power BI (for interactive reports and dashboards).
- **Cloud Analytics**: AWS, Azure (for large-scale data processing).

### Summary of the Flow:
- **Define the objective → Collect and prepare data → Identify data types → Choose analysis type → Select analysis technique → Visualize the data → Apply analysis methods → Interpret the results → Make decisions → Communicate findings → Take action → Refine and iterate**.

This flow ensures a systematic approach to data analysis and visualization, enabling you to extract meaningful insights, make informed decisions, and optimize outcomes for your organization or research.
### 3.3- Plots explanation and their use case
`Line Plot: Use plt.plot() to visualize continuous data.
Scatter Plot: Use plt.scatter() to show relationships between two variables.
Bar Plot: Use plt.bar() for categorical data.
Histogram: Use plt.hist() to show the distribution of data.
Heatmap: Use sns.heatmap() for 2D data with color coding.
Pair Plot: Use sns.pairplot() for visualizing pairwise relationships.
Joint Plot: Use sns.jointplot() for examining the relationship between two variables.
Customizing Plots: Add annotations using plt.annotate(), change styles using plt.style.use(), and apply themes using sns.set_theme().
Saving Plots: Use plt.savefig() to save plots in different formats like .png, .svg.`
### 4.1 Documentation of the Above Real World Project ,for Sales Data Analysis Workflow with Each Steps Implementatio
---
#### 1. Define the Objective
Analyzed the sales data to identify key trends and insights such as high-performing product categories, customer demographics, and sales patterns.
---
#### 2. Collect and Prepare Data
- The dataset is already loaded into a DataFrame named `df`.
- **Columns and Data Types:**
Order ID int64 Order Date object Product ID object Product Category object Buyer Gender object Buyer Age int64 Order Location object International Shipping object Sales Price int64 Shipping Charges int64 Sales per Unit int64 Quantity int64 Total Sales int64 Rating int64 Review object
---
#### 3. Identify Data Types
- The dataset contains:
- **Categorical Variables:** `Product Category`, `Buyer Gender`, `Order Location`, `International Shipping`, `Review`.
- **Numerical Variables:** `Sales Price`, `Shipping Charges`, `Sales per Unit`, `Quantity`, `Total Sales`, `Rating`, `Buyer Age`.
---
#### 4. Choose Analysis Type
- **Univariate Analysis:** To explore distributions of individual variables like `Sales Price`, `Rating`, and `Buyer Gender`.
- **Bivariate Analysis:** To analyze relationships between variables, e.g., `Product Category` vs. `Sales Price` or `Rating` vs. `Total Sales`.
---
#### 5. Select Analysis Techniques
- **Descriptive Statistics:** Mean, median, standard deviation of numerical columns.
- **Visualizations:** Histograms, bar charts, and correlation heatmaps.
- **Correlation Analysis:** To measure relationships between numerical variables.
---
#### 6. Visualize the Data
`plt.figure(figsize=(14, 10))`
**Distribution of Sales Price**
`plt.subplot(2, 2, 1)
plt.hist(df['Sales Price'], bins=30, color='blue', edgecolor='black')
plt.title('Distribution of Sales Price')
plt.xlabel('Sales Price')
plt.ylabel('Frequency')`

**Sales by Product Category**
`plt.subplot(2, 2, 2)
df['Product Category'].value_counts().plot(kind='bar', color='green')
plt.title('Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Number of Sales')`

**Sales by Buyer Gender**
`plt.subplot(2, 2, 3)
df['Buyer Gender'].value_counts().plot(kind='bar', color='purple')
plt.title('Sales by Buyer Gender')
plt.xlabel('Buyer Gender')
plt.ylabel('Number of Sales')`

**Average Rating by Product Category**
`plt.subplot(2, 2, 4)
df.groupby('Product Category')['Rating'].mean().plot(kind='bar', color='orange')
plt.title('Average Rating by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Average Rating')
plt.tight_layout()
plt.show()`

#### 7. Apply Analysis Methods
**Descriptive Statistics:** To summarize numerical columns (Sales Price, Total Sales, Rating).
**Correlation Analysis:** Calculating the correlation matrix for numerical variables.
`correlation_matrix = df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)`

#### 8. Interpret the Results
**Distribution of Sales Price:** Understand the frequency of different price ranges.
**High-Selling Product Categories:** From the bar chart, identify which product categories have the most sales.
**Buyer Demographics:** Analyze the gender distribution to see the major contributing group.
**Correlation Insights:** Identify relationships, e.g., between Sales Price, Quantity, and Total Sales.

#### Implementation of Steps 4 to 8
**Univariate Analysis (Step 4):**

Histogram of Sales Price to visualize distribution.
Count plot for Product Category and Buyer Gender.
**Bivariate Analysis (Step 5):**
Bar chart to analyze the relationship between Product Category and average Rating.
**Analysis Techniques (Step 6):**
Use visualizations for patterns and correlations.
Generate a correlation matrix to assess numerical relationships.
**Insights and Results (Step 7 and 8):**
Identify categories with high average ratings and significant correlations, e.g., if higher Quantity leads to lower Sales Price per unit.

#### 9. Make Decisions
**Example decisions:**
Focus marketing efforts on high-selling product categories.
Target underrepresented buyer demographics for growth opportunities.
Optimize pricing strategies based on insights from the correlation matrix.

#### 10. Communicate Findings
Create a summary report with visualizations and key takeaways.
Share findings with stakeholders through charts and actionable insights.

#### 11. Take Action
Implement recommendations such as adjusting marketing budgets or improving customer service in top-performing categories.

#### 12. Refine and Iterate
Continuously monitor sales data and refine strategies based on updated insights and new patterns.
### Important Things to remember
**Data distribution:** describes the range of values in a dataset and how often each value occurs. It can be used to understand the data's center, spread, shape, and modality.
-***Data distribution can be represented graphically using histograms and box plots.***
-Two types of data distribution: discrete and continuous. Discrete data has specific values, while continuous data can have an infinite number of values.
**Data manipulation** is the process of changing or organizing data to make it more accessible, readable, and useful: 
**Purpose**
Data manipulation is a key step in data analysis and processing. It helps prepare data for analysis, reporting, visualization, and other computational tasks. 
**Techniques**
Data manipulation can involve a variety of operations, including filtering, sorting, aggregating, merging, and transforming data
### Missing data can be categorised into three types:
- **Missing Completely at Random (MCAR):** This means that the missing data
is not in any way related to any other data in the dataset.
- **Missing at Random (MAR):** This is when the missing data is somehow
related to other data in the dataset. 
- **Missing Not at Random (MNAR):** This occurs when there is a direct relation
to some missing data that the researchers haven’t measured.
**Standardization** is useful when you want to center your data with a mean of 0 and a standard deviation of 1, especially when your algorithm assumes that the data is normally distributed (e.g., linear regression, PCA).
## Methods/Techniques to remove the Missing Data: For Both Continuos Data and Categorical Data
- **distribution of data**: statistics using mean,max,etc
- **imputation of KNN**:
- **mputation with linear regression**
This imputation technique utilises variables from the observed data to replace the
missing values with predicted values from a regression model.

## FEATURE SCALING: NORMALISATION AND STANDARDISATION for both ML(regression-->gradient descent,eucladien distance and DL(KNN,RNN,CNN) )
Another common problem we encounter when trying to analyse data is having
different units of measurement for a particular variable.
### Methods for scaling
- **Normalization**
- **Standard Deviation**

#### Example of Standardization
#### Original Data:
[10,12,14,16,18]
**Mean (𝜇):** 10+12+14+16+1/5=14 <br>

**Standard Deviation (𝜎):** Let's assume σ=2.83 for simplicity.<br>
**Standardization:** <br>
Using the formula <br>
`𝑋standardized=𝑋−𝜇/𝜎`<br>
For X=10:
𝑋standardized=10−14/2.83 =>−1.41<br>
for X=12:
Xstandardized=12−14/2.83=>−2<br>
Similar for all others aply same formula 
#### Standardized Data:
[-1.41, -0.71, 0, 0.71, 1.41]

Now, the data has a **mean of 0** and a **standard deviation of 1**.

**Normalization** is useful when you need to scale data into a specific range (usually [0, 1]) and works well for distance-based algorithms and neural networks.<br>
Formula:
`Xnormalized= X−Xmin/Xmax−Xmin` <br>
​Where:
X is the original value,
𝑋min
  is the minimum value in the dataset,
𝑋max​
  is the maximum value in the dataset.

`When to Use:`
`When features have different units or scales` (e.g., height in cm vs. weight in kg).
`Commonly used when applying algorithms that rely on distance metrics`, such as K-Nearest Neighbors (KNN) or Support Vector Machines (SVM).
`In neural networks where the activation functions` (like Sigmoid) expect values between 0 and 1. <br>
**But how do we know when**
`to use which one? Quite simply, you must standardise the data when it follows a Gaussian distribution. If not, normalise the data.`

Normalization and Standardization (Standard Deviation) are two common techniques used in data preprocessing to make datasets more comparable and improve machine learning model performance
## EDA (Explore Data Analysis)
When you're a beginner and find yourself stuck during the Exploratory Data Analysis (EDA) of a dataset, it's completely normal. EDA is a crucial step in data science that involves summarizing, visualizing, and interpreting the data to uncover patterns, anomalies, and insights. Here are some steps and tips to guide you through the process:

1. **Understand the Dataset**:
   - **Read the Data**: Use tools like pandas in Python to load the dataset (`pd.read_csv()`, `pd.read_excel()`, etc.).
   - **Check the Structure**: Use `df.info()` in pandas to get an overview of the data types, missing values, and the number of entries for each column.
   - **View the Data**: Use `df.head()` and `df.tail()` to see the first and last few rows of the dataset.

2. **Clean the Data**:
   - **Handle Missing Values**: Decide whether to fill them in, drop them, or use imputation techniques.
   - **Remove Duplicates**: Use `df.drop_duplicates()` to ensure you're working with unique data points.
   - **Correct Data Types**: Convert columns to the appropriate data types using `df['column'].astype()`.

3. **Summarize the Data**:
   - **Descriptive Statistics**: Use `df.describe()` to get summary statistics for numerical columns.
   - **Categorical Data Summary**: Use `df['category_column'].value_counts()` to see the distribution of categorical variables.

4. **Visualize the Data**:
   - **Histograms**: Plot histograms for numerical columns to understand their distribution.
   - **Box Plots**: Use box plots to identify outliers and the spread of the data.
   - **Bar Charts/Pie Charts**: Visualize the frequency or proportion of categorical data.
   - **Scatter Plots**: Plot scatter plots to explore relationships between two numerical variables.

5. **Ask Questions and Formulate Hypotheses**:
   - Think about what questions you want to answer with the data.
   - Formulate hypotheses based on your initial observations and domain knowledge.

6. **Dig Deeper**:
   - **Correlation Analysis**: Use `df.corr()` to see how numerical variables are related to each other.
   - **Grouping and Aggregation**: Use `groupby()` and aggregation functions to summarize data by categories.
   - **Feature Engineering**: Create new features from existing ones if they might be useful for your analysis.

7. **Document Your Findings**:
   - Keep a notebook or document where you record your observations, hypotheses, and the steps you took during EDA.
   - This will help you stay organized and communicate your findings effectively.

8. **Iterate**:
   - EDA is often an iterative process. As you uncover new insights, you may need to go back and refine your approach or explore new angles.

9. **Seek Help**:
   - Don't hesitate to ask for help from more experienced colleagues, online forums, or tutorials.
   - The data science community is generally very helpful and supportive.
  

## Model Evaluation Metrics

### Classification Metrics

**Purpose:** To evaluate the performance of models predicting categorical outcomes.

**1. Accuracy:**

* **Definition:** The proportion of correctly predicted instances out of the total instances.
* **Formula:** $\frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}$
* **Use Case:** Suitable when classes are balanced.
* **Caution:** Can be misleading with imbalanced datasets.

**2. Precision:**

* **Definition:** The proportion of correctly predicted positive instances out of the total instances predicted as positive.
* **Formula:** $\frac{\text{True Positives (TP)}}{\text{True Positives (TP) + False Positives (FP)}}$
* **Use Case:** Important when minimizing false positives is crucial.

**3. Recall (Sensitivity/True Positive Rate):**

* **Definition:** The proportion of correctly predicted positive instances out of all actual positive instances.
* **Formula:** $\frac{\text{True Positives (TP)}}{\text{True Positives (TP) + False Negatives (FN)}}$
* **Use Case:** Important when minimizing false negatives is crucial.

**4. F1-Score:**

* **Definition:** The harmonic mean of precision and recall.
* **Formula:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
* **Use Case:** Balances precision and recall, useful when there's an uneven class distribution.

**5. Area Under the ROC Curve (AUC-ROC):**

* **Definition:** Measures the model's ability to distinguish between classes across various threshold settings.
* **Use Case:** Useful for binary classification problems, especially when class imbalance exists.
* **Interpretation:** AUC-ROC of 1 indicates perfect classification, 0.5 indicates random guessing.

**6. Confusion Matrix:**

* **Definition:** A table that visualizes the performance of a classification model by showing the counts of true positives, true negatives, false positives, and false negatives.
* **Use Case:** Provides a comprehensive view of model performance.

---

### Regression Metrics

**Purpose:** To evaluate the performance of models predicting continuous numerical outcomes.

**1. Mean Absolute Error (MAE):**

* **Definition:** The average absolute difference between predicted and actual values.
* **Formula:** $\frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$
* **Use Case:** Robust to outliers compared to MSE.

**2. Mean Squared Error (MSE):**

* **Definition:** The average squared difference between predicted and actual values.
* **Formula:** $\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
* **Use Case:** Sensitive to outliers due to squaring.

**3. Root Mean Squared Error (RMSE):**

* **Definition:** The square root of MSE.
* **Formula:** $\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$
* **Use Case:** Provides error in the same units as the target variable.

**4. R-squared (Coefficient of Determination):**

* **Definition:** Measures the proportion of the variance in the dependent variable that is predictable from the independent variable(s).
* **Formula:** $1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$
* **Use Case:** Indicates how well the model fits the data.
* **Interpretation:** R-squared of 1 indicates a perfect fit, 0 indicates no linear relationship.

**5. Adjusted R-squared:**

* **Definition:** Modified version of R-squared that adjusts for the number of predictors in the model.
* **Use Case:** Useful when comparing models with different numbers of predictors.

**6. Mean Absolute Percentage Error (MAPE):**

* **Definition:** Average absolute percentage difference between predicted and actual values.
* **Formula:** $\frac{1}{n} \sum_{i=1}^{n} |\frac{y_i - \hat{y}_i}{y_i}| \times 100$
* **Use Case:** Provides error as a percentage, making it easy to interpret.
* **Caution:** Undefined when actual values are zero.

Remember, the goal of EDA is to gain a deep understanding of your data so that you can make informed decisions about how to proceed with your analysis or modeling. Take your time, be methodical, and don't be afraid to explore different approaches.
