# DataScienceCourseMaterial
   - **[Course for Absolute beginers Website](https://jovian.com/learn/data-analysis-with-python-zero-to-pandas), [YouTube Channel](https://www.youtube.com/@jovianhq/playlists)**
   - **[Overview of Data Science](https://www.linkedin.com/pulse/data-science-methodology-step-by-step-guide-uzair-azmat-5tekf/?trackingId=DOxr4vPTsiNgGbFTdDijoQ%3D%3D)**
   - **[General Concepts](https://www.linkedin.com/pulse/complete-data-analysis-guide-python-uzair-azmat-uavvf/?trackingId=QNtfgWzo5XW04hwg3EPwUQ%3D%3D)**
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
  - [Statistics](https://www.youtube.com/watch?v=L_OLifCqxCQ&pp=ygULbWF0aGVtYXRpY3M%3D),  [Best to Know Statistics](https://www.youtube.com/watch?v=S7LvZZNq4ys&pp=ygULbWF0aGVtYXRpY3M%3D),  [Better to know Statistics](https://www.youtube.com/watch?v=bLZ-LSsQMCc&pp=ygULbWF0aGVtYXRpY3M%3D), [Learn More about Statistics](https://www.youtube.com/watch?v=LZzq1zSL1bs)
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
