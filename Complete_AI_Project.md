# 🚀 COMPLETE AI PROJECT: Customer Intelligence Platform

## **One Project to Master Everything: ML → Deep Learning → Modern AI**

> **This single comprehensive project covers ALL concepts from the three guides, taking you from raw data to a production-ready AI system with RAG, Agents, and LLM integration.**

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Overview](#architecture-overview)
3. [Phase 1: Data Analysis & Classical ML](#phase-1-data-analysis--classical-ml)
4. [Phase 2: Deep Learning Models](#phase-2-deep-learning-models)
5. [Phase 3: Modern AI - LLMs, RAG & Agents](#phase-3-modern-ai---llms-rag--agents)
6. [Phase 4: Production Deployment](#phase-4-production-deployment)
7. [Concepts Mapping](#concepts-mapping)

---

## Project Overview

### What We're Building

An **AI-Powered Customer Intelligence Platform** for an e-commerce company that:

1. **Analyzes customer data** (ML concepts)
2. **Predicts customer behavior** (Classification, Regression)
3. **Understands customer reviews** (NLP, Deep Learning)
4. **Provides intelligent chat support** (LLMs, RAG)
5. **Automates customer service tasks** (AI Agents)
6. **Deploys to production** (MLOps)

### Business Problems Solved

| Problem | Solution | Concepts Used |
|---------|----------|---------------|
| Customer churn | Churn prediction model | Classification, Ensemble |
| Revenue forecasting | Sales prediction | Regression, Time Series |
| Review understanding | Sentiment analysis | NLP, Transformers |
| Support automation | RAG chatbot | LLMs, Vector DB |
| Task automation | AI agents | Function calling |

### Tech Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TECH STACK                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DATA & ML                                                                  │
│  ├── Python, Pandas, NumPy, Scikit-learn                                   │
│  ├── XGBoost, LightGBM                                                     │
│  └── Matplotlib, Seaborn, Plotly                                           │
│                                                                             │
│  DEEP LEARNING                                                              │
│  ├── PyTorch / TensorFlow                                                  │
│  ├── Transformers (HuggingFace)                                            │
│  └── BERT, GPT-2, Custom Models                                            │
│                                                                             │
│  MODERN AI                                                                  │
│  ├── OpenAI API / Claude API / Local LLMs                                  │
│  ├── LangChain, LlamaIndex                                                 │
│  ├── ChromaDB / Pinecone                                                   │
│  └── PEFT, LoRA (Fine-tuning)                                              │
│                                                                             │
│  PRODUCTION                                                                 │
│  ├── FastAPI, Uvicorn                                                      │
│  ├── Docker, Docker Compose                                                │
│  ├── PostgreSQL, Redis                                                     │
│  └── MLflow, Prometheus, Grafana                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CUSTOMER INTELLIGENCE PLATFORM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         DATA LAYER                                   │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │   │
│  │  │ Customer │  │  Orders  │  │ Reviews  │  │ Support  │           │   │
│  │  │   Data   │  │   Data   │  │   Data   │  │  Tickets │           │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘           │   │
│  └───────┼─────────────┼─────────────┼─────────────┼─────────────────┘   │
│          │             │             │             │                      │
│          ↓             ↓             ↓             ↓                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      ML/DL PROCESSING LAYER                         │   │
│  │                                                                     │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│  │  │    Churn     │  │    Sales     │  │  Sentiment   │             │   │
│  │  │  Prediction  │  │  Forecasting │  │   Analysis   │             │   │
│  │  │  (XGBoost)   │  │   (LSTM)     │  │   (BERT)     │             │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘             │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                       │
│                                    ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       MODERN AI LAYER                               │   │
│  │                                                                     │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│  │  │   Vector DB  │  │  RAG Engine  │  │  AI Agent    │             │   │
│  │  │   (Chroma)   │  │ (LangChain)  │  │  (Tools)     │             │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘             │   │
│  │           │               │                  │                     │   │
│  │           └───────────────┼──────────────────┘                     │   │
│  │                           ↓                                        │   │
│  │                    ┌──────────────┐                                │   │
│  │                    │     LLM      │                                │   │
│  │                    │  (GPT-4/     │                                │   │
│  │                    │   Claude)    │                                │   │
│  │                    └──────────────┘                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                       │
│                                    ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       API & FRONTEND                                │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│  │  │   FastAPI    │  │  Dashboard   │  │   Chatbot    │             │   │
│  │  │   Backend    │  │   (Streamlit)│  │   Interface  │             │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Data Analysis & Classical ML

### 1.1 Project Setup

```python
# project_structure/
"""
customer_intelligence_platform/
├── data/
│   ├── raw/
│   │   ├── customers.csv
│   │   ├── orders.csv
│   │   ├── reviews.csv
│   │   └── support_tickets.csv
│   ├── processed/
│   └── embeddings/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_ml_models.ipynb
│   ├── 04_deep_learning.ipynb
│   └── 05_llm_integration.ipynb
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── feature_engineer.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── churn_model.py
│   │   ├── sales_forecaster.py
│   │   ├── sentiment_model.py
│   │   └── embeddings.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── customer_agent.py
│   │   └── tools.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── indexer.py
│   │   ├── retriever.py
│   │   └── generator.py
│   └── api/
│       ├── __init__.py
│       ├── main.py
│       ├── routes/
│       └── schemas/
├── tests/
├── configs/
├── docker/
├── requirements.txt
└── README.md
"""

# requirements.txt
"""
# Data Processing
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0

# Visualization
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0

# ML Models
xgboost==1.7.6
lightgbm==4.0.0

# Deep Learning
torch==2.0.1
transformers==4.31.0
datasets==2.14.0

# LLM & RAG
langchain==0.0.300
langchain-openai==0.0.2
llama-index==0.8.0
chromadb==0.4.0
openai==1.0.0

# Fine-tuning
peft==0.5.0
trl==0.7.0
bitsandbytes==0.41.0

# API & Production
fastapi==0.103.0
uvicorn==0.23.0
redis==4.6.0
sqlalchemy==2.0.0

# MLOps
mlflow==2.6.0
prometheus-client==0.17.0

# Utilities
python-dotenv==1.0.0
pydantic==2.0.0
httpx==0.24.0
"""
```

### 1.2 Data Generation (Synthetic Dataset)

```python
# src/data/generate_data.py
"""
Generate synthetic e-commerce data for the project.
This covers: Data Collection, Understanding Data Structure
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)

def generate_customers(n_customers=10000):
    """
    Generate customer data.
    Concepts: Data Collection, Feature Types (Numerical, Categorical)
    """
    
    customers = pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'age': np.random.randint(18, 70, n_customers),
        'gender': np.random.choice(['M', 'F', 'Other'], n_customers, p=[0.48, 0.48, 0.04]),
        'country': np.random.choice(
            ['USA', 'UK', 'Germany', 'France', 'Canada', 'Australia'],
            n_customers,
            p=[0.35, 0.20, 0.15, 0.12, 0.10, 0.08]
        ),
        'registration_date': [
            datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1400))
            for _ in range(n_customers)
        ],
        'account_type': np.random.choice(
            ['Basic', 'Premium', 'VIP'],
            n_customers,
            p=[0.60, 0.30, 0.10]
        ),
        'email_subscribed': np.random.choice([True, False], n_customers, p=[0.7, 0.3]),
        'total_spent': np.random.exponential(500, n_customers).round(2),
        'num_orders': np.random.poisson(5, n_customers),
        'avg_order_value': np.random.exponential(100, n_customers).round(2),
        'days_since_last_order': np.random.exponential(30, n_customers).astype(int),
        'support_tickets': np.random.poisson(1, n_customers),
        'churned': np.random.choice([0, 1], n_customers, p=[0.85, 0.15])
    })
    
    return customers

def generate_orders(customers_df, n_orders=50000):
    """
    Generate order data.
    Concepts: Time Series Data, Transaction Data
    """
    
    customer_ids = customers_df['customer_id'].values
    
    orders = pd.DataFrame({
        'order_id': range(1, n_orders + 1),
        'customer_id': np.random.choice(customer_ids, n_orders),
        'order_date': [
            datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1400))
            for _ in range(n_orders)
        ],
        'order_value': np.random.exponential(80, n_orders).round(2),
        'num_items': np.random.randint(1, 10, n_orders),
        'category': np.random.choice(
            ['Electronics', 'Clothing', 'Home', 'Books', 'Sports', 'Beauty'],
            n_orders
        ),
        'payment_method': np.random.choice(
            ['Credit Card', 'PayPal', 'Bank Transfer', 'Crypto'],
            n_orders,
            p=[0.50, 0.30, 0.15, 0.05]
        ),
        'discount_applied': np.random.choice([True, False], n_orders, p=[0.3, 0.7]),
        'returned': np.random.choice([True, False], n_orders, p=[0.08, 0.92])
    })
    
    return orders

def generate_reviews(orders_df, n_reviews=20000):
    """
    Generate review data with text.
    Concepts: NLP Data, Sentiment Analysis
    """
    
    positive_templates = [
        "Absolutely love this product! {reason}. Would definitely recommend to others.",
        "Great quality and fast shipping. {reason}. Very satisfied with my purchase.",
        "Exceeded my expectations! {reason}. Will buy again.",
        "Perfect! {reason}. Exactly what I was looking for.",
        "Amazing value for money. {reason}. Very happy customer here.",
    ]
    
    neutral_templates = [
        "It's okay, nothing special. {reason}. Does the job.",
        "Average product. {reason}. Not bad but not great either.",
        "Decent quality for the price. {reason}. Might buy again.",
        "As expected, nothing more. {reason}. Acceptable.",
    ]
    
    negative_templates = [
        "Very disappointed with this purchase. {reason}. Would not recommend.",
        "Poor quality product. {reason}. Waste of money.",
        "Arrived damaged and customer service was unhelpful. {reason}. Never again.",
        "Does not match the description. {reason}. Returning immediately.",
        "Terrible experience. {reason}. Stay away from this product.",
    ]
    
    reasons = {
        'positive': [
            "The material feels premium",
            "Works perfectly",
            "Arrived earlier than expected",
            "Easy to use",
            "Great design",
        ],
        'neutral': [
            "Takes some getting used to",
            "Packaging could be better",
            "Size runs a bit small",
            "Instructions unclear",
        ],
        'negative': [
            "Broke after a week",
            "Completely different from pictures",
            "Missing parts",
            "Poor craftsmanship",
        ]
    }
    
    reviews = []
    order_ids = orders_df['order_id'].values
    
    for i in range(n_reviews):
        sentiment = np.random.choice(['positive', 'neutral', 'negative'], p=[0.60, 0.25, 0.15])
        
        if sentiment == 'positive':
            template = random.choice(positive_templates)
            rating = random.randint(4, 5)
        elif sentiment == 'neutral':
            template = random.choice(neutral_templates)
            rating = random.randint(3, 4)
        else:
            template = random.choice(negative_templates)
            rating = random.randint(1, 2)
        
        reason = random.choice(reasons[sentiment])
        review_text = template.format(reason=reason)
        
        reviews.append({
            'review_id': i + 1,
            'order_id': random.choice(order_ids),
            'rating': rating,
            'review_text': review_text,
            'sentiment': sentiment,
            'review_date': datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1400)),
            'helpful_votes': np.random.poisson(3),
            'verified_purchase': random.choice([True, False], p=[0.8, 0.2])
        })
    
    return pd.DataFrame(reviews)

def generate_support_tickets(customers_df, n_tickets=5000):
    """
    Generate support ticket data.
    Concepts: Text Data, Classification Categories
    """
    
    categories = ['Order Issue', 'Product Defect', 'Refund Request', 
                  'Account Problem', 'Shipping Delay', 'General Inquiry']
    
    ticket_templates = {
        'Order Issue': [
            "I haven't received my order #{order_id} yet. It's been {days} days.",
            "Wrong item received in order #{order_id}. Please help.",
            "Order #{order_id} shows delivered but I never got it.",
        ],
        'Product Defect': [
            "The product from order #{order_id} stopped working after {days} days.",
            "Item arrived damaged. Order #{order_id}.",
            "Quality issue with my recent purchase. Very disappointed.",
        ],
        'Refund Request': [
            "I want a refund for order #{order_id}. Product not as described.",
            "Requesting full refund. Order #{order_id} was never delivered.",
            "Please process refund for my returned item.",
        ],
        'Account Problem': [
            "Can't log into my account. Password reset not working.",
            "Need to update my email address on file.",
            "Having trouble with payment method on my account.",
        ],
        'Shipping Delay': [
            "My order #{order_id} was supposed to arrive {days} days ago.",
            "Tracking hasn't updated in a week for order #{order_id}.",
            "Express shipping but still waiting after {days} days.",
        ],
        'General Inquiry': [
            "When will item X be back in stock?",
            "Do you ship to {country}?",
            "What's your return policy?",
        ]
    }
    
    tickets = []
    customer_ids = customers_df['customer_id'].values
    
    for i in range(n_tickets):
        category = random.choice(categories)
        template = random.choice(ticket_templates[category])
        
        ticket_text = template.format(
            order_id=random.randint(1000, 9999),
            days=random.randint(1, 30),
            country=random.choice(['UAE', 'Brazil', 'Japan'])
        )
        
        tickets.append({
            'ticket_id': i + 1,
            'customer_id': random.choice(customer_ids),
            'category': category,
            'subject': f"{category} - Ticket #{i+1}",
            'description': ticket_text,
            'priority': random.choice(['Low', 'Medium', 'High', 'Urgent'], p=[0.3, 0.4, 0.2, 0.1]),
            'status': random.choice(['Open', 'In Progress', 'Resolved', 'Closed'], p=[0.2, 0.2, 0.3, 0.3]),
            'created_at': datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1400)),
            'resolution_time_hours': np.random.exponential(24) if random.random() > 0.3 else None
        })
    
    return pd.DataFrame(tickets)

# Generate all data
if __name__ == "__main__":
    print("Generating synthetic data...")
    
    customers = generate_customers(10000)
    orders = generate_orders(customers, 50000)
    reviews = generate_reviews(orders, 20000)
    tickets = generate_support_tickets(customers, 5000)
    
    # Save data
    customers.to_csv('data/raw/customers.csv', index=False)
    orders.to_csv('data/raw/orders.csv', index=False)
    reviews.to_csv('data/raw/reviews.csv', index=False)
    tickets.to_csv('data/raw/support_tickets.csv', index=False)
    
    print(f"Generated:")
    print(f"  - {len(customers)} customers")
    print(f"  - {len(orders)} orders")
    print(f"  - {len(reviews)} reviews")
    print(f"  - {len(tickets)} support tickets")
```

### 1.3 Exploratory Data Analysis (EDA)

```python
# notebooks/01_eda.ipynb
"""
Complete EDA covering all concepts from ML guide:
- Data Assessment
- Quality Dimensions
- Univariate/Bivariate/Multivariate Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STEP 1: DATA LOADING & INITIAL ASSESSMENT
# =============================================================================

# Load data
customers = pd.read_csv('data/raw/customers.csv')
orders = pd.read_csv('data/raw/orders.csv')
reviews = pd.read_csv('data/raw/reviews.csv')
tickets = pd.read_csv('data/raw/support_tickets.csv')

print("=" * 60)
print("DATA OVERVIEW")
print("=" * 60)

datasets = {
    'Customers': customers,
    'Orders': orders,
    'Reviews': reviews,
    'Tickets': tickets
}

for name, df in datasets.items():
    print(f"\n{name}:")
    print(f"  Shape: {df.shape}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# =============================================================================
# STEP 2: DATA QUALITY ASSESSMENT
# =============================================================================

def assess_data_quality(df, name):
    """
    Assess data quality using the 4 dimensions:
    1. Completeness
    2. Validity
    3. Accuracy
    4. Consistency
    """
    print(f"\n{'='*60}")
    print(f"DATA QUALITY ASSESSMENT: {name}")
    print(f"{'='*60}")
    
    # 1. COMPLETENESS - Check for missing values
    print("\n1. COMPLETENESS (Missing Values):")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing'] > 0])
    if missing.sum() == 0:
        print("   ✅ No missing values found!")
    
    # 2. VALIDITY - Check data types and ranges
    print("\n2. VALIDITY (Data Types):")
    print(df.dtypes)
    
    # 3. ACCURACY - Check for outliers using IQR
    print("\n3. ACCURACY (Outliers in Numeric Columns):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        if outliers > 0:
            print(f"   {col}: {outliers} outliers ({outliers/len(df)*100:.2f}%)")
    
    # 4. CONSISTENCY - Check for duplicates
    print("\n4. CONSISTENCY (Duplicates):")
    duplicates = df.duplicated().sum()
    print(f"   Duplicate rows: {duplicates}")
    
    return missing_df

# Assess all datasets
for name, df in datasets.items():
    assess_data_quality(df, name)

# =============================================================================
# STEP 3: COLUMN CLASSIFICATION
# =============================================================================

def classify_columns(df):
    """
    Classify columns into:
    - Numerical (Continuous, Discrete)
    - Categorical (Nominal, Ordinal)
    - DateTime
    - Text
    """
    classification = {
        'numerical_continuous': [],
        'numerical_discrete': [],
        'categorical_nominal': [],
        'categorical_ordinal': [],
        'datetime': [],
        'text': []
    }
    
    for col in df.columns:
        dtype = df[col].dtype
        unique_ratio = df[col].nunique() / len(df)
        
        if 'date' in col.lower() or 'time' in col.lower():
            classification['datetime'].append(col)
        elif dtype in ['object', 'string']:
            if df[col].str.len().mean() > 50:  # Likely text
                classification['text'].append(col)
            else:
                classification['categorical_nominal'].append(col)
        elif dtype in ['int64', 'int32']:
            if unique_ratio < 0.05:  # Few unique values
                classification['categorical_nominal'].append(col)
            else:
                classification['numerical_discrete'].append(col)
        elif dtype in ['float64', 'float32']:
            classification['numerical_continuous'].append(col)
    
    return classification

print("\n" + "="*60)
print("COLUMN CLASSIFICATION - Customers")
print("="*60)
customer_cols = classify_columns(customers)
for category, cols in customer_cols.items():
    if cols:
        print(f"\n{category.upper()}:")
        print(f"  {cols}")

# =============================================================================
# STEP 4: UNIVARIATE ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("UNIVARIATE ANALYSIS")
print("="*60)

# 4.1 Numerical Analysis
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Univariate Analysis - Numerical Features', fontsize=14)

numerical_cols = ['age', 'total_spent', 'num_orders', 
                  'avg_order_value', 'days_since_last_order', 'support_tickets']

for idx, col in enumerate(numerical_cols):
    ax = axes[idx // 3, idx % 3]
    
    # Histogram with KDE
    sns.histplot(customers[col], kde=True, ax=ax)
    
    # Add statistics
    mean_val = customers[col].mean()
    median_val = customers[col].median()
    ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
    ax.legend()
    ax.set_title(f'Distribution of {col}')

plt.tight_layout()
plt.savefig('data/processed/univariate_numerical.png', dpi=150)
plt.show()

# Descriptive Statistics
print("\nDescriptive Statistics - Numerical Features:")
print(customers[numerical_cols].describe())

# Skewness Analysis
print("\nSkewness Analysis:")
for col in numerical_cols:
    skew = customers[col].skew()
    if abs(skew) < 0.5:
        interpretation = "approximately symmetric"
    elif skew > 0:
        interpretation = "right-skewed (positive)"
    else:
        interpretation = "left-skewed (negative)"
    print(f"  {col}: {skew:.3f} ({interpretation})")

# 4.2 Categorical Analysis
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Univariate Analysis - Categorical Features', fontsize=14)

categorical_cols = ['gender', 'country', 'account_type', 'churned']

for idx, col in enumerate(categorical_cols):
    ax = axes[idx // 2, idx % 2]
    
    # Count plot
    value_counts = customers[col].value_counts()
    bars = ax.bar(value_counts.index.astype(str), value_counts.values)
    
    # Add percentage labels
    total = len(customers)
    for bar, count in zip(bars, value_counts.values):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{pct:.1f}%', ha='center', va='bottom')
    
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')

plt.tight_layout()
plt.savefig('data/processed/univariate_categorical.png', dpi=150)
plt.show()

# =============================================================================
# STEP 5: BIVARIATE ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("BIVARIATE ANALYSIS")
print("="*60)

# 5.1 Numerical vs Numerical (Correlation)
print("\nCorrelation Matrix (Numerical-Numerical):")
correlation_matrix = customers[numerical_cols].corr()
print(correlation_matrix.round(3))

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
            fmt='.2f', square=True)
plt.title('Correlation Heatmap - Customer Features')
plt.tight_layout()
plt.savefig('data/processed/correlation_heatmap.png', dpi=150)
plt.show()

# 5.2 Numerical vs Categorical (Churn Analysis)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Feature Distribution by Churn Status', fontsize=14)

for idx, col in enumerate(numerical_cols):
    ax = axes[idx // 3, idx % 3]
    
    # Box plot by churn status
    sns.boxplot(data=customers, x='churned', y=col, ax=ax)
    ax.set_title(f'{col} by Churn Status')
    ax.set_xticklabels(['Not Churned', 'Churned'])

plt.tight_layout()
plt.savefig('data/processed/bivariate_churn.png', dpi=150)
plt.show()

# Statistical Tests (t-test for numerical vs binary)
print("\nStatistical Tests (Churn vs Features):")
for col in numerical_cols:
    churned = customers[customers['churned'] == 1][col]
    not_churned = customers[customers['churned'] == 0][col]
    
    t_stat, p_value = stats.ttest_ind(churned, not_churned)
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    print(f"  {col}: t={t_stat:.3f}, p={p_value:.4f} {significance}")

# 5.3 Categorical vs Categorical (Cross-tabulation)
print("\nCross-tabulation: Account Type vs Churn")
cross_tab = pd.crosstab(customers['account_type'], customers['churned'], 
                        normalize='index') * 100
print(cross_tab.round(2))

# Chi-square test
chi2, p_value, dof, expected = stats.chi2_contingency(
    pd.crosstab(customers['account_type'], customers['churned'])
)
print(f"\nChi-square test: χ²={chi2:.3f}, p={p_value:.4f}")

# =============================================================================
# STEP 6: MULTIVARIATE ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("MULTIVARIATE ANALYSIS")
print("="*60)

# Pair plot for key features
selected_features = ['age', 'total_spent', 'num_orders', 'avg_order_value', 'churned']
sns.pairplot(customers[selected_features], hue='churned', diag_kind='kde')
plt.suptitle('Pair Plot - Key Features by Churn Status', y=1.02)
plt.savefig('data/processed/pairplot.png', dpi=150)
plt.show()

# =============================================================================
# STEP 7: TIME SERIES ANALYSIS (Orders)
# =============================================================================

print("\n" + "="*60)
print("TIME SERIES ANALYSIS - Orders")
print("="*60)

# Convert to datetime
orders['order_date'] = pd.to_datetime(orders['order_date'])

# Daily order volume
daily_orders = orders.groupby(orders['order_date'].dt.date).agg({
    'order_id': 'count',
    'order_value': 'sum'
}).rename(columns={'order_id': 'num_orders', 'order_value': 'revenue'})

# Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(daily_orders.index, daily_orders['num_orders'])
axes[0].set_title('Daily Order Volume')
axes[0].set_ylabel('Number of Orders')

axes[1].plot(daily_orders.index, daily_orders['revenue'], color='green')
axes[1].set_title('Daily Revenue')
axes[1].set_ylabel('Revenue ($)')

plt.tight_layout()
plt.savefig('data/processed/time_series.png', dpi=150)
plt.show()

# =============================================================================
# STEP 8: TEXT DATA ANALYSIS (Reviews)
# =============================================================================

print("\n" + "="*60)
print("TEXT DATA ANALYSIS - Reviews")
print("="*60)

# Review length distribution
reviews['review_length'] = reviews['review_text'].str.len()
reviews['word_count'] = reviews['review_text'].str.split().str.len()

print("\nReview Text Statistics:")
print(reviews[['review_length', 'word_count']].describe())

# Sentiment distribution
print("\nSentiment Distribution:")
print(reviews['sentiment'].value_counts(normalize=True).round(3) * 100)

# Rating distribution
plt.figure(figsize=(10, 5))
sns.countplot(data=reviews, x='rating', hue='sentiment')
plt.title('Rating Distribution by Sentiment')
plt.savefig('data/processed/sentiment_distribution.png', dpi=150)
plt.show()

print("\n" + "="*60)
print("EDA COMPLETE!")
print("="*60)
```

### 1.4 Feature Engineering

```python
# src/data/feature_engineer.py
"""
Feature Engineering covering all concepts:
- Feature Creation
- Feature Transformation
- Feature Encoding
- Feature Scaling
- Feature Selection
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from datetime import datetime

class FeatureEngineer:
    """
    Comprehensive feature engineering pipeline.
    
    Concepts covered:
    - Numerical transformations
    - Categorical encoding
    - Feature creation
    - Feature scaling
    - Feature selection (Variance, Correlation, PCA)
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.selected_features = None
        
    def create_customer_features(self, customers_df, orders_df):
        """
        Create derived features from customer and order data.
        
        Concepts: Feature Engineering, Domain Knowledge
        """
        df = customers_df.copy()
        
        # Convert dates
        df['registration_date'] = pd.to_datetime(df['registration_date'])
        
        # =================================================================
        # TEMPORAL FEATURES
        # =================================================================
        
        # Account age in days
        reference_date = datetime(2024, 1, 1)
        df['account_age_days'] = (reference_date - df['registration_date']).dt.days
        
        # Registration month and year
        df['registration_month'] = df['registration_date'].dt.month
        df['registration_year'] = df['registration_date'].dt.year
        df['registration_quarter'] = df['registration_date'].dt.quarter
        
        # Day of week registered (cyclical encoding)
        df['registration_dow'] = df['registration_date'].dt.dayofweek
        df['registration_dow_sin'] = np.sin(2 * np.pi * df['registration_dow'] / 7)
        df['registration_dow_cos'] = np.cos(2 * np.pi * df['registration_dow'] / 7)
        
        # =================================================================
        # BEHAVIORAL FEATURES
        # =================================================================
        
        # Order frequency (orders per month)
        df['order_frequency'] = df['num_orders'] / (df['account_age_days'] / 30 + 1)
        
        # Spending velocity (spend per month)
        df['spending_velocity'] = df['total_spent'] / (df['account_age_days'] / 30 + 1)
        
        # Recency score (inverse of days since last order)
        df['recency_score'] = 1 / (df['days_since_last_order'] + 1)
        
        # RFM-like features
        df['rfm_score'] = (
            df['recency_score'] * 100 + 
            df['order_frequency'] * 10 + 
            df['total_spent'] / 100
        )
        
        # =================================================================
        # RATIO FEATURES
        # =================================================================
        
        # Support ticket ratio (tickets per order)
        df['ticket_per_order'] = df['support_tickets'] / (df['num_orders'] + 1)
        
        # Average vs total spending ratio
        df['avg_to_total_ratio'] = df['avg_order_value'] / (df['total_spent'] + 1)
        
        # =================================================================
        # BINNING FEATURES
        # =================================================================
        
        # Age groups
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 25, 35, 50, 65, 100],
            labels=['18-25', '26-35', '36-50', '51-65', '65+']
        )
        
        # Spending tiers
        df['spending_tier'] = pd.qcut(
            df['total_spent'],
            q=5,
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        # =================================================================
        # AGGREGATE FEATURES FROM ORDERS
        # =================================================================
        
        if orders_df is not None:
            orders_agg = orders_df.groupby('customer_id').agg({
                'order_value': ['mean', 'std', 'min', 'max'],
                'num_items': ['mean', 'sum'],
                'returned': 'sum',
                'discount_applied': 'sum'
            }).reset_index()
            
            # Flatten column names
            orders_agg.columns = ['customer_id', 'order_mean', 'order_std', 
                                 'order_min', 'order_max', 'items_mean', 
                                 'items_total', 'returns_count', 'discounts_used']
            
            # Fill NaN std with 0
            orders_agg['order_std'] = orders_agg['order_std'].fillna(0)
            
            # Merge with customers
            df = df.merge(orders_agg, on='customer_id', how='left')
            
            # Return rate
            df['return_rate'] = df['returns_count'] / (df['num_orders'] + 1)
            
            # Discount usage rate
            df['discount_rate'] = df['discounts_used'] / (df['num_orders'] + 1)
        
        return df
    
    def encode_categorical(self, df, columns, method='onehot'):
        """
        Encode categorical variables.
        
        Methods:
        - onehot: One-hot encoding (creates dummy variables)
        - label: Label encoding (ordinal integers)
        - target: Target encoding (mean of target per category)
        """
        df_encoded = df.copy()
        
        if method == 'onehot':
            df_encoded = pd.get_dummies(df_encoded, columns=columns, drop_first=True)
            
        elif method == 'label':
            for col in columns:
                le = LabelEncoder()
                df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = le
                
        elif method == 'target':
            # Requires target column
            pass
        
        return df_encoded
    
    def scale_features(self, df, columns, method='standard'):
        """
        Scale numerical features.
        
        Methods:
        - standard: StandardScaler (z-score normalization)
        - minmax: MinMaxScaler (0-1 normalization)
        - robust: RobustScaler (uses median and IQR)
        """
        df_scaled = df.copy()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        
        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
        self.scalers['main'] = scaler
        
        return df_scaled
    
    def select_features(self, X, y, method='kbest', k=20):
        """
        Feature selection methods.
        
        Concepts:
        - Variance Threshold
        - Correlation-based
        - SelectKBest (statistical tests)
        - PCA (dimensionality reduction)
        """
        from sklearn.feature_selection import VarianceThreshold
        
        if method == 'variance':
            selector = VarianceThreshold(threshold=0.01)
            X_selected = selector.fit_transform(X)
            selected_mask = selector.get_support()
            self.selected_features = X.columns[selected_mask].tolist()
            
        elif method == 'kbest':
            selector = SelectKBest(f_classif, k=k)
            X_selected = selector.fit_transform(X, y)
            selected_mask = selector.get_support()
            self.selected_features = X.columns[selected_mask].tolist()
            
        elif method == 'correlation':
            # Remove highly correlated features
            corr_matrix = X.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [col for col in upper_tri.columns 
                      if any(upper_tri[col] > 0.95)]
            X_selected = X.drop(columns=to_drop)
            self.selected_features = X_selected.columns.tolist()
            
        elif method == 'pca':
            pca = PCA(n_components=k)
            X_selected = pca.fit_transform(X)
            print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
            self.selected_features = [f'PC{i+1}' for i in range(k)]
        
        return X_selected, self.selected_features
    
    def prepare_for_modeling(self, df, target_col='churned'):
        """
        Complete pipeline to prepare data for modeling.
        """
        # Separate features and target
        X = df.drop(columns=[target_col, 'customer_id', 'registration_date'], errors='ignore')
        y = df[target_col]
        
        # Handle remaining categorical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Fill any remaining NaN
        X = X.fillna(0)
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        
        return X, y


# Example usage
if __name__ == "__main__":
    # Load data
    customers = pd.read_csv('data/raw/customers.csv')
    orders = pd.read_csv('data/raw/orders.csv')
    
    # Initialize engineer
    fe = FeatureEngineer()
    
    # Create features
    df_features = fe.create_customer_features(customers, orders)
    
    # Prepare for modeling
    X, y = fe.prepare_for_modeling(df_features)
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")
    
    # Save processed data
    X.to_csv('data/processed/features.csv', index=False)
    y.to_csv('data/processed/target.csv', index=False)
```

### 1.5 Classical ML Models

```python
# src/models/churn_model.py
"""
Complete ML Pipeline for Churn Prediction covering:
- Multiple algorithms (Logistic Regression to XGBoost)
- Hyperparameter tuning
- Cross-validation
- Model evaluation
- Ensemble methods
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import joblib
import warnings
warnings.filterwarnings('ignore')


class ChurnPredictor:
    """
    Complete churn prediction pipeline.
    
    Concepts covered:
    - All major ML algorithms
    - Ensemble methods (Bagging, Boosting, Stacking)
    - Hyperparameter tuning
    - Cross-validation
    - Model evaluation metrics
    """
    
    def __init__(self, experiment_name="churn_prediction"):
        self.models = {}
        self.best_model = None
        self.results = {}
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        
    def get_models(self):
        """
        Define all models to train.
        
        Concepts: Algorithm Selection, Hyperparameters
        """
        models = {
            # Linear Models
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
            
            # Tree-based Models
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10,
                random_state=42
            ),
            
            # Ensemble - Bagging
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            
            # Ensemble - Boosting
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            ),
            
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            
            'LightGBM': LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            
            # Other Models
            'SVM': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            
            'KNN': KNeighborsClassifier(
                n_neighbors=5
            ),
            
            'Naive Bayes': GaussianNB()
        }
        
        return models
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Train all models and evaluate performance.
        
        Concepts: Model Training, Evaluation Metrics
        """
        models = self.get_models()
        results = []
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            with mlflow.start_run(run_name=name):
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                # Calculate metrics
                metrics = {
                    'model': name,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba)
                }
                
                results.append(metrics)
                self.models[name] = model
                
                # Log to MLflow
                mlflow.log_params(model.get_params())
                mlflow.log_metrics({
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'roc_auc': metrics['roc_auc']
                })
                mlflow.sklearn.log_model(model, name.replace(' ', '_'))
                
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1 Score: {metrics['f1']:.4f}")
                print(f"  ROC AUC:  {metrics['roc_auc']:.4f}")
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation for all models.
        
        Concepts: Cross-Validation, Stratified K-Fold
        """
        models = self.get_models()
        cv_results = []
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        for name, model in models.items():
            print(f"\nCross-validating {name}...")
            
            scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
            
            cv_results.append({
                'model': name,
                'cv_mean': scores.mean(),
                'cv_std': scores.std(),
                'cv_scores': scores
            })
            
            print(f"  F1 Score: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
        
        return pd.DataFrame(cv_results)
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='XGBoost'):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Concepts: Hyperparameter Optimization, Grid Search
        """
        param_grids = {
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        }
        
        if model_name not in param_grids:
            print(f"No param grid defined for {model_name}")
            return None
        
        print(f"\nTuning {model_name}...")
        
        model = self.get_models()[model_name]
        param_grid = param_grids[model_name]
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best F1 score: {grid_search.best_score_:.4f}")
        
        self.best_model = grid_search.best_estimator_
        return grid_search
    
    def create_ensemble(self, X_train, y_train, X_test, y_test):
        """
        Create ensemble models.
        
        Concepts: Voting Classifier, Stacking
        """
        # Voting Ensemble
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('xgb', XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')),
                ('lgb', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
            ],
            voting='soft'
        )
        
        # Stacking Ensemble
        stacking_clf = StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('xgb', XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')),
                ('lgb', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
            ],
            final_estimator=LogisticRegression(),
            cv=5
        )
        
        ensembles = {
            'Voting Ensemble': voting_clf,
            'Stacking Ensemble': stacking_clf
        }
        
        ensemble_results = []
        
        for name, model in ensembles.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'model': name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            ensemble_results.append(metrics)
            self.models[name] = model
            
            print(f"  F1 Score: {metrics['f1']:.4f}")
            print(f"  ROC AUC:  {metrics['roc_auc']:.4f}")
        
        return pd.DataFrame(ensemble_results)
    
    def plot_results(self, save_path='data/processed/'):
        """
        Visualize model comparison results.
        
        Concepts: Model Comparison, Visualization
        """
        if self.results is None or len(self.results) == 0:
            print("No results to plot. Run train_and_evaluate first.")
            return
        
        # Model Comparison Bar Plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            sorted_df = self.results.sort_values(metric, ascending=True)
            
            bars = ax.barh(sorted_df['model'], sorted_df[metric])
            ax.set_xlabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} Comparison')
            ax.set_xlim(0, 1)
            
            # Add value labels
            for bar, val in zip(bars, sorted_df[metric]):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}model_comparison.png', dpi=150)
        plt.show()
        
    def plot_confusion_matrix(self, model_name, X_test, y_test, save_path='data/processed/'):
        """
        Plot confusion matrix for a specific model.
        
        Concepts: Confusion Matrix, Classification Report
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Not Churned', 'Churned'],
                   yticklabels=['Not Churned', 'Churned'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'{save_path}confusion_matrix_{model_name.replace(" ", "_")}.png', dpi=150)
        plt.show()
        
        # Classification Report
        print(f"\nClassification Report - {model_name}:")
        print(classification_report(y_test, y_pred, target_names=['Not Churned', 'Churned']))
    
    def plot_roc_curves(self, X_test, y_test, save_path='data/processed/'):
        """
        Plot ROC curves for all models.
        
        Concepts: ROC Curve, AUC
        """
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc = roc_auc_score(y_test, y_pred_proba)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(f'{save_path}roc_curves.png', dpi=150)
        plt.show()
    
    def get_feature_importance(self, model_name='XGBoost', feature_names=None, top_n=20):
        """
        Get and plot feature importance.
        
        Concepts: Feature Importance, Model Interpretability
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print(f"Model {model_name} doesn't have feature importance.")
            return
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'data/processed/feature_importance_{model_name.replace(" ", "_")}.png', dpi=150)
        plt.show()
        
        return importance_df
    
    def save_model(self, model_name, path='models/'):
        """Save a trained model."""
        if model_name not in self.models:
            print(f"Model {model_name} not found.")
            return
        
        joblib.dump(self.models[model_name], f'{path}{model_name.replace(" ", "_")}.joblib')
        print(f"Model saved to {path}{model_name.replace(' ', '_')}.joblib")
    
    def load_model(self, path):
        """Load a saved model."""
        return joblib.load(path)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Load processed data
    X = pd.read_csv('data/processed/features.csv')
    y = pd.read_csv('data/processed/target.csv').values.ravel()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Churn rate in training: {y_train.mean():.3f}")
    print(f"Churn rate in test: {y_test.mean():.3f}")
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Train and evaluate all models
    results = predictor.train_and_evaluate(X_train, X_test, y_train, y_test)
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    print(results.sort_values('f1', ascending=False).to_string(index=False))
    
    # Cross-validation
    cv_results = predictor.cross_validate(X, y)
    
    # Create ensemble models
    ensemble_results = predictor.create_ensemble(X_train, y_train, X_test, y_test)
    
    # Hyperparameter tuning for best model
    grid_search = predictor.hyperparameter_tuning(X_train, y_train, 'XGBoost')
    
    # Plot results
    predictor.plot_results()
    predictor.plot_confusion_matrix('XGBoost', X_test, y_test)
    predictor.plot_roc_curves(X_test, y_test)
    predictor.get_feature_importance('XGBoost', feature_names=X.columns.tolist())
    
    # Save best model
    predictor.save_model('XGBoost', 'models/')
    
    print("\n" + "="*60)
    print("PHASE 1 COMPLETE: Classical ML Models Trained!")
    print("="*60)
```

---

## Phase 2: Deep Learning Models

### 2.1 Sentiment Analysis with BERT

```python
# src/models/sentiment_model.py
"""
Deep Learning for Sentiment Analysis covering:
- Text preprocessing for transformers
- BERT fine-tuning
- Custom PyTorch training loop
- Evaluation and inference
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class ReviewDataset(Dataset):
    """
    Custom PyTorch Dataset for reviews.
    
    Concepts: Data Loading, Tokenization, Padding
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class SentimentClassifier:
    """
    BERT-based sentiment classifier.
    
    Concepts covered:
    - Transfer Learning
    - Fine-tuning transformers
    - Learning rate scheduling
    - Model evaluation
    """
    
    def __init__(self, model_name='bert-base-uncased', num_labels=3):
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.to(device)
        
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
    def prepare_data(self, texts, labels, test_size=0.2, batch_size=16):
        """
        Prepare data loaders.
        
        Concepts: Train/Val Split, Batch Loading
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = ReviewDataset(
            X_train.values, y_train.values, 
            self.tokenizer
        )
        val_dataset = ReviewDataset(
            X_val.values, y_val.values,
            self.tokenizer
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        return self.train_loader, self.val_loader
    
    def train(self, epochs=3, learning_rate=2e-5, warmup_steps=0):
        """
        Train the model.
        
        Concepts:
        - Forward propagation
        - Backpropagation
        - Gradient descent (AdamW)
        - Learning rate scheduling
        """
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Scheduler
        total_steps = len(self.train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 40)
            
            # Training phase
            self.model.train()
            train_loss = 0
            
            progress_bar = tqdm(self.train_loader, desc="Training")
            
            for batch in progress_bar:
                # Move to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping (prevent exploding gradients)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(self.train_loader)
            
            # Validation phase
            val_loss, val_accuracy = self.evaluate()
            
            # Save history
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
    
    def evaluate(self):
        """
        Evaluate the model on validation set.
        
        Concepts: Inference, Metrics Calculation
        """
        self.model.eval()
        val_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                
                # Get predictions
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(self.val_loader)
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        
        return avg_val_loss, accuracy
    
    def predict(self, text):
        """
        Predict sentiment for a single text.
        
        Concepts: Inference, Softmax
        """
        self.model.eval()
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
        
        labels = ['negative', 'neutral', 'positive']
        
        return {
            'sentiment': labels[prediction],
            'confidence': confidence,
            'probabilities': {
                labels[i]: probabilities[0][i].item() 
                for i in range(len(labels))
            }
        }
    
    def plot_training_history(self, save_path='data/processed/'):
        """Plot training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        
        # Accuracy
        axes[1].plot(self.history['val_accuracy'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}bert_training_history.png', dpi=150)
        plt.show()
    
    def save_model(self, path='models/sentiment_bert'):
        """Save model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path='models/sentiment_bert'):
        """Load model and tokenizer."""
        self.model = BertForSequenceClassification.from_pretrained(path)
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model.to(device)
        print(f"Model loaded from {path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Load reviews data
    reviews = pd.read_csv('data/raw/reviews.csv')
    
    # Map sentiment to numeric
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    reviews['sentiment_label'] = reviews['sentiment'].map(sentiment_map)
    
    print(f"Total reviews: {len(reviews)}")
    print(f"Sentiment distribution:\n{reviews['sentiment'].value_counts()}")
    
    # Initialize classifier
    classifier = SentimentClassifier(num_labels=3)
    
    # Prepare data
    train_loader, val_loader = classifier.prepare_data(
        reviews['review_text'],
        reviews['sentiment_label'],
        batch_size=16
    )
    
    # Train
    classifier.train(epochs=3, learning_rate=2e-5)
    
    # Plot training history
    classifier.plot_training_history()
    
    # Test prediction
    test_texts = [
        "This product is amazing! I love it!",
        "Terrible quality, waste of money.",
        "It's okay, nothing special.",
    ]
    
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    for text in test_texts:
        result = classifier.predict(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})")
    
    # Save model
    classifier.save_model('models/sentiment_bert')
    
    print("\n" + "="*60)
    print("PHASE 2.1 COMPLETE: BERT Sentiment Model Trained!")
    print("="*60)
```

### 2.2 Sales Forecasting with LSTM

```python
# src/models/sales_forecaster.py
"""
Time Series Forecasting with LSTM covering:
- Sequence preparation
- LSTM architecture
- Training and prediction
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TimeSeriesDataset(Dataset):
    """Dataset for time series sequences."""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMForecaster(nn.Module):
    """
    LSTM model for time series forecasting.
    
    Concepts:
    - LSTM layers (hidden state, cell state)
    - Sequence processing
    - Dropout for regularization
    """
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Use last hidden state for prediction
        out = self.fc(out[:, -1, :])
        
        return out


class SalesForecaster:
    """
    Complete sales forecasting pipeline.
    
    Concepts covered:
    - Time series preprocessing
    - LSTM training
    - Multi-step forecasting
    """
    
    def __init__(self, seq_length=30):
        self.seq_length = seq_length
        self.scaler = MinMaxScaler()
        self.model = None
        self.history = {'train_loss': [], 'val_loss': []}
    
    def prepare_data(self, data, target_col='revenue'):
        """
        Prepare time series data for LSTM.
        
        Concepts: Windowing, Normalization
        """
        # Ensure data is sorted by date
        data = data.sort_values('date')
        
        # Get target values
        values = data[target_col].values.reshape(-1, 1)
        
        # Normalize
        values_scaled = self.scaler.fit_transform(values)
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(values_scaled) - self.seq_length):
            seq = values_scaled[i:i+self.seq_length]
            target = values_scaled[i+self.seq_length]
            sequences.append(seq)
            targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        # Split into train/val
        split_idx = int(len(sequences) * 0.8)
        
        X_train, X_val = sequences[:split_idx], sequences[split_idx:]
        y_train, y_val = targets[:split_idx], targets[split_idx:]
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        return self.train_loader, self.val_loader
    
    def build_model(self, hidden_size=64, num_layers=2):
        """Build LSTM model."""
        self.model = LSTMForecaster(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers
        ).to(device)
        
        return self.model
    
    def train(self, epochs=50, learning_rate=0.001):
        """
        Train the LSTM model.
        
        Concepts: Training loop, Loss computation
        """
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for sequences, targets in self.train_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = self.model(sequences)
                loss = criterion(outputs, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for sequences, targets in self.val_loader:
                    sequences = sequences.to(device)
                    targets = targets.to(device)
                    
                    outputs = self.model(sequences)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(self.train_loader)
            avg_val_loss = val_loss / len(self.val_loader)
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")
    
    def forecast(self, last_sequence, steps=30):
        """
        Generate multi-step forecast.
        
        Concepts: Autoregressive forecasting
        """
        self.model.eval()
        
        # Normalize input sequence
        last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1))
        
        predictions = []
        current_seq = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).to(device)
        
        with torch.no_grad():
            for _ in range(steps):
                # Predict next value
                pred = self.model(current_seq)
                predictions.append(pred.item())
                
                # Update sequence (sliding window)
                pred_expanded = pred.unsqueeze(1)
                current_seq = torch.cat([current_seq[:, 1:, :], pred_expanded], dim=1)
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        return predictions.flatten()
    
    def plot_forecast(self, actual, predicted, save_path='data/processed/'):
        """Plot actual vs predicted values."""
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train')
        plt.plot(self.history['val_loss'], label='Validation')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(range(len(actual)), actual, label='Actual', alpha=0.7)
        plt.plot(range(len(actual), len(actual) + len(predicted)), predicted, 
                label='Forecast', color='red')
        plt.title('Sales Forecast')
        plt.xlabel('Time')
        plt.ylabel('Revenue')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_path}lstm_forecast.png', dpi=150)
        plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Load and prepare orders data
    orders = pd.read_csv('data/raw/orders.csv')
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    
    # Aggregate daily revenue
    daily_revenue = orders.groupby(orders['order_date'].dt.date).agg({
        'order_value': 'sum'
    }).reset_index()
    daily_revenue.columns = ['date', 'revenue']
    daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])
    daily_revenue = daily_revenue.sort_values('date')
    
    print(f"Total days: {len(daily_revenue)}")
    print(f"Date range: {daily_revenue['date'].min()} to {daily_revenue['date'].max()}")
    
    # Initialize forecaster
    forecaster = SalesForecaster(seq_length=30)
    
    # Prepare data
    train_loader, val_loader = forecaster.prepare_data(daily_revenue, 'revenue')
    
    # Build model
    model = forecaster.build_model(hidden_size=64, num_layers=2)
    print(model)
    
    # Train
    forecaster.train(epochs=50, learning_rate=0.001)
    
    # Forecast next 30 days
    last_30_days = daily_revenue['revenue'].values[-30:]
    forecast = forecaster.forecast(last_30_days, steps=30)
    
    print(f"\n30-Day Forecast:")
    print(f"Average predicted revenue: ${forecast.mean():,.2f}")
    print(f"Min: ${forecast.min():,.2f}, Max: ${forecast.max():,.2f}")
    
    # Plot
    forecaster.plot_forecast(daily_revenue['revenue'].values, forecast)
    
    # Save model
    torch.save(forecaster.model.state_dict(), 'models/lstm_forecaster.pt')
    
    print("\n" + "="*60)
    print("PHASE 2.2 COMPLETE: LSTM Forecaster Trained!")
    print("="*60)
```

---

*[Continued in next section due to length...]*
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import chromadb
from chromadb.config import Settings


class RAGSystem:
    """
    Production-ready RAG system for customer support.
    
    Concepts covered:
    - Document processing pipeline
    - Chunking strategies
    - Embedding models
    - Vector databases
    - Retrieval strategies
    - Generation with context
    """
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 llm_model: str = "gpt-3.5-turbo",
                 collection_name: str = "customer_support"):
        
        # Initialize embeddings (local model for cost efficiency)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=llm_model,
            temperature=0.3
        )
        
        # Initialize vector store
        self.collection_name = collection_name
        self.vectorstore = None
        
        # Text splitter (chunking strategy)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    def load_documents(self, source_path: str) -> List:
        """
        Load documents from various sources.
        
        Concepts: Document Loaders, Multiple formats
        """
        documents = []
        
        if os.path.isdir(source_path):
            # Load all supported files from directory
            loaders = [
                DirectoryLoader(source_path, glob="**/*.txt", loader_cls=TextLoader),
                DirectoryLoader(source_path, glob="**/*.pdf", loader_cls=PyPDFLoader),
                DirectoryLoader(source_path, glob="**/*.csv", loader_cls=CSVLoader),
            ]
            
            for loader in loaders:
                try:
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading: {e}")
        else:
            # Load single file
            if source_path.endswith('.pdf'):
                loader = PyPDFLoader(source_path)
            elif source_path.endswith('.csv'):
                loader = CSVLoader(source_path)
            else:
                loader = TextLoader(source_path)
            
            documents = loader.load()
        
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def create_knowledge_base(self, documents: List = None, texts: List[str] = None):
        """
        Create vector store from documents.
        
        Concepts: Chunking, Embedding, Indexing
        """
        if documents:
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
        elif texts:
            # Split raw texts
            chunks = self.text_splitter.create_documents(texts)
        else:
            raise ValueError("Either documents or texts must be provided")
        
        print(f"Created {len(chunks)} chunks")
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory="./data/vectorstore"
        )
        
        # Persist to disk
        self.vectorstore.persist()
        
        print("Knowledge base created and persisted")
        return self.vectorstore
    
    def add_faq_data(self, faq_data: List[Dict]):
        """
        Add FAQ data to knowledge base.
        
        Format: [{"question": "...", "answer": "..."}]
        """
        texts = []
        metadatas = []
        
        for faq in faq_data:
            text = f"Question: {faq['question']}\nAnswer: {faq['answer']}"
            texts.append(text)
            metadatas.append({"type": "faq", "category": faq.get("category", "general")})
        
        # Add to vector store
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
        self.vectorstore.persist()
        
        print(f"Added {len(faq_data)} FAQ entries")
    
    def get_retriever(self, search_type: str = "similarity", k: int = 5):
        """
        Get retriever with specified search type.
        
        Concepts: Similarity search, MMR, Threshold
        """
        if search_type == "similarity":
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
        elif search_type == "mmr":
            # Maximum Marginal Relevance (diversity)
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": k * 2}
            )
        elif search_type == "threshold":
            retriever = self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": 0.7, "k": k}
            )
        else:
            raise ValueError(f"Unknown search type: {search_type}")
        
        return retriever
    
    def create_qa_chain(self, chain_type: str = "stuff"):
        """
        Create QA chain.
        
        Concepts: Chain types (stuff, map_reduce, refine)
        """
        # Custom prompt template
        prompt_template = """You are a helpful customer support assistant for an e-commerce company.
Use the following context to answer the customer's question. 
If you don't know the answer based on the context, say so politely and offer to help in another way.

Context:
{context}

Customer Question: {question}

Helpful Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        retriever = self.get_retriever()
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return qa_chain
    
    def create_conversational_chain(self):
        """
        Create conversational chain with memory.
        
        Concepts: Conversation memory, Multi-turn dialogue
        """
        retriever = self.get_retriever()
        
        conv_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=True
        )
        
        return conv_chain
    
    def query(self, question: str, use_conversation: bool = False) -> Dict:
        """
        Query the RAG system.
        
        Returns answer and source documents.
        """
        if use_conversation:
            chain = self.create_conversational_chain()
            result = chain({"question": question})
        else:
            chain = self.create_qa_chain()
            result = chain({"query": question})
        
        return {
            "answer": result.get("result") or result.get("answer"),
            "sources": [doc.page_content[:200] for doc in result.get("source_documents", [])]
        }
    
    def hybrid_search(self, query: str, k: int = 5):
        """
        Perform hybrid search (semantic + keyword).
        
        Concepts: Hybrid retrieval, BM25 + Dense
        """
        # Dense search (semantic)
        dense_results = self.vectorstore.similarity_search(query, k=k)
        
        # For true hybrid, you would add BM25 here
        # This is a simplified version
        
        return dense_results


# =============================================================================
# FAQ DATA FOR KNOWLEDGE BASE
# =============================================================================

FAQ_DATA = [
    {
        "question": "What is your return policy?",
        "answer": "We offer a 30-day return policy for all unused items in original packaging. To initiate a return, log into your account and go to Order History.",
        "category": "returns"
    },
    {
        "question": "How long does shipping take?",
        "answer": "Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days. International shipping takes 10-14 business days.",
        "category": "shipping"
    },
    {
        "question": "How do I track my order?",
        "answer": "You can track your order by logging into your account and visiting the Order History section. You'll also receive tracking updates via email.",
        "category": "orders"
    },
    {
        "question": "What payment methods do you accept?",
        "answer": "We accept Visa, MasterCard, American Express, PayPal, and bank transfers. All transactions are secured with SSL encryption.",
        "category": "payment"
    },
    {
        "question": "How do I cancel my order?",
        "answer": "Orders can be cancelled within 24 hours of placement. Go to Order History, select the order, and click Cancel Order. If the order has shipped, you'll need to return it.",
        "category": "orders"
    },
    {
        "question": "Do you offer international shipping?",
        "answer": "Yes, we ship to over 100 countries. International shipping rates and delivery times vary by location. Check our shipping calculator at checkout.",
        "category": "shipping"
    },
    {
        "question": "How do I reset my password?",
        "answer": "Click 'Forgot Password' on the login page. Enter your email address and we'll send you a reset link. The link expires in 24 hours.",
        "category": "account"
    },
    {
        "question": "What if my item arrives damaged?",
        "answer": "Contact us within 48 hours of delivery with photos of the damage. We'll arrange a replacement or full refund at no extra cost.",
        "category": "returns"
    },
    {
        "question": "How do I contact customer support?",
        "answer": "You can reach us via email at support@company.com, live chat on our website, or call 1-800-XXX-XXXX. Support hours are 9 AM - 6 PM EST.",
        "category": "support"
    },
    {
        "question": "Do you have a loyalty program?",
        "answer": "Yes! Join our VIP program to earn points on every purchase. Points can be redeemed for discounts, free shipping, and exclusive offers.",
        "category": "loyalty"
    }
]


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import os
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Set your API key
    
    # Initialize RAG system
    rag = RAGSystem(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="gpt-3.5-turbo"
    )
    
    # Create knowledge base from FAQ
    texts = [f"Q: {faq['question']}\nA: {faq['answer']}" for faq in FAQ_DATA]
    rag.create_knowledge_base(texts=texts)
    
    # Test queries
    test_questions = [
        "What is your return policy?",
        "How long does shipping take?",
        "My order arrived damaged, what should I do?",
        "Can I pay with Bitcoin?",  # Not in FAQ - should handle gracefully
    ]
    
    print("\n" + "="*60)
    print("RAG SYSTEM TEST")
    print("="*60)
    
    for question in test_questions:
        print(f"\nQ: {question}")
        result = rag.query(question)
        print(f"A: {result['answer']}")
        print(f"Sources: {len(result['sources'])} documents retrieved")
    
    print("\n" + "="*60)
    print("PHASE 3.1 COMPLETE: RAG System Built!")
    print("="*60)
```

### 3.2 AI Agent for Customer Service

```python
# src/agents/customer_agent.py
"""
AI Agent for Customer Service covering:
- Tool definition and usage
- ReAct pattern
- Multi-step reasoning
- Function calling
"""

from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool, StructuredTool
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field
import json
from datetime import datetime, timedelta
import random


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

class OrderLookupInput(BaseModel):
    """Input for order lookup tool."""
    order_id: str = Field(description="The order ID to look up")

class CustomerLookupInput(BaseModel):
    """Input for customer lookup tool."""
    customer_id: str = Field(description="The customer ID to look up")

class RefundInput(BaseModel):
    """Input for refund tool."""
    order_id: str = Field(description="The order ID to refund")
    reason: str = Field(description="Reason for the refund")


# Simulated database functions
def lookup_order(order_id: str) -> str:
    """
    Look up order details from database.
    
    Concepts: Tool function, External API simulation
    """
    # Simulated order data
    orders_db = {
        "ORD-12345": {
            "order_id": "ORD-12345",
            "customer_id": "CUST-789",
            "status": "Shipped",
            "items": ["Wireless Headphones", "Phone Case"],
            "total": 89.99,
            "order_date": "2024-01-10",
            "shipping_date": "2024-01-12",
            "tracking_number": "1Z999AA10123456784",
            "estimated_delivery": "2024-01-17"
        },
        "ORD-67890": {
            "order_id": "ORD-67890",
            "customer_id": "CUST-456",
            "status": "Processing",
            "items": ["Laptop Stand", "USB-C Cable"],
            "total": 45.50,
            "order_date": "2024-01-14",
            "shipping_date": None,
            "tracking_number": None,
            "estimated_delivery": "2024-01-21"
        }
    }
    
    if order_id in orders_db:
        order = orders_db[order_id]
        return json.dumps(order, indent=2)
    else:
        return f"Order {order_id} not found in our system."

def lookup_customer(customer_id: str) -> str:
    """Look up customer details."""
    customers_db = {
        "CUST-789": {
            "customer_id": "CUST-789",
            "name": "John Smith",
            "email": "john.smith@email.com",
            "membership": "Premium",
            "total_orders": 15,
            "account_status": "Active"
        },
        "CUST-456": {
            "customer_id": "CUST-456",
            "name": "Jane Doe",
            "email": "jane.doe@email.com",
            "membership": "Basic",
            "total_orders": 3,
            "account_status": "Active"
        }
    }
    
    if customer_id in customers_db:
        return json.dumps(customers_db[customer_id], indent=2)
    else:
        return f"Customer {customer_id} not found."

def process_refund(order_id: str, reason: str) -> str:
    """Process a refund request."""
    # Simulated refund processing
    refund_id = f"REF-{random.randint(10000, 99999)}"
    return json.dumps({
        "status": "Refund Initiated",
        "refund_id": refund_id,
        "order_id": order_id,
        "reason": reason,
        "estimated_processing": "3-5 business days",
        "message": f"Refund {refund_id} has been initiated for order {order_id}. The amount will be credited to the original payment method within 3-5 business days."
    })

def check_shipping_status(tracking_number: str) -> str:
    """Check shipping status."""
    return json.dumps({
        "tracking_number": tracking_number,
        "carrier": "UPS",
        "status": "In Transit",
        "last_location": "Chicago, IL",
        "last_update": "2024-01-15 14:30:00",
        "estimated_delivery": "2024-01-17"
    })

def search_faq(query: str) -> str:
    """Search FAQ database."""
    # This would connect to our RAG system
    faqs = {
        "return": "Our return policy allows returns within 30 days of purchase.",
        "shipping": "Standard shipping takes 5-7 business days.",
        "payment": "We accept Visa, MasterCard, PayPal, and bank transfers.",
        "cancel": "Orders can be cancelled within 24 hours of placement."
    }
    
    for key, answer in faqs.items():
        if key in query.lower():
            return answer
    
    return "I couldn't find a specific FAQ for that. Let me help you directly."


# =============================================================================
# AGENT DEFINITION
# =============================================================================

class CustomerServiceAgent:
    """
    AI Agent for handling customer service requests.
    
    Concepts covered:
    - Tool-based agents
    - ReAct reasoning pattern
    - Function calling
    - Multi-step planning
    """
    
    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0.1  # Low temperature for consistent responses
        )
        
        self.tools = self._create_tools()
        self.agent = self._create_agent()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def _create_tools(self) -> List[Tool]:
        """
        Create tools for the agent.
        
        Concepts: Tool definition, Descriptions for LLM
        """
        tools = [
            StructuredTool.from_function(
                func=lookup_order,
                name="lookup_order",
                description="Look up order details by order ID. Use this when a customer asks about their order status, tracking, or order details. Input should be the order ID (e.g., ORD-12345).",
                args_schema=OrderLookupInput
            ),
            StructuredTool.from_function(
                func=lookup_customer,
                name="lookup_customer",
                description="Look up customer account details by customer ID. Use this to check customer membership status, order history, or account information.",
                args_schema=CustomerLookupInput
            ),
            StructuredTool.from_function(
                func=process_refund,
                name="process_refund",
                description="Initiate a refund for an order. Use this when a customer requests a refund. Requires order ID and reason for refund.",
                args_schema=RefundInput
            ),
            Tool(
                name="check_shipping",
                func=check_shipping_status,
                description="Check the current shipping status using a tracking number. Input should be the tracking number."
            ),
            Tool(
                name="search_faq",
                func=search_faq,
                description="Search the FAQ database for common questions about returns, shipping, payments, etc. Use this for general policy questions."
            )
        ]
        
        return tools
    
    def _create_agent(self):
        """
        Create the ReAct agent.
        
        Concepts: ReAct pattern, Thought-Action-Observation
        """
        # ReAct prompt template
        react_template = """You are a helpful customer service agent for an e-commerce company.
Your goal is to help customers with their inquiries efficiently and professionally.

You have access to the following tools:
{tools}

Use the following format:

Question: the customer's question or request
Thought: think about what information you need and which tool to use
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now have enough information to respond to the customer
Final Answer: the final helpful response to the customer

Important guidelines:
- Always be polite and professional
- If you need to look up information, use the appropriate tool
- Provide clear, actionable information to customers
- If you can't find information, apologize and offer alternatives
- Never make up information - use the tools to get accurate data

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

        prompt = PromptTemplate(
            template=react_template,
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
        )
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )
        
        return agent_executor
    
    def handle_request(self, customer_message: str) -> str:
        """
        Handle a customer service request.
        
        Returns the agent's response.
        """
        try:
            result = self.agent.invoke({"input": customer_message})
            return result["output"]
        except Exception as e:
            return f"I apologize, but I encountered an issue processing your request. Error: {str(e)}. Please try again or contact us directly at support@company.com."
    
    def handle_conversation(self, messages: List[Dict]) -> str:
        """
        Handle multi-turn conversation.
        
        Concepts: Conversation memory, Context management
        """
        # Build context from previous messages
        context = "\n".join([
            f"{'Customer' if m['role'] == 'user' else 'Agent'}: {m['content']}"
            for m in messages[:-1]
        ])
        
        current_message = messages[-1]['content']
        
        full_input = f"""Previous conversation:
{context}

Current question: {current_message}"""
        
        return self.handle_request(full_input)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import os
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    # Initialize agent
    agent = CustomerServiceAgent(model_name="gpt-3.5-turbo")
    
    # Test scenarios
    test_requests = [
        "What's the status of my order ORD-12345?",
        "I want to return my order ORD-12345 because it doesn't fit.",
        "What is your return policy?",
        "Can you check the shipping for tracking number 1Z999AA10123456784?",
        "I'd like a refund for order ORD-67890, the item arrived damaged.",
    ]
    
    print("\n" + "="*60)
    print("CUSTOMER SERVICE AGENT TEST")
    print("="*60)
    
    for request in test_requests:
        print(f"\n{'='*60}")
        print(f"Customer: {request}")
        print("-"*60)
        response = agent.handle_request(request)
        print(f"\nAgent Response:\n{response}")
    
    print("\n" + "="*60)
    print("PHASE 3.2 COMPLETE: AI Agent Built!")
    print("="*60)
```

### 3.3 Fine-Tuning with LoRA

```python
# src/models/finetune_lora.py
"""
Fine-tuning LLMs with LoRA covering:
- QLoRA (4-bit quantization + LoRA)
- Training pipeline
- Evaluation
"""

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer
import pandas as pd


class CustomerSupportFineTuner:
    """
    Fine-tune an LLM for customer support using LoRA.
    
    Concepts covered:
    - Quantization (4-bit)
    - LoRA configuration
    - Supervised fine-tuning
    - PEFT (Parameter Efficient Fine-Tuning)
    """
    
    def __init__(self, 
                 base_model: str = "mistralai/Mistral-7B-v0.1",
                 output_dir: str = "models/customer_support_lora"):
        
        self.base_model = base_model
        self.output_dir = output_dir
        
        # Quantization config (4-bit for efficiency)
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        # LoRA config
        self.lora_config = LoraConfig(
            r=16,                    # Rank
            lora_alpha=32,           # Scaling factor
            lora_dropout=0.1,        # Dropout
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[         # Which layers to adapt
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ]
        )
        
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """
        Load base model with quantization.
        
        Concepts: Model loading, 4-bit quantization
        """
        print(f"Loading model: {self.base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA
        self.model = get_peft_model(self.model, self.lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        return self.model
    
    def prepare_dataset(self, data: pd.DataFrame) -> Dataset:
        """
        Prepare dataset for fine-tuning.
        
        Expected columns: 'instruction', 'input', 'output'
        """
        def format_prompt(row):
            prompt = f"""### Instruction:
{row['instruction']}

### Input:
{row['input']}

### Response:
{row['output']}"""
            return {"text": prompt}
        
        # Apply formatting
        formatted_data = data.apply(format_prompt, axis=1).tolist()
        
        # Create dataset
        dataset = Dataset.from_list(formatted_data)
        
        return dataset
    
    def train(self, dataset: Dataset, epochs: int = 3):
        """
        Train the model with SFT.
        
        Concepts: Supervised Fine-Tuning, Training Arguments
        """
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            weight_decay=0.01,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,
            optim="paged_adamw_8bit",
            report_to="none"
        )
        
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            dataset_text_field="text",
            tokenizer=self.tokenizer,
            args=training_args,
            max_seq_length=512
        )
        
        # Train
        trainer.train()
        
        # Save
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"Model saved to {self.output_dir}")
    
    def generate(self, prompt: str, max_length: int = 256) -> str:
        """Generate response using fine-tuned model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        return response


# =============================================================================
# TRAINING DATA FOR CUSTOMER SUPPORT
# =============================================================================

TRAINING_DATA = [
    {
        "instruction": "You are a helpful customer service agent. Respond to the customer inquiry.",
        "input": "I haven't received my order yet. It's been 10 days.",
        "output": "I apologize for the delay with your order. Let me look into this for you right away. Could you please provide your order number so I can check the shipping status? In the meantime, standard delivery typically takes 5-7 business days. If it's been longer, there might be a shipping delay we can investigate."
    },
    {
        "instruction": "You are a helpful customer service agent. Respond to the customer inquiry.",
        "input": "How do I return an item?",
        "output": "I'd be happy to help you with a return! Our return process is simple: 1) Log into your account and go to Order History, 2) Select the item you want to return and click 'Initiate Return', 3) Print the prepaid shipping label, and 4) Drop off the package at any authorized shipping location. Returns are accepted within 30 days of delivery for unused items in original packaging. Is there a specific order you'd like to return?"
    },
    {
        "instruction": "You are a helpful customer service agent. Respond to the customer inquiry.",
        "input": "My item arrived damaged. What can I do?",
        "output": "I'm so sorry to hear your item arrived damaged! That's definitely not the experience we want you to have. Please don't worry - we'll make this right. Could you please: 1) Take a few photos of the damage, and 2) Provide your order number? Once I have these details, I can immediately process either a replacement or a full refund, whichever you prefer. The return shipping will be completely free."
    },
    # Add more training examples...
]


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Convert training data to DataFrame
    train_df = pd.DataFrame(TRAINING_DATA)
    
    print(f"Training samples: {len(train_df)}")
    
    # Initialize fine-tuner
    # Note: This requires significant GPU memory
    # finetuner = CustomerSupportFineTuner(
    #     base_model="mistralai/Mistral-7B-v0.1"
    # )
    
    # Load model
    # finetuner.load_model()
    
    # Prepare dataset
    # dataset = finetuner.prepare_dataset(train_df)
    
    # Train
    # finetuner.train(dataset, epochs=3)
    
    print("\n" + "="*60)
    print("PHASE 3.3: LoRA Fine-tuning Setup Complete!")
    print("Note: Actual training requires GPU with sufficient memory")
    print("="*60)
```

---

## Phase 4: Production Deployment

### 4.1 FastAPI Backend

```python
# src/api/main.py
"""
Production API covering:
- REST endpoints
- Request validation
- Streaming responses
- Health checks
- Error handling
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import asyncio
from datetime import datetime
import json

# Import our models (would be loaded from saved files)
# from src.models.churn_model import ChurnPredictor
# from src.models.sentiment_model import SentimentClassifier
# from src.rag.rag_system import RAGSystem
# from src.agents.customer_agent import CustomerServiceAgent


app = FastAPI(
    title="Customer Intelligence Platform API",
    description="AI-powered customer intelligence platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REQUEST/RESPONSE SCHEMAS
# =============================================================================

class ChurnPredictionRequest(BaseModel):
    customer_id: str
    features: Dict[str, float] = Field(
        ..., 
        description="Customer features for prediction"
    )

class ChurnPredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    top_factors: List[Dict[str, float]]

class SentimentRequest(BaseModel):
    text: str
    
class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    use_agent: bool = False

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = None
    conversation_id: str

class ForecastRequest(BaseModel):
    days: int = 30

class ForecastResponse(BaseModel):
    forecast: List[Dict[str, Any]]
    summary: Dict[str, float]


# =============================================================================
# STARTUP/SHUTDOWN EVENTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    print("Loading models...")
    # app.state.churn_model = ChurnPredictor.load("models/xgboost.joblib")
    # app.state.sentiment_model = SentimentClassifier.load("models/sentiment_bert")
    # app.state.rag_system = RAGSystem()
    # app.state.agent = CustomerServiceAgent()
    print("Models loaded!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Shutting down...")


# =============================================================================
# HEALTH CHECK ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/health/models")
async def models_health():
    """Check model loading status."""
    return {
        "churn_model": "loaded",  # Check actual state
        "sentiment_model": "loaded",
        "rag_system": "loaded",
        "agent": "loaded"
    }


# =============================================================================
# CHURN PREDICTION ENDPOINTS
# =============================================================================

@app.post("/predict/churn", response_model=ChurnPredictionResponse)
async def predict_churn(request: ChurnPredictionRequest):
    """
    Predict customer churn probability.
    
    Concepts: ML Model Serving, Feature Input
    """
    try:
        # Simulate prediction (replace with actual model)
        import random
        probability = random.random()
        
        response = ChurnPredictionResponse(
            customer_id=request.customer_id,
            churn_probability=probability,
            churn_prediction=probability > 0.5,
            risk_level="High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low",
            top_factors=[
                {"days_since_last_order": 0.25},
                {"support_tickets": 0.20},
                {"total_spent": 0.15}
            ]
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SENTIMENT ANALYSIS ENDPOINTS
# =============================================================================

@app.post("/analyze/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """
    Analyze sentiment of text.
    
    Concepts: NLP Model Serving
    """
    try:
        # Simulate prediction (replace with actual model)
        import random
        sentiments = ["positive", "neutral", "negative"]
        probs = [random.random() for _ in range(3)]
        total = sum(probs)
        probs = [p/total for p in probs]
        
        max_idx = probs.index(max(probs))
        
        response = SentimentResponse(
            text=request.text,
            sentiment=sentiments[max_idx],
            confidence=probs[max_idx],
            probabilities={s: p for s, p in zip(sentiments, probs)}
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# CHAT/RAG ENDPOINTS
# =============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handle chat requests with RAG or Agent.
    
    Concepts: RAG, Agent Integration
    """
    try:
        import uuid
        
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        if request.use_agent:
            # Use AI agent for complex queries
            response_text = f"[Agent] Processing: {request.message}"
            sources = None
        else:
            # Use RAG for FAQ-type queries
            response_text = f"[RAG] Answer for: {request.message}"
            sources = ["FAQ Document 1", "Policy Document 2"]
        
        return ChatResponse(
            response=response_text,
            sources=sources,
            conversation_id=conversation_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream chat response token by token.
    
    Concepts: Streaming Response, SSE
    """
    async def generate():
        # Simulate streaming response
        response = f"This is a streaming response for: {request.message}"
        for word in response.split():
            yield f"data: {json.dumps({'token': word + ' '})}\n\n"
            await asyncio.sleep(0.1)
        yield f"data: {json.dumps({'done': True})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


# =============================================================================
# FORECAST ENDPOINTS
# =============================================================================

@app.post("/forecast/sales", response_model=ForecastResponse)
async def forecast_sales(request: ForecastRequest):
    """
    Generate sales forecast.
    
    Concepts: Time Series Prediction
    """
    try:
        import random
        
        # Simulate forecast
        forecast = []
        base_value = 10000
        
        for i in range(request.days):
            value = base_value + random.uniform(-1000, 1000)
            forecast.append({
                "day": i + 1,
                "predicted_revenue": round(value, 2),
                "lower_bound": round(value * 0.9, 2),
                "upper_bound": round(value * 1.1, 2)
            })
        
        values = [f["predicted_revenue"] for f in forecast]
        
        return ForecastResponse(
            forecast=forecast,
            summary={
                "mean": round(sum(values) / len(values), 2),
                "min": round(min(values), 2),
                "max": round(max(values), 2),
                "total": round(sum(values), 2)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# BATCH PROCESSING ENDPOINTS
# =============================================================================

@app.post("/batch/sentiment")
async def batch_sentiment(texts: List[str], background_tasks: BackgroundTasks):
    """
    Process multiple texts in background.
    
    Concepts: Background Tasks, Batch Processing
    """
    job_id = str(datetime.utcnow().timestamp())
    
    # Add to background tasks
    background_tasks.add_task(process_batch_sentiment, job_id, texts)
    
    return {
        "job_id": job_id,
        "status": "processing",
        "total_items": len(texts)
    }

async def process_batch_sentiment(job_id: str, texts: List[str]):
    """Background task for batch processing."""
    # Process texts
    results = []
    for text in texts:
        # Simulate processing
        await asyncio.sleep(0.1)
        results.append({"text": text, "sentiment": "positive"})
    
    # Store results (would save to database/cache)
    print(f"Job {job_id} completed with {len(results)} results")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
```

### 4.2 Docker Configuration

```dockerfile
# docker/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://user:pass@db:5432/customer_intel
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ../models:/app/models
      - ../data:/app/data

  db:
    image: postgres:14
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=customer_intel
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  redis_data:
  grafana_data:
```

---

## Concepts Mapping

### Complete Coverage of All Three Guides

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONCEPTS COVERED IN THIS PROJECT                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MACHINE LEARNING GUIDE                                                     │
│  ├── ✅ 5-Step Data Analyst Process (Phase 1.3 - EDA)                      │
│  ├── ✅ Data Assessment & Quality Dimensions (Phase 1.3)                   │
│  ├── ✅ EDA - Univariate/Bivariate/Multivariate (Phase 1.3)               │
│  ├── ✅ Feature Engineering & Selection (Phase 1.4)                        │
│  ├── ✅ All ML Algorithms - Regression & Classification (Phase 1.5)       │
│  ├── ✅ Ensemble Methods - Bagging, Boosting, Stacking (Phase 1.5)        │
│  ├── ✅ Model Evaluation Metrics (Phase 1.5)                               │
│  ├── ✅ Cross-Validation & Hyperparameter Tuning (Phase 1.5)              │
│  └── ✅ Feature Importance & Interpretability (Phase 1.5)                  │
│                                                                             │
│  DEEP LEARNING GUIDE                                                        │
│  ├── ✅ Neural Network Fundamentals (Phase 2.1, 2.2)                       │
│  ├── ✅ Forward/Backpropagation (Phase 2.1, 2.2)                          │
│  ├── ✅ Activation Functions (Phase 2.1, 2.2)                              │
│  ├── ✅ Loss Functions (Phase 2.1, 2.2)                                    │
│  ├── ✅ LSTM for Time Series (Phase 2.2)                                   │
│  ├── ✅ Transformers/BERT (Phase 2.1)                                      │
│  ├── ✅ Transfer Learning (Phase 2.1)                                      │
│  ├── ✅ Fine-tuning (Phase 2.1, 3.3)                                       │
│  └── ✅ Tokenization & Embeddings (Phase 2.1)                              │
│                                                                             │
│  MODERN AI GUIDE                                                            │
│  ├── ✅ LLMs - GPT, Claude Integration (Phase 3.1, 3.2)                   │
│  ├── ✅ RAG - Complete Pipeline (Phase 3.1)                                │
│  │   ├── Document Loading & Chunking                                       │
│  │   ├── Embedding Generation                                              │
│  │   ├── Vector Database (ChromaDB)                                        │
│  │   ├── Retrieval Strategies                                              │
│  │   └── Generation with Context                                           │
│  ├── ✅ AI Agents (Phase 3.2)                                              │
│  │   ├── Tool Definition                                                   │
│  │   ├── ReAct Pattern                                                     │
│  │   ├── Function Calling                                                  │
│  │   └── Multi-step Reasoning                                              │
│  ├── ✅ Fine-tuning with LoRA/QLoRA (Phase 3.3)                           │
│  ├── ✅ Prompt Engineering (Phase 3.1, 3.2)                                │
│  ├── ✅ MLOps (Phase 4)                                                    │
│  │   ├── Experiment Tracking (MLflow)                                      │
│  │   ├── Model Serving (FastAPI)                                           │
│  │   ├── Containerization (Docker)                                         │
│  │   └── Monitoring (Prometheus/Grafana)                                   │
│  └── ✅ Production Deployment (Phase 4)                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## How to Run This Project

### Step 1: Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Generate Data

```bash
python src/data/generate_data.py
```

### Step 3: Run EDA

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Step 4: Train ML Models

```bash
python src/models/churn_model.py
```

### Step 5: Train Deep Learning Models

```bash
python src/models/sentiment_model.py
python src/models/sales_forecaster.py
```

### Step 6: Setup RAG System

```bash
python src/rag/rag_system.py
```

### Step 7: Test AI Agent

```bash
python src/agents/customer_agent.py
```

### Step 8: Run API

```bash
uvicorn src.api.main:app --reload
```

### Step 9: Deploy with Docker

```bash
cd docker
docker-compose up -d
```

---

## Conclusion

This single project covers **ALL concepts** from the three comprehensive guides:

1. **Machine Learning**: From data analysis to ensemble models
2. **Deep Learning**: From neural networks to transformers
3. **Modern AI**: From LLMs to production deployment

By building this Customer Intelligence Platform, you will:
- ✅ Master data preprocessing and EDA
- ✅ Understand all major ML algorithms
- ✅ Learn deep learning with PyTorch/Transformers
- ✅ Build production RAG systems
- ✅ Create AI agents with tools
- ✅ Fine-tune LLMs with LoRA
- ✅ Deploy ML systems to production

**This is your complete hands-on journey from beginner to production-ready AI engineer!**
