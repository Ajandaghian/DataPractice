# 🚢 Titanic: Machine Learning from Disaster (part 2)

<div align="center">
<img src="https://raw.githubusercontent.com/Ajandaghian/DataPractice/506e78f670e3ae34d5c888b97bfa210e986311e2/Titanic%20(ML%20Research%20%26%20Prototyping)%20%7C%20Kaggle%20/.static/titanic.png" alt="Titanic" />

*Predicting survival on the Titanic using machine learning*

</div>


### *A Machine Learning Pipeline for Survival Prediction*

---
## 🎯 About This Project

The **Titanic ML Production Structure Pipeline** is a comprehensive machine learning solution that demonstrates end-to-end ML engineering best practices. This project goes beyond simple model training to showcase a **production pipeline** with proper data validation, feature engineering, model persistence, and prediction serving.

### 🔍 **Problem Statement**
Predict passenger survival on the Titanic using historical data while maintaining code quality, reproducibility, and scalability standards suitable for production environments.

### 💡 **Solution Approach**
- **Modular Architecture**: Separated concerns with distinct modules for data management, feature engineering, and model training
- **Configuration-Driven**: YAML-based configuration for easy parameter tuning and environment management
- **Pipeline-First Design**: Scikit-learn pipelines ensure reproducible transformations and predictions
- **Production Standards**: Proper logging, error handling, and validation throughout

### Still To Do

- [ ] 1. 🧪 Testing Framework (pytest)
- [ ] 2. 🐳 Containerization (Docker)
- [ ] 3. 🌐 API Development (FastAPI)
- [ ] 4. 🔄 CI/CD Pipeline (GitHub Actions)
- [ ] 5. 📊 Monitoring & Logging
- [ ] 6. 🚀 Cloud Deployment (AWS/GCP)


### 🎯 **Why This Project Matters**
This project demonstrates real-world ML engineering skills that bridge the gap between data science experimentation and production deployment. It showcases industry best practices for maintaining, scaling, and deploying ML models.

---

## 🚀 Project Goals & Objectives

- 🎯 **High Accuracy**: Achieve 83%+ prediction accuracy on Titanic survival
- 🔧 **Production Structure**: Follow best practices for ML pipeline development
- 📊 **Feature Engineering**: Implement advanced feature extraction and transformation
- 🏗️ **Modular Design**: Create reusable, testable components with clear separation of concerns
- ⚙️ **Configuration Management**: Enable easy parameter tuning through YAML configurations
- 📈 **Scalability**: Design patterns that support future enhancements and deployment

---

## 📁 Repository Structure

```
🚢 Titanic ML Production Pipeline/
├── 📄 README.md                     # Project documentation
├── 📄 environment.yml               # Conda environment specification
├── 📄 submission.csv                # Kaggle competition submission
├── 📂 src/                          # Main source code
│   ├── 🐍 __init__.py
│   ├── 🔧 pipeline.py               # ML pipeline definitions
│   ├── 🎯 predict.py                # Prediction interface
│   ├── 🏋️ train_pipeline.py         # Training orchestration
│   ├── 📂 config/                   # Configuration management
│   │   ├── 🐍 __init__.py
│   │   ├── ⚙️ configuration.yml     # Project parameters
│   │   └── 🔧 core.py               # Config loading utilities
│   ├── 📂 data_manager/             # Data handling & validation
│   │   ├── 🐍 __init__.py
│   │   ├── 📥 data_loader.py        # Data I/O operations
│   │   ├── ✅ data_validator.py     # Data quality checks
│   │   └── 📂 datasets/             # Raw and processed data
│   │       ├── 📂 raw/              # Original datasets
│   │       └── 📂 processed/        # Transformed datasets
│   ├── 📂 features/                 # Feature engineering
│   │   ├── 🐍 __init__.py
│   │   └── 🛠️ feature_engineering.py # Custom transformers
│   ├── 📂 model/                    # Trained models & artifacts
│   │   ├── 🐍 __init__.py
│   │   ├── 🔧 feature_engineering.pkl # Fitted feature pipeline
│   │   └── 🤖 final_model.pkl       # Trained ML model
│   └── 📂 script/                   # Utility scripts
│       └── 📤 submission_script.py  # Kaggle submission generator
└── 📂 test/                         # Test suite (expandable)
```

### 🗂️ **Architecture Highlights**
- **`src/config/`**: Centralized configuration management with YAML
- **`src/data_manager/`**: Robust data handling with validation
- **`src/features/`**: Custom sklearn transformers for domain-specific features
- **`src/model/`**: Serialized pipeline artifacts for deployment
- **Separation of Concerns**: Each module has a single, well-defined responsibility

---

## 🔬 Technical Deep Dive

### **Phase 1: Data Management & Validation** 🗃️
```python
# Robust data loading with built-in validation
raw_data = load_data(path=config['path']['train'])
validate_data(raw_data)  # Schema and quality checks
```

**Key Components:**
- **Data Loader**: Handles CSV reading with error management
- **Data Validator**: Ensures data quality and schema compliance
- **Path Management**: Configuration-driven file path handling

### **Phase 2: Advanced Feature Engineering** 🛠️
```python
# Custom transformers for domain-specific features
feature_pipeline = Pipeline([
    ('title_extractor', TitleExtractor()),          # Extract titles from names
    ('age_group_encoder', AgeGroupEncoder()),       # Age categorization
    ('family_features', IsFamilyOnBoard()),         # Family size indicators
    ('fare_outlier_capping', CapFareOutliers()),    # Outlier handling
    ('ticket_counter', TicketCounter()),            # Ticket pattern analysis
])
```

**Feature Engineering Highlights:**
- **Title Extraction**: Extract social titles from passenger names (Mr, Mrs, Miss, etc.)
- **Age Grouping**: Convert continuous age to meaningful categories
- **Family Detection**: Engineer family size and traveling alone indicators
- **Fare Outlier Handling**: Class-based outlier capping using IQR method
- **Ticket Analysis**: Extract patterns from ticket numbers

### **Phase 3: Model Training & Persistence** 🤖
```python
# Gradient Boosting with optimized hyperparameters
model_pipeline = Pipeline([
    ('model', GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=20250708
    ))
])
```

**Technical Decisions:**
- **Gradient Boosting**: Chosen for its robustness to feature scales and handling of mixed data types
- **Pipeline Architecture**: Ensures consistent preprocessing in training and inference
- **Model Persistence**: Joblib serialization for deployment readiness
- **Configuration-Driven**: All hyperparameters externalized to YAML

### **Phase 4: Prediction Interface** 🎯
```python
# Production-ready prediction interface
def make_predictions(data: pd.DataFrame | dict) -> np.ndarray:
    validate_data(data)
    X_transformed = feature_pipeline.transform(data)
    return model_pipeline.predict(X_transformed)
```

**Production Features:**
- **Input Flexibility**: Accepts both DataFrames and dictionaries
- **Automatic Validation**: Built-in data quality checks
- **Error Handling**: Graceful failure with meaningful error messages
- **Type Safety**: Modern Python type hints for better code quality

---

## 📊 Key Results & Insights

### **🎯 Model Performance**
| Metric | Value | Description |
|--------|-------|-------------|
| **Training Accuracy** | 80%+ | Performance on training dataset |
| **Cross-Validation** | 80%+ | 5-fold CV average accuracy |
| **Feature Count** | 12 | Engineered features after selection |
| **Training Time** | <30s | End-to-end pipeline execution |

---

## 🛠️ Technologies Used

<div align="center">

![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)  ![Feature-Engine](https://img.shields.io/badge/Feature--Engine-FF6B6B?style=for-the-badge) ![YAML](https://img.shields.io/badge/YAML-CB171E?style=for-the-badge&logo=yaml&logoColor=white)
</div>

---

## 🚀 Quick Start Guide

### **Prerequisites** 📋
- **Python 3.12+**
- **Conda** (recommended) or pip
- **Git** for cloning

### **Installation Steps** ⚡

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/titanic-ml-pipeline.git
   cd titanic-ml-pipeline
   ```

2. **Create Environment**
   ```bash
   # Using Conda (Recommended)
   conda env create -f environment.yml
   conda activate data_science_proj

   # Or using pip
   pip install -r requirements.txt
   ```


---

## 🔮 Next Steps & Future Work

### **🚀 Production Enhancements**
- [ ] **API Development**: Flask/FastAPI REST interface
- [ ] **Docker Containerization**: Deployment-ready containers
- [ ] **Model Monitoring**: Performance tracking and drift detection
- [ ] **A/B Testing Framework**: Model comparison infrastructure

### **📊 Advanced Analytics**
- [ ] **SHAP Integration**: Model explainability and interpretability
- [ ] **Feature Store**: Centralized feature management
- [ ] **MLOps Pipeline**: CI/CD for model deployment
- [ ] **Real-time Predictions**: Streaming prediction capabilities

### **🔧 Technical Debt**
- [ ] **Comprehensive Testing**: Unit and integration test suite
- [ ] **Documentation**: API documentation with Sphinx
- [ ] **Code Quality**: Pre-commit hooks and linting
- [ ] **Performance Optimization**: Memory and speed improvements

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### **🌟 Ways to Contribute**
- 🐛 **Bug Reports**: Found an issue? Let me know!
- 💡 **Feature Requests**: Have ideas for improvements? share!
- 🔧 **Code Contributions**: Submit pull requests

### **🛠️ Development Setup**
1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** with proper tests
4. **Submit a pull request** with clear description

---

## 📧 Contact

**👓 Am. Janian**
*A Curious Product Manager 👓 Exploring the Data Science World*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/amirh-jandaghian/) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ajandaghian) [![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:amirh.jandaghian@gmail.com) [![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/amirhjandaghian)


<!--
---

<div align="center">

### **⭐ If this project helped you, please give it a star!**

*Built with ❤️ and lots of ☕ by the ML community*

**[⬆ Back to Top](#-titanic-ml-production-pipeline)**

</div>

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. -->

-- *Happy Coding! 🚀* --