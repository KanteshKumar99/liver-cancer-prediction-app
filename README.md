# Liver Cancer Prediction using Machine Learning 🩺📊

This project is a **Machine Learning classification system** that performs EDA (Exploratory Data Analysis) and prediction on a liver cancer dataset using a **Random Forest Classifier**.

## 📌 Project Overview

This script:
- Loads the dataset (CSV/XLS format)
- Checks for missing values and target class distribution
- Visualizes the data (age distribution, correlation heatmap, etc.)
- Encodes categorical features
- Trains a Random Forest model to predict liver cancer
- Evaluates model performance (accuracy, classification report, confusion matrix)

## 🛠️ Tech Stack

- **Python 3**
- **Pandas** – Data manipulation
- **NumPy** – Numerical operations
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-learn** – Machine learning (Random Forest, Label Encoding, Train-Test Split, Metrics)

## 📂 Dataset

Dataset name: `liver_cancer_prediction (1).xls`
(Note: The file is actually in CSV format, saved with an `.xls` extension)

The dataset includes the following columns:
- `Age`
- `Population`
- `Incidence_Rate`
- `Mortality_Rate`
- `Survival_Rate`
- `Cost_of_Treatment`
- `Country`, `Region` (categorical)
- `Prediction` (Target: Yes/No)

## 📊 Exploratory Data Analysis (EDA)

The script generates the following visualizations:

| Plot | Description |
|------|-------------|
| `target_distribution.png` | Count distribution of Prediction (Yes/No) |
| `age_distribution.png` | Histogram of Age vs Prediction |
| `correlation_heatmap.png` | Correlation heatmap of numeric features |
| `confusion_matrix.png` | Confusion matrix of model predictions |

## 🤖 Model

- **Algorithm:** Random Forest Classifier
- **Train-Test Split:** 80% training, 20% testing
- **Random State:** 42 (for reproducibility)
- **Target Encoding:** No → 0, Yes → 1

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone <your-repo-link>
   cd <repo-folder>
   ```

2. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. Place the dataset file (`liver_cancer_prediction (1).xls`) in the project folder.

4. Run the script:
   ```bash
   python main.py
   ```

5. Console output will be printed, and all visualizations will be saved as PNG files in the project folder.

## 📈 Output

The script produces:
- Dataset shape & info
- Missing values summary
- Target class distribution (%)
- Model accuracy score
- Detailed classification report
- 4 saved visualization images

## 📁 Project Structure

```
├── main.py
├── liver_cancer_prediction (1).xls
├── target_distribution.png
├── age_distribution.png
├── correlation_heatmap.png
├── confusion_matrix.png
└── README.md
```

## 👤 Author

**Kantesh Kumar**
Aspiring Cloud Data Engineer | ETL/ELT Pipeline Developer
📍 Karachi, Pakistan

## 📄 License

This project is for educational/personal purposes. Feel free to fork and modify it as needed.
