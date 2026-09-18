import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Style settings for plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def save_plot(fig, filename):
    """Save a plot to disk and close it to avoid blocking the script."""
    fig.tight_layout()
    fig.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close(fig)


def main():
    # 1. Load the dataset
    file_name = 'liver_cancer_prediction.csv.gz'    
    with open(file_name, 'rb') as f:
        header = f.read(200)

    # The dataset is actually CSV content saved with an .xls extension.
    if b',' in header:
        df = pd.read_csv(file_name)
    else:
        df = pd.read_csv(file_name)
    print(f"Dataset Loaded Successfully! Shape: {df.shape}\n")

    # 2. Exploratory Data Analysis (EDA) summary
    print("--- Dataset Info ---")
    print(df.info())

    print("\n--- Missing Values Check ---")
    print(df.isnull().sum())

    print("\n--- Target Distribution (Prediction) ---")
    print(df['Prediction'].value_counts(normalize=True) * 100)

    # 3. Visualization section: save all plots to image files
    # Plot 1: Target variable count
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Prediction', data=df, hue='Prediction', palette='Set2', dodge=False, ax=ax1, legend=False)
    ax1.set_title('Liver Cancer Prediction Distribution')
    ax1.set_xlabel('Prediction (Yes / No)')
    ax1.set_ylabel('Count')
    save_plot(fig1, 'target_distribution.png')

    # Plot 2: Age distribution by prediction
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df, x='Age', hue='Prediction', kde=True, bins=30, palette='viridis', alpha=0.6, ax=ax2)
    ax2.set_title('Age Distribution based on Liver Cancer Prediction')
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Frequency')
    save_plot(fig2, 'age_distribution.png')

    # Plot 3: Correlation matrix for numeric features
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    num_cols = ['Age', 'Population', 'Incidence_Rate', 'Mortality_Rate', 'Survival_Rate', 'Cost_of_Treatment']
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax3)
    ax3.set_title('Correlation Heatmap of Numerical Features')
    save_plot(fig3, 'correlation_heatmap.png')

    # 4. Data preprocessing for machine learning
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    if 'Prediction' in cat_cols:
        cat_cols.remove('Prediction')

    df_encoded = df.copy()
    le = LabelEncoder()
    for col in cat_cols:
        df_encoded[col] = le.fit_transform(df_encoded[col])

    # Encode target variable: 'No' -> 0, 'Yes' -> 1
    df_encoded['Prediction'] = df_encoded['Prediction'].map({'No': 0, 'Yes': 1})

    # Define features (X) and target (y)
    X = df_encoded.drop(columns=['Prediction', 'Country', 'Region'])
    y = df_encoded['Prediction']

    # 5. Train-test split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Train the Random Forest model
    print("\nTraining the Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 7. Evaluate the model
    y_pred = model.predict(X_test)
    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot 4: Confusion matrix
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax4)
    ax4.set_title('Confusion Matrix')
    ax4.set_xlabel('Predicted Label')
    ax4.set_ylabel('True Label')
    save_plot(fig4, 'confusion_matrix.png')

    # Final status message
    print("\nAll plots have been saved in the project folder.")
    print("Generated files: target_distribution.png, age_distribution.png, correlation_heatmap.png, confusion_matrix.png")


if __name__ == "__main__":
    main()
