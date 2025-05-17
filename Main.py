# netflix_data_project.py

# ========== STEP 1: Import Libraries ==========
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import logging

# ========== STEP 2: Set Up Logging ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ========== STEP 3: Load Dataset ==========
def load_dataset(csv_path='netflix_titles.csv'):
    logging.info("Loading Netflix dataset...")
    df = pd.read_csv(csv_path)
    logging.info(f"Data loaded with shape: {df.shape}")
    return df

# ========== STEP 4: Clean Data ==========
def clean_data(df):
    logging.info("Cleaning data...")
    df.drop_duplicates(inplace=True)
    df['date_added'] = pd.to_datetime(df['date_added'])
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month
    df['duration'] = df['duration'].fillna('0 Unknown')

    df[['duration_int', 'duration_type']] = df['duration'].str.extract(r'(\d+)\s*(\w+)')
    df['duration_int'] = pd.to_numeric(df['duration_int'], errors='coerce').fillna(0).astype(int)
    df['duration_type'] = df['duration_type'].fillna('Unknown')
    df.fillna({'country': 'Unknown', 'rating': 'Unknown', 'director': 'Unknown'}, inplace=True)
    logging.info("Cleaning complete.")
    return df

# ========== STEP 5: Store in SQL ==========
def store_to_sql(df, db_name='netflix_data.db'):
    logging.info("Storing data to SQLite database...")
    conn = sqlite3.connect(db_name)
    df.to_sql('netflix', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    logging.info("Data stored successfully.")

# ========== STEP 6: Load from SQL ==========
def load_from_sql(db_name='netflix_data.db'):
    logging.info("Loading data from SQL database...")
    conn = sqlite3.connect(db_name)
    df = pd.read_sql('SELECT * FROM netflix', conn)
    conn.close()
    logging.info("Data loaded successfully.")
    return df

# ========== STEP 7: Exploratory Data Analysis ==========
def plot_eda(df):
    logging.info("Generating visualizations...")

    plt.figure(figsize=(10,6))
    sns.countplot(data=df, x='type')
    plt.title('Content Type Distribution')
    plt.show()

    top_countries = df['country'].value_counts().head(10)
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_countries.values, y=top_countries.index)
    plt.title('Top 10 Countries by Number of Titles')
    plt.xlabel('Number of Titles')
    plt.show()

    plt.figure(figsize=(10,6))
    sns.histplot(data=df, x='year_added', bins=20, kde=False)
    plt.title('Titles Added Per Year')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()

# ========== STEP 8: Basic Machine Learning: Popular Genre Predictor ==========
def genre_prediction_model(df):
    logging.info("Running genre popularity model (simple ML)...")

    df_model = df[['type', 'rating', 'duration_type', 'duration_int']].copy()
    df_model.dropna(inplace=True)
    le = LabelEncoder()
    for col in ['type', 'rating', 'duration_type']:
        df_model[col] = le.fit_transform(df_model[col])

    df_model['label'] = np.where(df_model['duration_int'] >= 60, 1, 0)  # Popular if duration >= 60

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    X = df_model.drop('label', axis=1)
    y = df_model['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Model Accuracy: {acc:.2f}")
    logging.info("Classification Report:\n" + classification_report(y_test, y_pred))

# ========== STEP 9: Export to Excel ==========
def export_to_excel(df, filename='Netflix_Cleaned_Report.xlsx'):
    logging.info("Exporting data to Excel...")
    df.to_excel(filename, index=False)
    logging.info("Data exported successfully.")

# ========== STEP 10: Full Pipeline Execution ==========
def run_pipeline():
    df = load_dataset()
    df = clean_data(df)
    store_to_sql(df)
    df_sql = load_from_sql()
    plot_eda(df_sql)
    genre_prediction_model(df_sql)
    export_to_excel(df_sql)

if __name__ == '__main__':
    run_pipeline()
