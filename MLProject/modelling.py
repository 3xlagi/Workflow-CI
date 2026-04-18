import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

with mlflow.start_run():
    print("Memuat data...")
    # Path relatif ke luar folder MLProject
    df = pd.read_csv('../Data_Cleaned.csv')
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Melatih model...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    acc = rf.score(X_test, y_test)
    print(f"Akurasi: {acc}")
    
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(rf, "model")
    print("Model berhasil disimpan ke MLflow!")