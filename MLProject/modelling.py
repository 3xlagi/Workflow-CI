import os
import shutil
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

with mlflow.start_run():
    print("Memuat data...")
    df = pd.read_csv('Data_Cleaned.csv')
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Melatih model...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    acc = rf.score(X_test, y_test)
    print(f"Akurasi: {acc}")
    
    # Simpan ke MLflow tracking 
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(rf, "model")
    
    workspace_path = os.getenv('GITHUB_WORKSPACE', '.')
    deploy_path = os.path.join(workspace_path, 'deploy_model')
    

    if os.path.exists(deploy_path):
        shutil.rmtree(deploy_path)
        

    mlflow.sklearn.save_model(rf, deploy_path)
    print(f"Berhasil {deploy_path}")