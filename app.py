import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import optuna
import shap
import joblib
import base64
from io import BytesIO
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import time

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)

# Set your Google Gemini API key
GEMINI_API_KEY = ""

@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None
    return df

def preprocess_data(df):
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in df_imputed.select_dtypes(include=['object']):
        df_imputed[col] = le.fit_transform(df_imputed[col])
    
    # Scale numerical features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns)
    
    return df_scaled

def detect_task_type(y):
    unique_values = np.unique(y)
    if len(unique_values) > 10 or np.issubdtype(y.dtype, np.number):
        return 'regression'
    else:
        return 'classification'

def select_features(X, y, task_type, k=10):
    if task_type == 'classification':
        mi_scores = mutual_info_classif(X, y)
    else:
        mi_scores = mutual_info_regression(X, y)
    
    mi_scores = pd.Series(mi_scores, index=X.columns)
    top_features = mi_scores.nlargest(k).index.tolist()
    return X[top_features], mi_scores

def split_data(df, target_column, test_size=0.2):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=42)

def objective(trial, X, y, task_type):
    if task_type == 'classification':
        model = XGBClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 1000),
            max_depth=trial.suggest_int('max_depth', 1, 10),
            learning_rate=trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
            subsample=trial.suggest_uniform('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_uniform('colsample_bytree', 0.5, 1.0)
        )
    else:
        model = XGBRegressor(
            n_estimators=trial.suggest_int('n_estimators', 100, 1000),
            max_depth=trial.suggest_int('max_depth', 1, 10),
            learning_rate=trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
            subsample=trial.suggest_uniform('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_uniform('colsample_bytree', 0.5, 1.0)
        )
    
    score = cross_val_score(model, X, y, cv=5, scoring='accuracy' if task_type == 'classification' else 'neg_mean_squared_error')
    return score.mean()

def train_and_evaluate_models(X_train, X_test, y_train, y_test, task_type):
    if task_type == 'classification':
        models = {
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "SVM": SVC(),
            "XGBoost": XGBClassifier(),
            "Neural Network": MLPClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }
    else:
        models = {
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "SVM": SVR(),
            "XGBoost": XGBRegressor(),
            "Neural Network": MLPRegressor(),
            "Gradient Boosting": GradientBoostingRegressor()
        }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if task_type == 'classification':
            results[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted'),
                "recall": recall_score(y_test, y_pred, average='weighted'),
                "f1": f1_score(y_test, y_pred, average='weighted')
            }
        else:
            results[name] = {
                "mse": mean_squared_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred)
            }
        
        # Add cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2' if task_type == 'regression' else 'accuracy')
        results[name]["cv_score"] = cv_scores.mean()
    
    return results, models

def visualize_results(results, task_type):
    if task_type == 'classification':
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'cv_score']
    else:
        metrics = ['mse', 'r2', 'cv_score']
    
    df_results = pd.DataFrame(results).T[metrics]
    
    fig = px.imshow(df_results, 
                    labels=dict(x="Metric", y="Model", color="Score"),
                    x=metrics,
                    y=df_results.index,
                    color_continuous_scale="YlGnBu")
    fig.update_layout(title="Model Performance Comparison")
    st.plotly_chart(fig)

def visualize_feature_importance(mi_scores):
    fig = px.bar(x=mi_scores.index, y=mi_scores.values, 
                 labels={'x': 'Feature', 'y': 'Importance Score'},
                 title="Feature Importance")
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig)

def get_gemini_recommendations(df, task_type):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "contents": [{
            "parts": [{
                "text": f"Given this dataset: {df.head().to_json()}\n\nTask type: {task_type}\n\nProvide recommendations for:\n1. Feature engineering\n2. Model selection\n3. Hyperparameter tuning\n4. Performance optimization\n5. Advanced analytics techniques"
            }]
        }]
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return f"Error: {response.status_code} - {response.text}"

def perform_pca(X):
    pca = PCA()
    pca_result = pca.fit_transform(X)
    explained_variance_ratio = pca.explained_variance_ratio_
    
    fig = px.line(x=range(1, len(explained_variance_ratio) + 1), y=np.cumsum(explained_variance_ratio),
                  labels={"x": "Number of Components", "y": "Cumulative Explained Variance"},
                  title="PCA: Cumulative Explained Variance")
    st.plotly_chart(fig)
    
    return pca_result

def detect_anomalies(X):
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = iso_forest.fit_predict(X)
    return anomalies

def detect_data_drift(X_train, X_test):
    drift_scores = {}
    for col in X_train.columns:
        train_mean = X_train[col].mean()
        test_mean = X_test[col].mean()
        drift_scores[col] = abs(train_mean - test_mean) / train_mean
    
    fig = px.bar(x=list(drift_scores.keys()), y=list(drift_scores.values()),
                 labels={"x": "Feature", "y": "Drift Score"},
                 title="Data Drift Detection")
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig)

def export_model(model):
    buffer = BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="model.joblib">Download Model</a>'

def perform_clustering(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=clusters,
                     labels={"x": "PCA 1", "y": "PCA 2"},
                     title=f'KMeans Clustering (n_clusters={n_clusters})')
    st.plotly_chart(fig)
    
    return clusters

def perform_text_analysis(text_column):
    # Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    sentiments = text_column.apply(lambda x: sia.polarity_scores(x)['compound'])
    
    fig = px.histogram(sentiments, nbins=50, 
                       labels={"value": "Sentiment Score", "count": "Frequency"},
                       title="Sentiment Distribution")
    st.plotly_chart(fig)
    
    # Word Cloud
    text = ' '.join(text_column)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
    
    # Text Clustering
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_column)
    
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(tfidf_matrix)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(tsne_result)
    
    fig = px.scatter(x=tsne_result[:, 0], y=tsne_result[:, 1], color=clusters,
                     labels={"x": "t-SNE 1", "y": "t-SNE 2"},
                     title="Text Clustering")
    st.plotly_chart(fig)

def show_help():
    st.markdown("""
    # Advanced AutoML Pipeline Builder Help

    This application is a comprehensive AutoML (Automated Machine Learning) pipeline builder that helps you analyze your data, select features, train models, and provide insights.

    ## Use Cases:
    1. **Data Exploration**: Understand your dataset's characteristics and distributions.
    2. **Feature Selection**: Identify the most important features for your target variable.
    3. **Model Training and Evaluation**: Automatically train and compare multiple machine learning models.
    4. **Hyperparameter Tuning**: Optimize model parameters for better performance.
    5. **Anomaly Detection**: Identify outliers in your dataset.
    6. **Data Drift Detection**: Detect changes in your data distribution between training and test sets.
    7. **Text Analysis**: Perform sentiment analysis, generate word clouds, and text clustering for text data.
    8. **AI-Powered Recommendations**: Get suggestions for feature engineering and model optimization.

    ## How to Use:
    1. Upload your CSV or Excel file.
    2. Select the target column for prediction.
    3. Click "Run AutoML Pipeline" to start the analysis.
    4. Explore the interactive visualizations, results, and recommendations provided by the application.

    ## Features:
    - Automated task type detection (classification or regression)
    - Interactive feature importance visualization
    - Principal Component Analysis (PCA) with interactive plot
    - KMeans clustering with interactive scatter plot
    - SHAP (SHapley Additive exPlanations) for model interpretability
    - Hyperparameter tuning using Optuna
    - Text analysis capabilities (sentiment analysis, word cloud, and clustering)
    - Integration with Google's Gemini API for AI-powered recommendations
    - Interactive model performance comparison
    - Data drift detection with interactive bar plot
    - Model export functionality

    For more information or support, please contact our team.
    """)

def main():
    st.title("AutoML Pipeline Builder")
    
    # Add help button
    if st.button("Help"):
        show_help()
    
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.write("Dataset Preview:")
            st.write(df.head())
            
            st.write("Data Exploration:")
            st.write(df.describe())
            
            # Interactive data exploration
            st.subheader("Interactive Data Exploration")
            selected_columns = st.multiselect("Select columns for exploration", df.columns)
            if selected_columns:
                fig = px.scatter_matrix(df[selected_columns])
                st.plotly_chart(fig)
            
            target_column = st.selectbox("Select the target column", df.columns)
            
            if st.button("Run AutoML Pipeline"):
                with st.spinner("Running AutoML Pipeline..."):
                    df_processed = preprocess_data(df)
                    X_train, X_test, y_train, y_test = split_data(df_processed, target_column)
                    
                    task_type = detect_task_type(y_train)
                    st.write(f"Detected task type: {task_type}")
                    
                    X_train_selected, mi_scores = select_features(X_train, y_train, task_type)
                    X_test_selected = X_test[X_train_selected.columns]
                    
                    st.subheader("Feature Importance:")
                    visualize_feature_importance(mi_scores)
                    
                    st.subheader("Performing PCA:")
                    pca_result = perform_pca(X_train_selected)
                    
                    st.subheader("Detecting Anomalies:")
                    anomalies = detect_anomalies(X_train_selected)
                    st.write(f"Number of anomalies detected: {sum(anomalies == -1)}")
                    
                    st.subheader("Detecting Data Drift:")
                    detect_data_drift(X_train_selected, X_test_selected)
                    
                    st.subheader("Performing Clustering:")
                    n_clusters = st.slider("Select number of clusters", 2, 10, 3)
                    clusters = perform_clustering(X_train_selected, n_clusters)
                    
                    st.subheader("Training and Evaluating Models:")
                    results, models = train_and_evaluate_models(X_train_selected, X_test_selected, y_train, y_test, task_type)
                    visualize_results(results, task_type)
                    
                    best_model_name = max(results, key=lambda x: results[x]['cv_score'])
                    best_model = models[best_model_name]
                    st.write(f"Best performing model: {best_model_name}")
                    
                    st.subheader("Hyperparameter Tuning:")
                    study = optuna.create_study(direction='maximize')
                    study.optimize(lambda trial: objective(trial, X_train_selected, y_train, task_type), n_trials=100)
                    st.write("Best hyperparameters:", study.best_params)
                    
                    st.subheader("SHAP Values for Model Interpretability:")
                    try:
                        if isinstance(best_model, (RandomForestClassifier, RandomForestRegressor, XGBClassifier, XGBRegressor)):
                            explainer = shap.TreeExplainer(best_model)
                        else:
                            explainer = shap.KernelExplainer(best_model.predict, X_test_selected)
                        
                        shap_values = explainer.shap_values(X_test_selected)
                        fig, ax = plt.subplots()
                        shap.summary_plot(shap_values, X_test_selected, plot_type="bar", show=False)
                        st.pyplot(fig)
                    except Exception as e:
                        st.write(f"Error calculating SHAP values: {str(e)}")
                    
                    st.subheader("Gemini API Recommendations:")
                    recommendations = get_gemini_recommendations(df, task_type)
                    st.write(recommendations)
                    
                    st.subheader("Export Model:")
                    st.markdown(export_model(best_model), unsafe_allow_html=True)
                    
                    # Text Analysis
                    text_columns = df.select_dtypes(include=['object']).columns
                    if len(text_columns) > 0:
                        st.subheader("Text Analysis:")
                        selected_text_column = st.selectbox("Select a text column for analysis", text_columns)
                        perform_text_analysis(df[selected_text_column])
                    
                    # Confusion Matrix for Classification
                    if task_type == 'classification':
                        st.subheader("Confusion Matrix:")
                        y_pred = best_model.predict(X_test_selected)
                        cm = confusion_matrix(y_test, y_pred)
                        fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                                        x=np.unique(y_test), y=np.unique(y_test),
                                        title="Confusion Matrix")
                        st.plotly_chart(fig)
                    
                    # Feature Interaction
                    st.subheader("Feature Interaction:")
                    if len(X_train_selected.columns) >= 2:
                        feature1 = st.selectbox("Select first feature", X_train_selected.columns)
                        feature2 = st.selectbox("Select second feature", X_train_selected.columns)
                        fig = px.scatter(X_train_selected, x=feature1, y=feature2, color=y_train,
                                         title=f"Interaction between {feature1} and {feature2}")
                        st.plotly_chart(fig)
                    
                    # Model Comparison
                    st.subheader("Model Comparison:")
                    model_names = list(results.keys())
                    selected_models = st.multiselect("Select models to compare", model_names, default=model_names[:3])
                    if selected_models:
                        compare_data = {model: results[model] for model in selected_models}
                        fig = go.Figure()
                        for model, scores in compare_data.items():
                            fig.add_trace(go.Bar(name=model, x=list(scores.keys()), y=list(scores.values())))
                        fig.update_layout(title="Model Comparison", barmode='group')
                        st.plotly_chart(fig)
                
                st.success("AutoML Pipeline completed successfully!")

if __name__ == "__main__":
    main()
