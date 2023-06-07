import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(training_file, testing_file):
    try:
        training_data = pd.read_csv(training_file)
        testing_data = pd.read_csv(testing_file)
    except ValueError as e:
        print(f"ValueError: {e}")
        return None, None
    # Convert DataFrame to floats, invalid entries (non-convertible strings) will be replaced with NaN
    training_data = training_data.apply(pd.to_numeric, errors='coerce')
    testing_data = testing_data.apply(pd.to_numeric, errors='coerce')
    # Optionally, remove or impute rows with NaN values
    training_data = training_data.dropna()
    testing_data = testing_data.dropna()
    return training_data, testing_data

def extract_features(training_data):
    X = training_data.iloc[:, :-1]
    y = training_data.iloc[:, -1]
    return X, y

def split_data(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

def normalize_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test

def balance_data(X_train, y_train):
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    return X_train_res, y_train_res

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, y_train)
    return model

def make_predictions(X_test, model):
    predicted_labels = model.predict(X_test)
    return predicted_labels

def save_predictions(testing_data, predicted_labels, output_file):
    testing_results = pd.DataFrame(testing_data)
    testing_results['label'] = predicted_labels
    testing_results.to_csv(output_file, index=False)
    return testing_results

def evaluate_model(y_val, y_pred):
    f1 = f1_score(y_val, y_pred, average='macro')
    print("F1 Score:", f1)
    print("Classification Report:")
    print(classification_report(y_val, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # Feature Importance
    if isinstance(model, RandomForestClassifier):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
        plt.xticks(range(X_train.shape[1]), indices)
        plt.xlim([-1, X_train.shape[1]])
        plt.show()

if __name__ == "__main__":
    training_data, testing_data = load_data("TrainingDataMulti.csv", "TestingDataMulti.csv")
    X, y = extract_features(training_data)
    X_train, X_val, y_train, y_val = split_data(X, y)
    X_train, X_val, testing_data = normalize_data(X_train, X_val, testing_data)
    X_train, y_train = balance_data(X_train, y_train)
    model = train_model(X_train, y_train)
    y_pred = make_predictions(X_val, model)
    evaluate_model(y_val, y_pred)
    predicted_labels = make_predictions(testing_data, model)
    save_predictions(testing_data, predicted_labels, "TestingResultsMulti.csv")
