import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load data from CSV files
def load_data(training_file, testing_file):
    try:
        # Read the training and testing data from CSV files
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

# Function to extract features and target variable from training data
def extract_features(training_data):
    X = training_data.iloc[:, :-1]  # Extract all columns except the last one as features
    y = training_data.iloc[:, -1]   # Extract the last column as the target variable
    return X, y

# Function to split data into training and validation sets
def split_data(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

# Function to normalize the data
def normalize_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Normalize training data
    X_val = scaler.transform(X_val)          # Normalize validation data
    X_test = scaler.transform(X_test)        # Normalize testing data
    return X_train, X_val, X_test

from imblearn.over_sampling import SMOTE

# Function to balance the data using SMOTE
def balance_data(X_train, y_train):
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    return X_train_res, y_train_res

from sklearn.ensemble import RandomForestClassifier

# Function to train the random forest classifier model
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, y_train)
    return model

# Function to make predictions using the trained model
def make_predictions(X_test, model):
    predicted_labels = model.predict(X_test)
    return predicted_labels

# Function to save the predictions to a CSV file
def save_predictions(testing_data, predicted_labels, output_file):
    testing_results = pd.DataFrame(testing_data)
    testing_results['label'] = predicted_labels
    testing_results.to_csv(output_file, index=False)
    return testing_results
def evaluate_model(y_val, y_pred):
    # Calculate the F1 score and print it
    f1 = f1_score(y_val, y_pred, average='macro')
    print("F1 Score:", f1)

    # Print the classification report
    print("Classification Report:")
    print(classification_report(y_val, y_pred))

    # Generate and display the confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # Feature Importance
    if isinstance(model, RandomForestClassifier):
        # Retrieve feature importances from the model
        importances = model.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Plot the feature importances
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
        plt.xticks(range(X_train.shape[1]), indices)
        plt.xlim([-1, X_train.shape[1]])
        plt.show()

if __name__ == "__main__":
    # Load training and testing data
    training_data, testing_data = load_data("TrainingDataMulti.csv", "TestingDataMulti.csv")

    # Extract features from the training data
    X, y = extract_features(training_data)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = split_data(X, y)

    # Normalize the data
    X_train, X_val, testing_data = normalize_data(X_train, X_val, testing_data)

    # Balance the training data
    X_train, y_train = balance_data(X_train, y_train)

    # Train the model
    model = train_model(X_train, y_train)

    # Make predictions on the validation set
    y_pred = make_predictions(X_val, model)

    # Evaluate the model on the validation set
    evaluate_model(y_val, y_pred)

    # Make predictions on the testing data
    predicted_labels = make_predictions(testing_data, model)

    # Save the predicted labels to a file
    save_predictions(testing_data, predicted_labels, "TestingResultsMulti.csv")