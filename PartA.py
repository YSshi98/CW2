import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(training_path, testing_path):
    training_data = pd.read_csv(training_path)
    testing_data = pd.read_csv(testing_path)

    X_train = training_data.iloc[:, :-1]
    y_train = training_data.iloc[:, -1]
    X_test = testing_data

    return X_train, y_train, X_test

def create_model(X_train, y_train):
    model = RandomForestClassifier(random_state=0)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')

    # Display cross-validation scores as a table
    scores_table = pd.DataFrame(scores, columns=['Cross-Validation Scores'])
    scores_table.index.name = 'Iteration'
    print(scores_table)

    model.fit(X_train, y_train)

    return model

def save_results(testing_data, predicted_labels, output_path):
    testing_results = pd.DataFrame(testing_data)
    testing_results['marker'] = predicted_labels
    testing_results.to_csv(output_path, index=False)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_feature_importances(model):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

X_train, y_train, X_test = load_data("TrainingDataBinary.csv", "TestingDataBinary.csv")
model = create_model(X_train, y_train)
predicted_labels = model.predict(X_test)
save_results(X_test, predicted_labels, "TestingResultsBinary.csv")
f1 = f1_score(y_train, model.predict(X_train))
print("\nF1 Score:", f1)
plot_confusion_matrix(y_train, model.predict(X_train))
plot_feature_importances(model)
