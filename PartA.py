import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(training_path, testing_path):
    # Load training data from the specified path
    training_data = pd.read_csv(training_path)

    # Load testing data from the specified path
    testing_data = pd.read_csv(testing_path)

    # Extract the features (inputs) from the training data
    X_train = training_data.iloc[:, :-1]

    # Extract the labels (outputs) from the training data
    y_train = training_data.iloc[:, -1]

    # Assign the testing data as the input for testing
    X_test = testing_data

    # Return the training features, training labels, and testing features
    return X_train, y_train, X_test

def create_model(X_train, y_train):
    # Create a RandomForestClassifier model with a random_state of 0
    model = RandomForestClassifier(random_state=0)

    # Perform cross-validation using the training data
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')

    # Display the cross-validation scores as a table
    scores_table = pd.DataFrame(scores, columns=['Cross-Validation Scores'])
    scores_table.index.name = 'Iteration'
    print(scores_table)

    # Fit the model on the entire training data
    model.fit(X_train, y_train)

    # Return the trained model
    return model

def save_results(testing_data, predicted_labels, output_path):
    # Create a DataFrame with the testing data
    testing_results = pd.DataFrame(testing_data)

    # Add a column 'marker' containing the predicted labels
    testing_results['marker'] = predicted_labels

    # Save the results to a CSV file at the specified output path
    testing_results.to_csv(output_path, index=False)

def plot_confusion_matrix(y_true, y_pred):
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix as a heatmap with annotations
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_feature_importances(model):
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort indices in descending order
    indices = np.argsort(importances)[::-1]

    # Create a new figure for the plot
    plt.figure()
    # Set the title of the plot
    plt.title("Feature importances")
    # Create a bar plot with the sorted importances
    plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
    # Set the x-axis labels to the sorted indices
    plt.xticks(range(X_train.shape[1]), indices)
    # Set the x-axis limits
    plt.xlim([-1, X_train.shape[1]])
    # Show the plot
    plt.show()

# Load the training and testing data
X_train, y_train, X_test = load_data("TrainingDataBinary.csv", "TestingDataBinary.csv")
# Create a machine learning model
model = create_model(X_train, y_train)
# Predict labels for the testing data
predicted_labels = model.predict(X_test)
# Save the predicted labels to a file
save_results(X_test, predicted_labels, "TestingResultsBinary.csv")
# Calculate the F1 score on the training data
f1 = f1_score(y_train, model.predict(X_train))
# Print the F1 score
print("\nF1 Score:", f1)
# Plot the confusion matrix for the training data
plot_confusion_matrix(y_train, model.predict(X_train))
# Plot the feature importances of the model
plot_feature_importances(model)
