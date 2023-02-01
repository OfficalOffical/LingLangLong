import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot





def main():
    mlModel()


def mlModel():
    # Read in the data
    data = pd.read_csv('balimCleaned.csv')
    # Print the first 5 rows of the data
    print(data.head())
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=1000, random_state=42)
    # Train the model on training data
    rf.fit(data, data)
    # Use the forest's predict method on the test data
    predictions = rf.predict(data)
    # Calculate the absolute errors
    errors = abs(predictions - data)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / data)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    # Pull out one tree from the forest
    tree = rf.estimators_[5]
    # Export the image to a dot file
    export_graphviz(tree, out_file='tree.dot', feature_names=data.columns, rounded=True, precision=1)
    # Use dot file to create a graph
    (graph, ) = pydot.graph_from_dot_file('tree.dot')
    # Write graph to a png file
    graph.write_png('tree.png')
    # Limit depth of tree to 3 levels
    rf_small = RandomForestRegressor(n_estimators=10, max_depth=3)
    rf_small.fit(data, data)
    # Extract the small tree
    tree_small = rf_small.estimators_[5]
    # Save the tree as a png image
    export_graphviz(tree_small, out_file='small_tree.dot', feature_names=data.columns, rounded=True, precision=1)
    (graph, ) = pydot.graph_from_dot_file('small_tree.dot')
    graph.write_png('small_tree.png')
    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(data.columns, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
    # New random forest with only the two most important variables
    rf_most_important = RandomForestRegressor(n_estimators=1000, random_state=42)
    # Extract the two most important features
    important_indices = [data.columns.get_loc('feature1'), data.columns.get_loc('feature2')]
    train_important = data.iloc[:, important_indices]
    # Train the random forest
    rf_most_important.fit(train_important, data)
    # Make predictions and determine the error
    predictions = rf_most_important.predict(train_important)
    errors = abs(predictions - data)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / data)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    # Import matplotlib for plotting and use magic command for Jupyter Notebooks
    # Set the style
    plt.style.use('fivethirtyeight')
    # list of x locations for plotting
    x_values = list(range(len(importances)))
    # Make a bar chart
    plt.bar(x_values, importances, orientation='vertical')
    # Tick labels for x axis
    plt.xticks(x_values, data.columns, rotation='vertical')
    # Axis labels and title
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.title('Variable Importances')
    plt.show()
    # Pull out one tree from the forest
    tree = rf.estimators_[5]
    # Export the image to a dot file
    export_graphviz(tree, out_file='tree.dot', feature_names=data.columns, rounded=True, precision=1)
    # Use dot file to create a graph
    (graph, ) = pydot.graph_from_dot_file('tree.dot')
    # Write graph to a png file
    graph.write_png('tree.png')
    # Limit depth of tree to 3 levels
    rf_small = RandomForestRegressor(n_estimators=10, max_depth=3)
    rf_small.fit(data, data)
    # Extract the small tree
    tree_small = rf_small.estimators_[5]
    # Save the tree as a png image
    export_graphviz(tree_small, out_file='small_tree.dot', feature_names=data.columns, rounded=True, precision=1)
    (graph, ) = pydot.graph_from_dot_file('small_tree.dot')
    graph.write_png('small_tree.png')














    


