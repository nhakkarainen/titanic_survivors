# Setup and import dependencies
from flask import Flask, jsonify, render_template, redirect, render_template_string, url_for
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import os

# Flask setup
app = Flask(__name__, template_folder="templates")

# Read data in CSV file
df = pd.read_csv("resources/titanic.csv")
df.to_csv("resources/titanic.csv", index = None)

# Flask routes
@app.route("/")
def Homepage():
    """List all API routes."""
    return render_template("index.html")

# APP ROUTE 1 - SHOW PASSENGER ROSTER
@app.route("/api/v1.0/dataset")
def dataset():
    # Read CSV and convert to HTML table
    data = pd.read_csv("resources/titanic.csv")
    return render_template("table.html", tables = [data.to_html()], titles = [""])

# APP ROUTE 2 - SHOW PROBABILITY OF SURVIVAL
@app.route("/api/v1.0/survival")
def survival():
    # Read File and store into Pandas data frame
    titanic_df = pd.read_csv("resources/titanic.csv")
    test_df = pd.read_csv("resources/titanic.csv")
    train_df = pd.read_csv("resources/titanic.csv")

    # Clean datasets
    titanic_df["Sex"] = titanic_df["Sex"].apply(lambda x: 1 if x == "female" else 0)
    train_df = train_df.drop(['Name'], axis=1)
    
    # Random Forest Classifier
    y = titanic_df['Survived']
    X = titanic_df.drop(columns=['Survived', 'Name'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    clf = RandomForestClassifier(random_state=1, n_estimators=500).fit(X_train, y_train)
    training_score = clf.score(X_train, y_train)
    testing_score = clf.score(X_test, y_test)
    return render_template_string("""Training Score: {{ train }} 
    Testing Score: {{ test }}""", train = training_score, test = testing_score)

# APP ROUTE 3 - SHOW TABLEAU VISUALIZATION
@app.route("/api/v1.0/graphs")
def graphs():
    return render_template("graphs.html")

if __name__ == '__main__':
    app.run(debug=True)
