# ===================== IMPORT LIBRARIES =====================

import tkinter as tk
from tkinter import filedialog, Text, Scrollbar, Label, Button
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm


# ===================== GLOBAL VARIABLES =====================

dataset = None
X, Y = None, None
X_train, X_test, y_train, y_test = None, None, None, None

results = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "fscore": []
}

model_rf = None


# ===================== GUI SETUP =====================

app = tk.Tk()
app.title("Fraud Detection using Machine Learning")
app.geometry("1200x700")

text_area = Text(app, height=20, width=120)
scroll = Scrollbar(text_area)

text_area.configure(yscrollcommand=scroll.set)
text_area.pack()


# ===================== UPLOAD DATASET =====================

def upload_dataset():
    global dataset

    file_path = filedialog.askopenfilename()
    dataset = pd.read_csv(file_path)

    text_area.delete('1.0', tk.END)
    text_area.insert(tk.END, "Dataset Loaded Successfully\n\n")
    text_area.insert(tk.END, str(dataset.head()))

    # Plot class distribution
    dataset.groupby('FLAG').size().plot(
        kind="bar",
        title="Fraud vs Normal"
    )

    plt.show()


# ===================== PREPROCESSING =====================

def preprocess_data():
    global X, Y, X_train, X_test, y_train, y_test

    dataset.fillna(0, inplace=True)

    Y = dataset['FLAG'].values

    X = dataset.iloc[:, 4:-2].values

    X = normalize(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42
    )

    text_area.insert(
        tk.END,
        "\nData Preprocessing Completed\n"
    )


# ===================== EVALUATION METRICS =====================

def evaluate_model(name, y_pred):

    acc = accuracy_score(y_test, y_pred) * 100

    pre = precision_score(
        y_test,
        y_pred,
        average='macro'
    ) * 100

    rec = recall_score(
        y_test,
        y_pred,
        average='macro'
    ) * 100

    f1 = f1_score(
        y_test,
        y_pred,
        average='macro'
    ) * 100

    results["accuracy"].append(acc)
    results["precision"].append(pre)
    results["recall"].append(rec)
    results["fscore"].append(f1)

    text_area.insert(
        tk.END,
        f"\n{name} Results:\n"
    )

    text_area.insert(
        tk.END,
        f"Accuracy : {acc:.2f}%\n"
    )

    text_area.insert(
        tk.END,
        f"Precision: {pre:.2f}%\n"
    )

    text_area.insert(
        tk.END,
        f"Recall : {rec:.2f}%\n"
    )

    text_area.insert(
        tk.END,
        f"F1 Score : {f1:.2f}%\n"
    )


# ===================== MACHINE LEARNING MODELS =====================

def run_logistic():

    model = LogisticRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    evaluate_model(
        "Logistic Regression",
        y_pred
    )


def run_decision_tree():

    model = DecisionTreeClassifier()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    evaluate_model(
        "Decision Tree",
        y_pred
    )


def run_naive_bayes():

    model = GaussianNB()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    evaluate_model(
        "Naive Bayes",
        y_pred
    )


def run_svm():

    model = svm.SVC()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    evaluate_model(
        "SVM",
        y_pred
    )


def run_random_forest():

    global model_rf

    model_rf = RandomForestClassifier()

    model_rf.fit(X_train, y_train)

    y_pred = model_rf.predict(X_test)

    evaluate_model(
        "Random Forest",
        y_pred
    )


def run_adaboost():

    model = AdaBoostClassifier()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    evaluate_model(
        "AdaBoost",
        y_pred
    )


def run_mlp():

    model = MLPClassifier()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    evaluate_model(
        "MLP",
        y_pred
    )


# ===================== PREDICTION =====================

def predict_data():

    global model_rf

    file_path = filedialog.askopenfilename()

    test_data = pd.read_csv(file_path)

    test_data.fillna(0, inplace=True)

    X_new = normalize(
        test_data.iloc[:, 4:-2].values
    )

    predictions = model_rf.predict(X_new)

    text_area.insert(
        tk.END,
        "\nPrediction Results:\n"
    )

    for i, val in enumerate(predictions):

        result = "FRAUD" if val == 1 else "NORMAL"

        text_area.insert(
            tk.END,
            f"Record {i+1}: {result}\n"
        )


# ===================== BUTTONS =====================

Button(
    app,
    text="Upload Dataset",
    command=upload_dataset
).pack()

Button(
    app,
    text="Preprocess Data",
    command=preprocess_data
).pack()

Button(
    app,
    text="Logistic Regression",
    command=run_logistic
).pack()

Button(
    app,
    text="Decision Tree",
    command=run_decision_tree
).pack()

Button(
    app,
    text="Naive Bayes",
    command=run_naive_bayes
).pack()

Button(
    app,
    text="SVM",
    command=run_svm
).pack()

Button(
    app,
    text="Random Forest",
    command=run_random_forest
).pack()

Button(
    app,
    text="AdaBoost",
    command=run_adaboost
).pack()

Button(
    app,
    text="MLP",
    command=run_mlp
).pack()

Button(
    app,
    text="Predict",
    command=predict_data
).pack()


# ===================== RUN APP =====================

app.mainloop()
