import requests
import numpy as np
from sklearn.externals.joblib import Memory
from sklearn import datasets, svm
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
from os import listdir
from flask import Flask, request
from flask import jsonify
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import json
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from flask import Flask, request, send_file, make_response
import io

def get_data(filename):
    data = load_svmlight_file(filename)
    return data[0], data[1]

def iris_svm(arg1,arg2):
    # get the iris dataset
    iris_data = datasets.load_iris()
    X = iris_data.data
    y = iris_data.target
    class_names = iris_data.target_names
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    c_value = float(arg1)
    kernel_val = arg2
    my_classifier = SVC(kernel=kernel_val, C = c_value)
    y_pred = my_classifier.fit(X_train, y_train).predict(X_test)
    results = confusion_matrix(y_test,y_pred)
    results = results.tolist()
    results_str = json.dumps(results)
    scores = my_classifier.score(X_train, y_train)
    scores = scores.tolist()
    scores_str = json.dumps(scores)
    all_res = [results_str, scores_str, "hello", "Again hello"]
    
    return all_res

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return bytes_image

# Important things to note are io.BytesIO() and send_file magicial powers, io from the io module and send_file is flask

def gen_cof_mat(arg1,arg2,norm1):
    iris_data = datasets.load_iris()
    x = iris_data.data
    y = iris_data.target
    class_names = iris_data.target_names
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    c_value = float(arg1)
    kernel_val = arg2
    classifier = SVC(kernel=kernel_val, C = c_value)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    # Plot normalized confusion matrix
    bytes_obj = plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=norm1,title=None,cmap=plt.cm.Blues)

    return send_file(bytes_obj,
                     attachment_filename='plot.png',
                     mimetype='image/png')



def svm():
    #Xtrain, ytrain = get_data("data/iris.scale_train")
    #Xtest, ytest = get_data("data/iris.scale_test")
    Xtrain, ytrain = get_data("data/test_train_0.8")
    Xtest, ytest = get_data("data/test_test_0.8")

    clf = SVC(gamma=0.001, C=100, kernel='linear')
    clf.fit(Xtrain, ytrain)

    test_size = Xtest.shape[0]
    accuarcy_holder = []
    for i in range(0, test_size):
        prediction = clf.predict(Xtest[i])
        print("Prediction from SVM: "+str(prediction)+", Expected Label : "+str(ytest[i]))
        accuarcy_holder.append(prediction==ytest[i])

    correct_predictions = sum(accuarcy_holder)
    print(correct_predictions)
    total_samples = test_size
    accuracy = float(float(correct_predictions)/float(total_samples))*100
    print("Prediction Accuracy: "+str(accuracy))
    return "Prediction Accuracy: "+str(accuracy)+" Total Samples: "+str(total_samples)+" Correct Predictions: "+str(correct_predictions)
