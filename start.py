from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, Normalizer

from arff import Arff
from graph_tools import *
import pandas as pd
import matplotlib.pyplot as plt

from mlp import MLPClassifier

if __name__ == "__main__":
    # mat = Arff("linsep2nonorigin.arff", label_count=1)
    # mat = Arff("data_banknote_authentication.arff", label_count=1)
    # mat = Arff("iris.arff", label_count=1)
    mat = Arff("vowel.arff", label_count=1)

    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1].reshape(-1, 1)

    # remove vowel redundant data column
    data = data[:,2:]

    # normalizing
    transformer = Normalizer().fit(data)
    data = transformer.transform(data)
    # one hot encoding
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(labels)
    labels = enc.transform(labels).toarray()
    # splitting data into test and training
    X, X_test, y, y_test = train_test_split(data, labels, test_size=0.25)

    instance, features = X.shape
    # MLP = MLPClassifier([features*2], lr=0.1, momentum=0.5, shuffle=False, deterministic=10)
    MLP = MLPClassifier([features*2], lr=0.1, momentum=0.9, shuffle=True)
    MLP.fit(X, y)

    # Accuracy = MLP.score(data, labels)
    classifying_rate = MLP.score(X_test, y_test)
    MSE = MLP.getMSE(X_test, y_test)
    # print("testing MSE")
    # print(MSE)
    # print(MLP.epoch_counter)
    # print("accuracy")
    # print(classifying_rate)
    # np.savetxt("VS.csv", MLP.VS_MSE_history, delimiter=',')
    # np.savetxt("training.csv", MLP.training_MSE_history, delimiter=',')
    # np.savetxt("class_accu.csv",MLP.classification_accuracy_history,delimiter=',')


