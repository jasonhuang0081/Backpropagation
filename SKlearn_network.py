from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer, OneHotEncoder

from arff import Arff



if __name__ == "__main__":
    # mat = Arff("iris.arff", label_count=1)
    mat = Arff("diabetes.arff", label_count=1)

    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1].reshape(-1, 1)

    # normalizing
    transformer = Normalizer().fit(data)
    data = transformer.transform(data)
    # # one hot encoding
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(labels)
    labels = enc.transform(labels).toarray()
    X, X_test, y, y_test = train_test_split(data, labels, test_size=0.25)
    instance, features = X.shape

    tuned_parameters =  {'activation': ['relu','logistic'], 'solver': ['lbfgs','adam'],
                     'alpha': [0.00001,0.0001, 0.001, 0.01, 0.1], 'learning_rate':['adaptive','constant'],
                         'learning_rate_init':[1,0.1,0.01,0.001,0.0001],'momentum':[0.9,0.5,0.1], 'max_iter':[500]
                         ,'hidden_layer_sizes':[(features*3, features*2),(features*3),(features*2),(features*3, features*3)],
                         'tol':[0.005],'early_stopping':[True, False]}
    clf = RandomizedSearchCV(MLPClassifier(), tuned_parameters)
    clf.fit(X,y)
    print(clf.best_score_)
    print(clf.best_params_)

    # clf = MLPClassifier(solver='adam', hidden_layer_sizes=(features*3, features*2), learning_rate='adaptive',
    #                     alpha=0.00001, momentum=0.9, early_stopping=True, max_iter=500, learning_rate_init=0.1, activation='relu')
    # clf.fit(X,y)

    accuracy = clf.score(X_test,y_test)
    print(accuracy)