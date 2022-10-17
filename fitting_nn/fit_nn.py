import numpy as np
import time
import mlrose_hiive as mlrose
import matplotlib.pyplot as plt
from dataset import get_dataset
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
dataset_name = 'titanic'
X, y = get_dataset(dataset_name)


X_train, X_test, y_train, y_test = train_test_split(X.to_numpy().astype(np.float32), 
                                                    y.to_numpy().astype(np.float32), test_size=0.2, random_state=2)

for i in [2,10,20]:
    clf_hill = mlrose.NeuralNetwork(hidden_nodes = [40,40], activation = 'relu', 
                                algorithm = 'genetic_alg', 
                                max_iters=50, bias = True, is_classifier = True, 
                                learning_rate = i, early_stopping = True, clip_max = 1e+10, 
                                max_attempts = 10,curve=True,mutation_prob=0.4)
    clf_hill.fit(X_train, y_train)
    print(classification_report(y_test, clf_hill.predict(X_test)))
    plt.plot(clf_hill.fitness_curve[:,0],label="Step Size: "+str(i))
    plt.legend()
    fig=plt.gcf()
fig.set_size_inches(10,6)
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Loss", fontsize=18)
plt.savefig('nn_ga_epoch_vs_loss.png')
plt.clf()
for i in [5,10,20]:
    clf_hill = mlrose.NeuralNetwork(hidden_nodes = [40,40], activation = 'relu', 
                                algorithm = 'simulated_annealing', 
                                max_iters=1500, bias = True, is_classifier = True, 
                                learning_rate = i, early_stopping = True, clip_max =1e+10,
                                max_attempts = 100,curve=True)
    clf_hill.fit(X_train, y_train)
    print(classification_report(y_test, clf_hill.predict(X_test)))
    plt.plot(clf_hill.fitness_curve[:,0],label="Step Size: "+str(i))
    plt.legend()
    fig=plt.gcf()
fig.set_size_inches(10,6)
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Loss", fontsize=18)
plt.savefig('nn_sa_epoch_vs_loss.png')
plt.clf()

for i in [10,30,50]:
    clf_hill = mlrose.NeuralNetwork(hidden_nodes = [40,40], activation = 'relu', 
                                algorithm = 'random_hill_climb', 
                                max_iters=1500, bias = True, is_classifier = True, 
                                learning_rate = i, early_stopping = True, clip_max = 1e+10, 
                                max_attempts = 100,curve=True)
    clf_hill.fit(X_train, y_train)
    print(classification_report(y_test, clf_hill.predict(X_test)))
    plt.plot(clf_hill.fitness_curve[:,0],label="Step Size: "+str(i))
    plt.legend()
    fig=plt.gcf()

plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Loss", fontsize=18)
plt.savefig('nn_rhc_epoch_vs_loss.png')