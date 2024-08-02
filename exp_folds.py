import Ada_kNN as ada
import sklearn as sk
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load iris dataset
name = 'music'
dataset = pd.read_csv(f'Datasets/{name}.csv')
X = dataset.drop('class', axis=1).values
y = dataset['class'].values

# 10-fold indices split
kf = KFold(n_splits=10, shuffle=True, random_state=0)

# experiments on Ada-kNN
acc = []
aux_acc = []
print('=== Ada-kNN ===')
for i, (train_index, test_index) in enumerate(kf.split(X)):
    # data split
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    # predictions
    classifier = ada.Ada_kNN()
    classifier.fit(X_train, y_train)
    preds = classifier.predict(X_test)
    accF = sk.metrics.accuracy_score(y_true=y_test, y_pred=preds)
    aux_acc.append(accF)

    print(f"- Fold {i+1}: {accF}")
print(f'- Mean: {np.mean(aux_acc)}')
acc.append(aux_acc)

# experiments on kNN
print('\n=== kNN ===')
ks = [1,3,5,7,9]
for ki in ks:
    print(f'\n### K = {ki} ###')
    aux_acc = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        # data split
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        # predictions
        classifier = KNeighborsClassifier(n_neighbors=ki)
        classifier.fit(X_train, y_train)
        preds = classifier.predict(X_test)
        accF = sk.metrics.accuracy_score(y_true=y_test, y_pred=preds)
        aux_acc.append(accF)

        print(f"- Fold {i+1}: {accF}")
    print(f'- Mean: {np.mean(aux_acc)}')
    acc.append(aux_acc)

#acc = np.array(acc)
#print(acc)

# grafico
fig, ax = plt.subplots()
ax.set_ylabel('accuracy')
#ax.set_xlabel('')
ax.set_title(f'Accuracy on 10-folds on {name} dataset')
ax.boxplot(acc, tick_labels=['Ada-kNN', 'kNN(1)', 'kNN(3)', 'kNN(5)', 'kNN(7)', 'kNN(9)'])

fig.savefig(f'imgs/ks_{name}',bbox_inches ="tight",dpi=300)