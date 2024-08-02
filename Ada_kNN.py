import numpy as np
import tensorflow as tf

class Euclidean:
    def __init__(self):
        pass
    
    def distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
    

    def metric_aa(self, X):
        r = len(X) # number of rows (instances) in X
        distances = np.zeros(shape=(r,r), dtype=np.float32) # output matrix
        
        # triangular calculation of distances
        for i in range(r-1):
            for j in range(i+1, r):
                aux = self.distance(X[i], X[j]) # d(i,j) = d(j,i)
                distances[i,j] = aux
                distances[j,i] = aux
        
        return distances


    def metric_ab(self, X1, X2):
        r1 = len(X1) # number of rows (instances) in X1
        r2 = len(X2) # number of rows (instances) in X2
        distances = np.zeros(shape=(r1,r2), dtype=np.float32) # output matrix
        
        # calculation of distances
        for i in range(r1):
            for j in range(r2):
                distances[i,j] = self.distance(X1[i], X2[j])
        
        return distances


class Ada_kNN:
    def __init__(self, alpha=10, lr=0.01, batch_size=4, epochs=100): # 01
        self.metric = Euclidean() # distance metric
        self.alpha = alpha # number of points considered in the experiments for K_xi

        # hyperparameters for MLP training
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.tags = np.unique(self.y_train) # list of unique classes in y_train

        # calculation of cross distances of elements in X_train
        self.Dmatrix = self.metric.metric_aa(self.X_train) 

        self.r1, self.c1 = self.X_train.shape # number of instances and atributes of X_train
        self.k_max = int(np.ceil(np.sqrt(self.r1))) # k max value
        #print(f'k_max: {self.k_max}')

        # identification of the best suited values of k for classifing every xi in X_train
        self.Kxs = []
        for xi in range(self.r1): # xi is the index on instances
        #for xi in range(5):
            Kxi = [] # optimal k values for xi
            auxk = [j for j in range(1,self.k_max+1)]
            Krand = np.random.permutation(auxk) # permutation of posible k values
            #print(f'Krand: {Krand}')

            j = 0 # counter
            while (j < self.alpha or len(Kxi)==0) and j < self.k_max: # perform alpha experiment or the first succesfull case or all cases (k_max)
                kNNs_ind = self.get_kNN(k=Krand[j]+1, xi=xi) # get the k+1-NN indexes to xi (get the k+1 since the nearest neigboh is itself)
                pred = self.vote(kNNs_ind[1:]) # the first index is not considered beacuse it belongs to xi
                #print(f'kj: {Krand[j]}  pred: {pred}  y_train: {self.y_train[xi]}')
                if pred == self.y_train[xi]:
                    Kxi.append(Krand[j]) # add succesfull k values
                j += 1
            
            # posible case when no value in permutation perfroms a succesfull clasification
            if j == self.k_max:
                #print(f'{xi} EMPTY')
                Kxi = list(Krand)
            #print(f'Kxi: {Kxi}')
            self.Kxs.append(Kxi)
        #print(f'Kxs: {self.Kxs}')

        # set of k values used for training the MLP
        k_train = []
        for Kxi in self.Kxs:
            if len(Kxi) == 1: 
                k_train.append(Kxi[0])
            else:
                auxi = np.random.randint(0, len(Kxi))
                k_train.append(Kxi[auxi])
        k_train = np.array(k_train)
        k_train = k_train-1 # set into the range 0 to k_max-1 to traning
        #print(f'k_train: {k_train}')

        # definition of MLP
        input = tf.keras.layers.Input((self.c1,))
        bn = tf.keras.layers.BatchNormalization()(input)
        d1 = tf.keras.layers.Dense(32)(bn)
        d2 = tf.keras.layers.Dense(32)(d1)
        d3 = tf.keras.layers.Dense(1, 'sigmoid')(d2)
        out = d3*(self.k_max-1)
        self.MLP = tf.keras.Model(input, out)

        # compilation
        self.MLP.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss='mean_squared_error'
        )

        # training
        self.MLP.fit(self.X_train, k_train, batch_size=self.batch_size, epochs=self.epochs, verbose=0)
    
    def get_kNN(self, k, xi):
        distances = self.Dmatrix[xi, 0:self.r1] # distances between xi and all the points on X_train
        k_i = np.argsort(distances)[:int(k)] # indeces of the k neares neighbors of xi
        return k_i

    def vote(self, inds):
        k_labels = [self.y_train[i] for i in inds] # class of each index
        values, counts = np.unique(k_labels, return_counts = True) # repetions of each class
        
        # index of the majority vote class
        ind = np.argmax(counts) 
        tag = values[ind]
        return tag

    def predict(self, X_test):
        k_test = self.MLP.predict(X_test) # predict the k values with trained MLP
        #k_test - np.reshape(k_test, (len(X_test),))
        k_test = np.squeeze(k_test)
        k_test = np.rint(k_test) + 1. # add 1 to fit the real k values
        #print(f'k_test: {k_test}')

        # calcualte the distances betwen X_train and X_test
        dist = self.metric.metric_ab(X_test, self.X_train)
        auxD = self.Dmatrix
        self.Dmatrix = np.concatenate([self.Dmatrix, dist], axis=0)
        #print(f'Dmatrix: {self.Dmatrix}')

        # perform clasification of X_test
        preds = [] # predictions
        for i in range(len(X_test)):
            kNNs_ind = self.get_kNN(k_test[i], xi=self.r1+i) # get the k-NN xi (considering the matriz distance)
            pred = self.vote(kNNs_ind) # votation
            preds.append(pred)
        return preds
            









if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    dataset = pd.read_csv('Datasets/wine.csv')
    Xs = dataset.drop('class', axis=1).values
    ys = dataset['class'].values

    # dividimos datos
    X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.2, random_state=2) #12

    classifier = Ada_kNN()
    classifier.fit(X_train, y_train)
    preds = classifier.predict(X_test)
    
    print(f'Preds: {preds}')
    print(f'y_test: {list(y_test)}')
    print(f'accuracy: {accuracy_score(y_true=y_test, y_pred=preds)}')