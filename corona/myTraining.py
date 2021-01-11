import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
def data_split(data,ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data)*ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

if __name__ == "__main__":
    df=pd.read_csv('data.csv')
    train,test=data_split(df,0.2)
    X_train= train[['Fever','BodyPain','Age','RunnyNose','DifficultBreathing']]
    X_test= test[['Fever','BodyPain','Age','RunnyNose','DifficultBreathing']]
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    Y_train= train[['InfectionProb']]
    Y_test= test[['InfectionProb']]
    Y_train =np.array(Y_train)
    Y_test =np.array(Y_test)
    Y_train.reshape(3200 ,)
    Y_test.reshape(799,)
    clf=LogisticRegression()
    clf.fit(X_train,Y_train.ravel())
    file = open('model.pkl', 'wb')
    pickle.dump(clf, file)
    file.close()
    