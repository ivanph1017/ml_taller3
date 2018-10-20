import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix,roc_curve
import csv


file_train="svm.csv"
file_test="svm.csv"

def get_data():
    data=np.genfromtxt("svm.csv",delimiter=",", dtype=np.unicode_)
    
    X=data[:,:-1].astype(float)    
    X=np.delete(X,3,axis=1)
    print(X)
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    # Formula para estandarizar
    X = (X - means) / stds
        
    
    y=data[:,-1]
    le = preprocessing.LabelEncoder()
    #le.fit(y)
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = get_data()
    print 
    clf = svm.SVC(C=1, class_weight=None, coef0=0.0,
                           decision_function_shape=None, degree=3, gamma='auto',
                           kernel='linear', max_iter=-1, probability=False,
                           random_state=None, shrinking=True, tol=0.001, verbose=False)          
    
    clf.fit(X_train, y_train)   
    training_result = clf.predict(X_train)
    print metrics.classification_report(y_train, training_result)
    
    
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=20)
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='f1_macro')
    print scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    


if __name__=="__main__":
    main()
