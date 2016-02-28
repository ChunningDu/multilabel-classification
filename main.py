from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import reuters as dataset

X_train, Y_train = dataset.get_train_set()
X_test, Y_test = dataset.get_test_set()
classif = OneVsRestClassifier(SVC(kernel='linear'))
classif.fit(X_train, Y_train)
Y_test_predict = classif.predict(X_test)
[precision, recall, F1, support] = \
    precision_recall_fscore_support(Y_test, Y_test_predict, average='samples')
accuracy = accuracy_score(Y_test, Y_test_predict)
print 'accuracy, precision, recall, F1'
print accuracy, precision, recall, F1
