"""
分类器工厂 (pipeline.py 和 run.py 共用)
"""
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

CLASSIFIERS = {
    'SVM-RBF':     lambda: SVC(kernel='rbf', random_state=42),
    'SVM-Linear':  lambda: LinearSVC(max_iter=5000, random_state=42, dual='auto'),
    'LR':          lambda: LogisticRegression(max_iter=5000, random_state=42),
    'KNN-3':       lambda: KNeighborsClassifier(3),
    'KNN-5':       lambda: KNeighborsClassifier(5),
    'KNN-7':       lambda: KNeighborsClassifier(7),
    'RF':          lambda: RandomForestClassifier(n_estimators=100, random_state=42),
}
CLF_NAMES = list(CLASSIFIERS.keys())
