import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, RepeatedKFold, train_test_split

from codebase.data import utils

KL_cols = ['Meff', 'H(HD)', 'B(T)', 'P_ICRH(MW)', 'q95', 'P_TOTPNBIPohmPICRHPshi(MW)',
               'gasflowrateofmainspecies1022(es)', 'P_NBI(MW)', 'plasmavolume(m3)',
               'Ip(MA)', 'a(m)', 'averagetriangularity']
KL_cols = list(reversed(KL_cols))  # if this isn't programming then I don't know what is

feature_space, y, _, _, _, _ = utils.process_data(numerical_cols=KL_cols, return_numpy=True, return_necessary=False)

high_neped = y[y['nepedheight1019(m3)'] > 8.25].index.to_numpy()

class_targets = np.zeros(len(y))
class_targets[high_neped] = 1
print(np.sum(class_targets) / len(class_targets))
clf = RandomForestClassifier()
scores = cross_validate(clf, feature_space, class_targets, cv=5, scoring=['accuracy', 'f1'])
X_train, X_test, y_train, y_test = train_test_split(feature_space, class_targets,
                                                    test_size=.5,
                                                    random_state=42)
print(scores)
print(np.mean(scores['test_accuracy']))
print(scores['test_f1'])

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import average_precision_score

classifier = svm.LinearSVC(random_state=42)
classifier.fit(X_train, y_train)
y_score = classifier.decision_function(X_test)

average_precision = average_precision_score(y_test, y_score)

disp = plot_precision_recall_curve(classifier, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))

plt.show()

clf.fit(X_train, y_train)
# y_score = clf.predict_proba(X_test)

# average_precision = average_precision_score(y_test, y_score)
disp = plot_precision_recall_curve(clf, X_test, y_test)
"""disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))
"""
plt.show()