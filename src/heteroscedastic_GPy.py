from codebase.data import utils
from sklearn.preprocessing import StandardScaler

import numpy as np
import GPy


KL_cols = ['B(T)', 'P_ICRH(MW)', 'q95', 'P_TOTPNBIPohmPICRHPshi(MW)',
               'gasflowrateofmainspecies1022(es)', 'P_NBI(MW)', 'plasmavolume(m3)',
               'Ip(MA)', 'a(m)', 'averagetriangularity']
KL_cols = list(reversed(KL_cols))  # if this isn't programming then I don't know what is

feature_space, y, _, _, target_error, feature_error = utils.process_data(numerical_cols=KL_cols, return_numpy=True,
                                                                         return_necessary=False)

from sklearn.model_selection import RepeatedKFold
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import OrderedDict
scaler_feature = StandardScaler()
x = scaler_feature.fit_transform(feature_space)
y = y.to_numpy()
y_err, x_err = target_error.to_numpy(), feature_error.to_numpy()

cv = RepeatedKFold(n_repeats=5, n_splits=5)
iterator = tqdm(cv.split(x), desc='CV', leave=False, position=0)
log_likelihood = []
ordered_preds = OrderedDict((index, []) for index in feature_space.index.to_list())
ordered_error = OrderedDict((index, []) for index in feature_space.index.to_list())
ordered_yvals = OrderedDict((index, []) for index in feature_space.index.to_list())

error_dict = {'MAE': [], 'RMSE': []}
for train, test in iterator:
    X_train, X_test = x[train], x[test]
    y_train, y_test = y[train], y[test]

    kern = GPy.kern.MLP(10)

    m = GPy.models.GPHeteroscedasticRegression(X_train, y_train, kern)
    # m['.*het_Gauss.variance'] = abs(y_err[train])
    # m.het_Gauss.variance.fix()
    m.optimize()

    mean_preds, std_preds = m._raw_predict(X_test)

    for index, std, mean_pred, yval in zip(test, std_preds, mean_preds, y_test):
        ordered_error[index].append(std)
        ordered_preds[index].append(mean_pred)
        ordered_yvals[index].append(yval)

    RMSE = mean_squared_error(y_true=y_test, y_pred=mean_preds, squared=False)
    MAE = mean_absolute_error(y_true=y_test, y_pred=mean_preds)

    error_dict['RMSE'].append(RMSE)
    error_dict['MAE'].append(MAE)
    iterator.set_postfix(RMSE=RMSE, MAE=MAE)
    log_likelihood.append(m.log_likelihood())

list_uncert = []
list_yvals = []
list_preds = []
for index in ordered_error.keys():
    list_uncert.append(np.mean(ordered_error[index]))
    list_yvals.append(np.mean(ordered_yvals[index]))
    list_preds.append(np.mean(ordered_preds[index]))

scores = {'RMSE': (np.mean(error_dict['RMSE']),np.std(error_dict['RMSE'])), 'MAE': (np.mean(error_dict['MAE']),np.std(error_dict['MAE']))}
results = {'y_vals': np.array(list_yvals), 'preds': np.array(list_preds), 'uncert': np.array(list_uncert), 'score': scores, 'MLL': log_likelihood}

print(results)
import pickle
with open('./out/GP/UQ_MLP_heteroscedastic_control.pickle', 'wb') as file:
    pickle.dump(results, file)
