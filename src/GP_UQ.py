from codebase.data import utils
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from tqdm import tqdm
from collections import OrderedDict
import GPy
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


def map_fixed_kernel(vals, fix_type='1d'):
    if fix_type == '1d':
        fixed_kern = GPy.kern.Fixed(1, GPy.util.linalg.tdot(vals))
    elif fix_type == 'all':
        fixed_kern = GPy.kern.Fixed(10, GPy.util.linalg.tdot(vals))
    else:
        return None
    return fixed_kern


def map_kernel(kernel_name='MLP'):
    if kernel_name == 'MLP':
        kernel = GPy.kern.MLP(input_dim=10, ARD=True)
    else:
        kernel = GPy.kern.MLP(input_dim=10, ARD=True)
    return kernel


def map_optim_kernel(model, kernel_name='MLP'):
    if kernel_name == 'MLP':
        optim_kern = model.kern.mlp.copy()
    if kernel_name == 'None':
        optim_kern = model.kern.copy()
    else:
        optim_kern = model.kern.rq.copy()
    return optim_kern


def main(kernel_name='MLP'):
    KL_cols = ['B(T)', 'P_ICRH(MW)', 'q95', 'P_TOTPNBIPohmPICRHPshi(MW)',
               'gasflowrateofmainspecies1022(es)', 'P_NBI(MW)', 'plasmavolume(m3)',
               'Ip(MA)', 'a(m)', 'averagetriangularity']
    KL_cols = list(reversed(KL_cols))  # if this isn't programming then I don't know what is

    feature_space, y, _, _, target_error, feature_error = utils.process_data(numerical_cols=KL_cols, return_numpy=True,
                                                                             return_necessary=False)

    scaler_feature = StandardScaler()
    x = scaler_feature.fit_transform(feature_space)
    y = y.to_numpy()
    y_err, x_err = target_error.to_numpy(), feature_error.to_numpy()

    ordered_preds = OrderedDict((index, []) for index in feature_space.index.to_list())
    ordered_error = OrderedDict((index, []) for index in feature_space.index.to_list())
    ordered_yvals = OrderedDict((index, []) for index in feature_space.index.to_list())

    error_dict = {'MAE': [], 'RMSE': []}

    cv = RepeatedKFold(n_repeats=5, n_splits=5)
    iterator = tqdm(cv.split(x), desc='CV', leave=False, position=0)
    log_likelihood = []
    for train, test in iterator:
        X_train, X_test = x[train], x[test]
        y_train, y_test = y[train], y[test]

        fixed_kern = map_fixed_kernel(x_err[train], fix_type='none')

        base_kern = map_kernel(kernel_name)
        if fixed_kern is not None:
            kernel = base_kern + fixed_kern
        else:
            kernel = base_kern
        model = GPy.models.GPRegression(X_train, y_train, kernel)
        model.optimize(messages=True)
        print(model)
        optim_kern = map_optim_kernel(model, kernel_name='None')

        predictive_model = GPy.models.GPRegression(X_test, y_test, optim_kern)
        mean_preds, std_preds = predictive_model.predict(X_test)

        for index, std, mean_pred, yval in zip(test, std_preds, mean_preds, y_test):
            ordered_error[index].append(std)
            ordered_preds[index].append(mean_pred)
            ordered_yvals[index].append(yval)

        RMSE = mean_squared_error(y_true=y_test, y_pred=mean_preds, squared=False)
        MAE = mean_absolute_error(y_true=y_test, y_pred=mean_preds)
        error_dict['RMSE'].append(RMSE)
        error_dict['MAE'].append(MAE)
        iterator.set_postfix(RMSE=RMSE, MAE=MAE)
        log_likelihood.append(predictive_model.log_likelihood())
    list_uncert = []
    list_yvals = []
    list_preds = []
    for index in ordered_error.keys():
        list_uncert.append(np.mean(ordered_error[index]))
        list_yvals.append(np.mean(ordered_yvals[index]))
        list_preds.append(np.mean(ordered_preds[index]))

    scores = {'RMSE': (np.mean(error_dict['RMSE']),np.std(error_dict['RMSE'])), 'MAE': (np.mean(error_dict['MAE']),np.std(error_dict['MAE']))}
    results = {'y_vals': list_yvals, 'preds': list_preds, 'uncert': list_uncert, 'score': scores, 'MLL': log_likelihood}
    return results


import pickle

if __name__ == '__main__':
    results = main()
    print(results)
    with open('./out/GP/UQ_MLP_hetero.pickle', 'wb') as file:
        pickle.dump(results, file)
