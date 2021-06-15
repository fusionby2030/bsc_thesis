import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import RepeatedKFold
from codebase.data import utils
from collections import OrderedDict
from tqdm import tqdm

def main(**kwargs):
    KL_cols = ['Meff', 'H(HD)', 'B(T)', 'P_ICRH(MW)', 'q95', 'P_TOTPNBIPohmPICRHPshi(MW)',
               'gasflowrateofmainspecies1022(es)', 'P_NBI(MW)', 'plasmavolume(m3)',
               'Ip(MA)', 'a(m)', 'averagetriangularity']
    KL_cols = list(reversed(KL_cols))  # if this isn't programming then I don't know what is

    feature_space, y, _, _, _, _ = utils.process_data(numerical_cols=KL_cols, return_numpy=True, return_necessary=False)
    y = y.to_numpy()
    y = y.ravel()

    ordered_preds = OrderedDict((index, []) for index in feature_space.index.to_list())
    ordered_error = OrderedDict((index, []) for index in feature_space.index.to_list())
    ordered_yvals = OrderedDict((index, []) for index in feature_space.index.to_list())
    x = feature_space.to_numpy()
    error_dict = {'MAE': [], 'RMSE': []}

    cv = RepeatedKFold(n_repeats=5, n_splits=5)
    iterator = tqdm(cv.split(x), desc='CV', leave=False, position=0)

    for train, test in iterator:
        X_train, X_test = x[train], x[test]
        y_train, y_test = y[train], y[test]

        if kwargs['type'] == 'RF':
            model = RandomForestRegressor(n_estimators=kwargs['n_estimators'], max_features=kwargs['n_features'])
        else:
            model = ExtraTreesRegressor(n_estimators=kwargs['n_estimators'], max_features=kwargs['n_features'])

        model.fit(X_train, y_train)
        estimations = np.array([est.predict(X_test) for est in model.estimators_])
        mean_preds = np.mean(estimations, axis=0)
        std_preds = np.std(estimations, axis=0)

        for index, std, mean_pred, yval in zip(test, std_preds, mean_preds, y_test):
            ordered_error[index].append(std)
            ordered_preds[index].append(mean_pred)
            ordered_yvals[index].append(yval)

        RMSE = mean_squared_error(y_true=y_test, y_pred=mean_preds, squared=False)
        MAE = mean_absolute_error(y_true=y_test, y_pred=mean_preds)
        error_dict['RMSE'].append(RMSE)
        error_dict['MAE'].append(MAE)
        iterator.set_postfix(RMSE=RMSE, MAE=MAE)

    list_uncert = []
    list_yvals = []
    list_preds = []
    for index in ordered_error.keys():
        list_uncert.append(np.mean(ordered_error[index]))
        list_yvals.append(np.mean(ordered_yvals[index]))
        list_preds.append(np.mean(ordered_preds[index]))

    scores = {'RMSE': (np.mean(error_dict['RMSE']), np.std(error_dict['RMSE'])),
              'MAE': (np.mean(error_dict['MAE']), np.std(error_dict['MAE']))}
    results = {'y_vals': np.array(list_yvals), 'preds': np.array(list_preds), 'uncert': np.array(list_uncert),
               'score': scores}

    return results

if __name__ == '__main__':
    np.random.seed(42)
    args = {'n_estimators': 142, 'type': 'ERT', 'n_features': None}
    results = main(**args)
    with open('./out/RF_ERT/ERT_UQ_best.pickle', 'wb') as file:
        pickle.dump(args, file)
        pickle.dump(results, file)