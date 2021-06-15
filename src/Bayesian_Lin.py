import numpy as np
import pickle

from sacred import Experiment
from sacred.observers import FileStorageObserver

from codebase.data import utils

from collections import OrderedDict

from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer

ex = Experiment()

ex.observers.append(FileStorageObserver('./out/Linear/bayes'))


@ex.config
def bayes_config():
    target_col = ['nepedheight1019(m3)']

    numerical_cols = ['averagetriangularity', 'a(m)', 'Ip(MA)', 'P_NBI(MW)',
                      'plasmavolume(m3)', 'P_TOTPNBIPohmPICRHPshi(MW)', 'q95',
                      'gasflowrateofmainspecies1022(es)', 'B(T)', 'P_ICRH(MW)', 'H(HD)',
                      'Meff']
    scale_data = None



@ex.capture
def get_model():
    return BayesianRidge(compute_score=True)


def calculate_scores(predictions, targets):
    """
    Calculate the RMSE, MAE
    """
    RMSE = np.sqrt(np.mean((predictions - targets) ** 2))
    MAE = (np.sum(np.abs(predictions - targets))) / len(predictions)
    return RMSE, MAE


def unravel_all(ordered_error, ordered_yvals, ordered_preds):
    list_uncert = []
    list_yvals = []
    list_preds = []
    for index in ordered_error.keys():
        list_uncert.append(np.mean(ordered_error[index]))
        list_yvals.append(np.mean(ordered_yvals[index]))
        list_preds.append(np.mean(ordered_preds[index]))
    return list_yvals, list_preds, list_uncert


@ex.automain
def main(target_col, numerical_cols, scale_data):
    feature_space, target_space, _, _, = utils.process_data(target_col, numerical_cols)

    if scale_data is not None:
        feat_scale = scale_data
        X = feat_scale.fit_transform(feature_space)
        y = target_space.to_numpy().ravel()
    else:
        X = feature_space.to_numpy()
        y = target_space.to_numpy().ravel()
    ordered_preds = OrderedDict((index, []) for index in feature_space.index.to_list())
    ordered_error = OrderedDict((index, []) for index in feature_space.index.to_list())
    ordered_yvals = OrderedDict((index, []) for index in feature_space.index.to_list())
    error_dict = {'MAE': [], 'RMSE': []}
    cv = RepeatedKFold(n_splits=5, n_repeats=5)

    for cv_id, (train, test) in enumerate(cv.split(X)):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        model = get_model()
        model.fit(X_train, y_train)
        # print(model.scores_)
        mean_preds, std_preds = model.predict(X_test, return_std=True)
        for index, std, mean_pred, yval in zip(test, std_preds, mean_preds, y_test):
            ordered_error[index].append(std)
            ordered_preds[index].append(mean_pred)
            ordered_yvals[index].append(yval)
        RMSE, MAE = calculate_scores(mean_preds, y_test)
        error_dict['RMSE'].append(RMSE)
        error_dict['MAE'].append(MAE)

    list_yvals, list_preds, list_uncert = unravel_all(ordered_error, ordered_yvals, ordered_preds)

    scores = {'RMSE': (np.mean(error_dict['RMSE']), np.std(error_dict['RMSE'])),
              'MAE': (np.mean(error_dict['MAE']), np.std(error_dict['MAE']))}
    results = {'y_vals': np.array(list_yvals), 'preds': np.array(list_preds), 'uncert': np.array(list_uncert),
               'score': scores}
    with open('./out/Linear/bayes_lin_reg.pickle', 'wb') as file:
        pickle.dump(results, file)
    ex.add_artifact('./out/Linear/bayes_lin_reg.pickle')
    utils.plot_predictions(list_preds, list_yvals)

    return results
