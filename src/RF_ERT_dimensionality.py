"""
EXP for RFS and ERTs to see how they perform with different number of feature sampling

pickle file contains
    - args : - min ests, max_ests, ERT vs RF
    - results: RMSE and MAE list for each number estimators in range (min_est, max_est) for each feature sampled (1-> 12)
        - also includes the feature importances averaged over all the estimator sizes
To do some plotting
for label, clf_err in results[1].items():
    xs, ys = zip(*clf_err)
"""

from codebase.data import utils
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from collections import OrderedDict
from tqdm import tqdm
import pickle
import argparse

def main(ensemble_list, **kwargs):
    KL_cols = ['Meff', 'H(HD)', 'B(T)', 'P_ICRH(MW)', 'q95', 'P_TOTPNBIPohmPICRHPshi(MW)',
               'gasflowrateofmainspecies1022(es)', 'P_NBI(MW)', 'plasmavolume(m3)',
               'Ip(MA)', 'a(m)', 'averagetriangularity']
    KL_cols = list(reversed(KL_cols))  # if this isn't programming then I don't know what is

    feature_space, y, _, _ = utils.process_data(numerical_cols=KL_cols, return_numpy=True)
    y = y.ravel()
    RMSE_rate = OrderedDict((label, []) for label, _ in ensemble_list)
    MAE_rate = OrderedDict((label, []) for label, _ in ensemble_list)
    features_chosen = dict((label, []) for label,_ in ensemble_clfs)

    for label, reg in ensemble_list:
        running_feature_importances = np.zeros(len(KL_cols))
        iterator = tqdm(range(kwargs['min_estimators'], kwargs['max_estimators'] + 1), position=0, leave=True, desc=label)
        for i in iterator:
            reg.set_params(n_estimators=i)
            scores = cross_validate(reg, feature_space, y,
                                    scoring=['neg_root_mean_squared_error', 'neg_mean_absolute_error'], n_jobs=-1,
                                    return_estimator=True, cv=5)

            estimators = scores['estimator']
            feature_importances = [est.feature_importances_ for est in estimators]
            feature_import = np.stack(feature_importances)
            averages = np.average(feature_import, axis=0)
            running_feature_importances += averages

            RMSE_rate[label].append((i, np.abs(np.mean(scores['test_neg_root_mean_squared_error']))))
            MAE_rate[label].append((i, np.abs(np.mean(scores['test_neg_mean_absolute_error']))))

        running_feature_importances /= float(kwargs['max_estimators'] - kwargs['min_estimators'] + 1)

        features_chosen[label] = running_feature_importances

    return (RMSE_rate, MAE_rate, features_chosen), KL_cols

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-type", help='RF for RFs and ERT for ERTs', type=str, choices=['ERT', 'RF'])
    parser.add_argument('-max_estimators', help='max number of decision trees in ensemble to test, i.e., end size', type=int, default=394)
    parser.add_argument('-min_estimators', help='min number of decision trees in ensemble to test, i.e., start size', type=int, default=20)
    parser.add_argument('--smoke_test', help='Smoke Test, quickly check if it works', action="count", default=0)
    args_namespace = parser.parse_args()
    args = vars(args_namespace)

    ensemble_clfs = [
        ("max_features=2", RandomForestRegressor(warm_start=True, max_features=2, random_state=42)),
        ("max_features=3", RandomForestRegressor(warm_start=True, max_features=3, random_state=42)),
        ("max_features=4", RandomForestRegressor(warm_start=True, max_features=4, random_state=42)),
        ("max_features=5", RandomForestRegressor(warm_start=True, max_features=5, random_state=42)),
        ("max_features=6", RandomForestRegressor(warm_start=True, max_features=6, random_state=42)),
        ("max_features=7", RandomForestRegressor(warm_start=True, max_features=7, random_state=42)),
        ("max_features=8", RandomForestRegressor(warm_start=True, max_features=8, random_state=42)),
        ("max_features=9", RandomForestRegressor(warm_start=True, max_features=9, random_state=42)),
        ("max_features=10", RandomForestRegressor(warm_start=True, max_features=10, random_state=42)),
        ("max_features=11", RandomForestRegressor(warm_start=True, max_features=11, random_state=42)),
        ("max_features=all", RandomForestRegressor(warm_start=True, max_features=None, random_state=42))]

    ensemble_clfs_ET = [
        ("max_features=2", ExtraTreesRegressor(warm_start=True, bootstrap=True, max_features=2, random_state=42)),
        ("max_features=3", ExtraTreesRegressor(warm_start=True, bootstrap=True, max_features=3, random_state=42)),
        ("max_features=4", ExtraTreesRegressor(warm_start=True, bootstrap=True, max_features=4, random_state=42)),
        ("max_features=5", ExtraTreesRegressor(warm_start=True, bootstrap=True, max_features=5, random_state=42)),
        ("max_features=6",
         ExtraTreesRegressor(warm_start=True, bootstrap=True, max_features=6, random_state=42)),
        ("max_features=7",
         ExtraTreesRegressor(warm_start=True, bootstrap=True, max_features=7, random_state=42)),
        ("max_features=8",
         ExtraTreesRegressor(warm_start=True, bootstrap=True, max_features=8, random_state=42)),
        ("max_features=9",
         ExtraTreesRegressor(warm_start=True, bootstrap=True, max_features=9, random_state=42)),
        ("max_features=10",
         ExtraTreesRegressor(warm_start=True, bootstrap=True, max_features=10, random_state=42)),
        ("max_features=11",
         ExtraTreesRegressor(warm_start=True, bootstrap=True, max_features=11, random_state=42)),
        ("max_features=all",
         ExtraTreesRegressor(warm_start=True, bootstrap=True, max_features=None, random_state=42))
    ]
    if args['type'] == 'ERT':
        ensemble_regs = ensemble_clfs_ET
    else:
        ensemble_regs = ensemble_clfs
    np.random.seed(42)
    if args['smoke_test']:
        args['min_estimators'] = 15
        args['max_estimators'] = 25
        args['type'] = 'RF'
        results, features = main(ensemble_list=ensemble_regs, **args)
        print('SMOKE TEST PASSED')
    else:
        results, features = main(ensemble_list=ensemble_regs, **args)
        args['features'] = features
        with open('./out/RF_ERT/{}_dimensionality.pickle'.format(args['type']), 'wb') as file:
            pickle.dump(args, file)
            pickle.dump(results, file)
