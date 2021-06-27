"""
EXP for RFS and ERTs to see how they perform with different number of feature sampling

pickle file contains
    - args : - min ests, max_ests, ERT vs RF
    - results: OOB list for each number estimators in range (min_est, max_est) for each feature sampled (1-> 12)

To do some plotting
for label, clf_err in results.items():
    xs, ys = zip(*clf_err)

you may get some warnings from sklearn. Sorry
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
    OOB_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    for label, reg in ensemble_list:
        iterator = tqdm(range(kwargs['min_estimators'], kwargs['max_estimators'] + 1), position=0, leave=True, desc=label)
        for i in iterator:
            reg.set_params(n_estimators=i)
            reg.fit(feature_space, y)
            oob_error = 1 - reg.oob_score_
            OOB_rate[label].append(oob_error)

    return OOB_rate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-type", help='RF for RFs and ERT for ERTs', type=str, choices=['ERT', 'RF'], default='RF')
    parser.add_argument('-max_estimators', help='max number of decision trees in ensemble to test, i.e., end size', type=int, default=394)
    parser.add_argument('-min_estimators', help='min number of decision trees in ensemble to test, i.e., start size', type=int, default=20)
    parser.add_argument('--smoke_test', help='Smoke Test, quickly check if it works', action="count", default=0)
    args_namespace = parser.parse_args()
    args = vars(args_namespace)

    ensemble_clfs = [("max_features={}".format(i), RandomForestRegressor(warm_start=True, oob_score=True, max_features=i, random_state=42)) for i in range(2, 13)]
    ensemble_clfs_ET = [("max_features={}".format(i), ExtraTreesRegressor(warm_start=True, bootstrap=True, oob_score=True, max_features=i, random_state=42)) for i in range(2, 13)]
    
    if args['type'] == 'ERT':
        ensemble_regs = ensemble_clfs_ET
    else:
        ensemble_regs = ensemble_clfs
    np.random.seed(42)
    # args = {'max_estimators': 350, 'min_estimators': 15, 'estimator': 'ERT'}

    if args['smoke_test']:
        args['min_estimators'] = 15
        args['max_estimators'] = 25
        args['type'] = 'RF'
        OOB_results = main(ensemble_list=ensemble_regs, **args)
        print('SMOKE TEST PASSED')
    else:
        OOB_results = main(ensemble_list=ensemble_regs, **args)
        with open('./out/RF_ERT/oob_exp_{}.pickle'.format(args['type']), 'wb') as file:
            pickle.dump(args, file)
            pickle.dump(OOB_results, file)
