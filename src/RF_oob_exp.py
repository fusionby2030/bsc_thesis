from codebase.data import utils
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from collections import OrderedDict
from tqdm import tqdm
import pickle


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
    ensemble_clfs = [
        ("max_features=2", RandomForestRegressor(warm_start=True, oob_score=True, max_features=2, random_state=42)),
        ("max_features=3", RandomForestRegressor(warm_start=True, oob_score=True, max_features=3, random_state=42)),
        ("max_features=4", RandomForestRegressor(warm_start=True, oob_score=True, max_features=4, random_state=42)),
        ("max_features=5", RandomForestRegressor(warm_start=True, oob_score=True, max_features=5, random_state=42)),
        ("max_features=6", RandomForestRegressor(warm_start=True, oob_score=True, max_features=6, random_state=42)),
        ("max_features=7", RandomForestRegressor(warm_start=True, oob_score=True, max_features=7, random_state=42)),
        ("max_features=8", RandomForestRegressor(warm_start=True, oob_score=True, max_features=8, random_state=42)),
        ("max_features=9", RandomForestRegressor(warm_start=True, oob_score=True, max_features=9, random_state=42)),
        ("max_features=10", RandomForestRegressor(warm_start=True, oob_score=True, max_features=10, random_state=42)),
        ("max_features=11", RandomForestRegressor(warm_start=True, oob_score=True, max_features=11, random_state=42)),
        (
        "max_features=all", RandomForestRegressor(warm_start=True, oob_score=True, max_features=None, random_state=42))]

    ensemble_clfs_ET = [
        ("max_features=2",
         ExtraTreesRegressor(warm_start=True, bootstrap=True, oob_score=True, max_features=2, random_state=42)),
        ("max_features=3",
         ExtraTreesRegressor(warm_start=True, bootstrap=True, oob_score=True, max_features=3, random_state=42)),
        ("max_features=4",
         ExtraTreesRegressor(warm_start=True, bootstrap=True, oob_score=True, max_features=4, random_state=42)),
        ("max_features=5",
         ExtraTreesRegressor(warm_start=True, bootstrap=True, oob_score=True, max_features=5, random_state=42)),
        ("max_features=6",
         ExtraTreesRegressor(warm_start=True, bootstrap=True, oob_score=True, max_features=6, random_state=42)),
        ("max_features=7",
         ExtraTreesRegressor(warm_start=True, bootstrap=True, oob_score=True, max_features=7, random_state=42)),
        ("max_features=8",
         ExtraTreesRegressor(warm_start=True, bootstrap=True, oob_score=True, max_features=8, random_state=42)),
        ("max_features=9",
         ExtraTreesRegressor(warm_start=True, bootstrap=True, oob_score=True, max_features=9, random_state=42)),
        ("max_features=10",
         ExtraTreesRegressor(warm_start=True, bootstrap=True, oob_score=True, max_features=10, random_state=42)),
        ("max_features=11",
         ExtraTreesRegressor(warm_start=True, bootstrap=True, oob_score=True, max_features=11, random_state=42)),
        ("max_features=all",
         ExtraTreesRegressor(warm_start=True, bootstrap=True, oob_score=True, max_features=None, random_state=42))
    ]

    np.random.seed(42)
    args = {'max_estimators': 350, 'min_estimators': 15, 'estimator': 'ERT'}
    OOB_results = main(ensemble_list=ensemble_clfs_ET, **args)

    with open('./out/RF_ERT/oob_exp_ERT.pickle', 'wb') as file:
        pickle.dump(args, file)
        pickle.dump(OOB_results, file)
