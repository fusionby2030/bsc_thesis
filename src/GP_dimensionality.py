"""
This is some multiprocessing shenanigans!

at the end there should be 4 objects in a file stored in dictionary format idk why
"""
import numpy as np

import GPy
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp
import pickle

from codebase.data import utils

# bad idea but fuck it
import warnings

warnings.filterwarnings("ignore")


def kern_performance(controls, targets, kernel=None):
    cv = RepeatedKFold(n_splits=5, n_repeats=5)
    MAE = []
    RMSE = []

    for train, test in cv.split(controls):
        X_train, X_test = controls[train], controls[test]
        y_train, y_test = targets[train], targets[test]

        if kernel is None:
            kernel = GPy.kern.RBF(input_dim=controls.shape[0], ARD=True) + GPy.kern.Bias(input_dim=controls.shape[0])
        model = GPy.models.GPRegression(X_train, y_train, kernel)
        model.optimize()

        y_preds, y_var = model.predict(X_test)

        MAE.append(mean_absolute_error(y_pred=y_preds, y_true=y_test))
        RMSE.append(mean_squared_error(y_pred=y_preds, y_true=y_test, squared=False))

    MAE_avg = np.mean(MAE)
    RMSE_avg = np.mean(RMSE)

    return MAE_avg, RMSE_avg


def worker(kernel, x, y, q):
    RMSE_list = []
    MAE_list = []

    test_dims = np.arange(1, 13)
    for dim_in in test_dims:
        print(kernel, dim_in)
        if kernel == 'MLP':
            gpy_kern = GPy.kern.MLP(input_dim=dim_in, ARD=True) + GPy.kern.Bias(input_dim=dim_in)
        elif kernel == 'RBF':
            gpy_kern = GPy.kern.RBF(input_dim=dim_in, ARD=True) + GPy.kern.Bias(input_dim=dim_in)
        elif kernel == 'RatQuad':
            gpy_kern = GPy.kern.RatQuad(input_dim=dim_in, ARD=True) + GPy.kern.Bias(input_dim=dim_in)
        elif kernel == 'Mat32':
            gpy_kern = GPy.kern.Matern32(input_dim=dim_in, ARD=True) + GPy.kern.Bias(input_dim=dim_in)
        else:
            gpy_kern = None
        x_reduced = x[:, :dim_in]
        MA, RM = kern_performance(x_reduced, y, kernel=gpy_kern)
        RMSE_list.append(RM)
        MAE_list.append(MA)

    res = {'kernel': kernel, 'MAE': MAE_list, 'RMSE': RMSE_list}
    q.put(res)
    return res


def listener(q):
    # Listens for messages on the q, and writes to file

    with open('./out/GP/dimensionality.txt', 'w') as file:
        while 1:
            m = q.get()
            if m == 'kill':
                file.write('killed')
                break
            file.write(str(m) + '\n')
            file.flush()

results = {}
def collect_results(result):
    results[result[0]] = result[1]

def main():
    KL_cols = ['Meff', 'H(HD)', 'B(T)', 'P_ICRH(MW)', 'q95', 'P_TOTPNBIPohmPICRHPshi(MW)',
               'gasflowrateofmainspecies1022(es)', 'P_NBI(MW)', 'plasmavolume(m3)',
               'Ip(MA)', 'a(m)', 'averagetriangularity']
    KL_cols = list(reversed(KL_cols))  # if this isn't programming then I don't know what is

    feature_space, y, _, _ = utils.process_data(numerical_cols=KL_cols, return_numpy=True)

    scaler_feature = StandardScaler()
    x = scaler_feature.fit_transform(feature_space)

    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count() + 2)
    # noinspection PyUnusedLocal
    watcher = pool.apply_async(listener, (q,))

    jobs = []

    kernels = ['MLP', 'Mat32', 'RatQuad', 'RBF']
    for kern in kernels:
        job = pool.apply_async(worker, (kern, x, y, q))
        jobs.append(job)

    for job in jobs:
        job.get()

    # done now so kill all and close the damn pickle

    q.put('kill')
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
