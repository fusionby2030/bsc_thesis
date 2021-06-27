"""
output of this file is stored in out/GP/sensitivity_analysis_RBF.pickle
pickle contains in the following order
    1) list of features
    2) dictionary of rankings that follow above list that are self titled, with average being the mean of them all
    3) dictionary of arguments for experiment


to sort the dictionaries from lowest relevant to highest relevant:

k1_ard is list stored in dictionary of rankings, i.e., 2) -> dictionary_2['ARD']
feature_names is list found in 1)

k1_ARD_dict = {key: value for key, value in zip(feature_names, k1_ARD)}
    -> combines feature names and ARD (list of feature names in order of what ARD list is)
sorted(k1_VAR_dict, key=k1_VAR_dict.get)
    -> outputs list in order

rest is up to you but check out the list reversal shit in GP_dimensionality!


"""

from codebase.GPs import sensitivity_analysis
from codebase.data import utils
import pickle

from sklearn.preprocessing import StandardScaler
import numpy as np


def main(numerical_cols=None, **kwargs):
    control_space, target_space, _, _ = utils.process_data(numerical_cols=numerical_cols, return_numpy=True)
    feature_scaler = StandardScaler()
    x = feature_scaler.fit_transform(control_space)
    k_ARD, k_KL, k_VAR = sensitivity_analysis.kernel_relevance(x, target_space, **kwargs)
    avg = np.mean(np.array([k_ARD, k_KL, k_VAR]), axis=0)
    return {'ARD': k_ARD, 'KLD': k_KL, 'VAR': k_VAR, 'Average': avg}


if __name__ == '__main__':

    args = {'kernel': None, "repeats": 5, 'm': 12, 'delta': 0.001, 'nquadr': 11}
    numerical_cols = ['averagetriangularity', 'a(m)', 'Ip(MA)',
                      'plasmavolume(m3)', 'q95', 'gasflowrateofmainspecies1022(es)', 'B(T)',
                      'P_ICRH(MW)', 'P_NBI(MW)', 'P_TOTPNBIPohmPICRHPshi(MW)',
                      'H(HD)', 'Meff']
    results_dict = main(numerical_cols=numerical_cols, **args)

    with open('./out/GP/sensitivity_analysis_RBF.pickle', 'wb') as file:
        pickle.dump(numerical_cols, file)
        pickle.dump(results_dict, file)
        pickle.dump(args, file)
