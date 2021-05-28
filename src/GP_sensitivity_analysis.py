from codebase.GPs import sensitivity_analysis
from codebase.data import utils
import pickle

from sklearn.preprocessing import StandardScaler


def main():
    control_space, target_space, _, _ = utils.process_data(return_numpy=True)
    feature_scaler = StandardScaler()
    x = feature_scaler.fit_transform(control_space)
    k_ARD, k_KL, k_VAR = sensitivity_analysis.kernel_relevance(x, target_space)
    return {'ARD': k_ARD, 'KLD': k_KL, 'VAR': k_VAR}


if __name__ == '__main__':
    results_dict = main()
    with open('./out/GP/sensitivity_analysis_RBF.pickle', 'wb') as file:
        pickle.dump(results_dict, file)
