import pandas as pd
import sklearn as sk
import torch


class ANNtorchdataset(torch.utils.data.Dataset):
    def __init__(self, controls, targets):
        self.inputs = controls
        self.outputs = targets

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, item):
        return self.inputs[item], self.outputs[item]


def plot_predictions(predictions, true_vals, color='orange', save_fig=False, save_dir='..../out/STANDALONE'):
    import matplotlib.pyplot as plt
    SMALL_SIZE = 22
    MEDIUM_SIZE = 30

    plt.rc('font', size=MEDIUM_SIZE, weight='bold')
    plt.rc('axes', titlesize=MEDIUM_SIZE, labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=MEDIUM_SIZE)
    fig = plt.figure(figsize=(18, 18))

    plt.scatter(true_vals, predictions, label='Predictions', c=color, edgecolors=(0, 0, 0))
    plt.plot([min(true_vals), max(true_vals)], [min(true_vals), max(true_vals)], 'r--')
    plt.xlabel('Real $n_e^{ped}$')
    plt.ylabel('Predicted $n_e^{ped}$')
    if save_fig and save_dir is not None:
        plt.savefig(save_dir)
    plt.show()


def process_data(target_col=None, numerical_cols=None, flags=None, return_necessary=True, return_numpy=False, file_loc='/home/adam/EDGE/2021/thesis/filtered_dataset.csv', filter_neped=None, filter_control = None):
    """
    filter_neped -> float, anything above it will be dropped
    filter_control -> (column, value, above/below) (string, float/double, 'above' or 'below')
    - return_necessary: (feature_df, target_df, feature_err_df, target_err_df)
    - return necessary + return numpy -> above but with numpy arrays
    - return necessary = False -> feature_space, target_space, df, joint_df, target_error_df, feature_error_df
    """

    if target_col is None:
        target_col = ['nepedheight1019(m3)']

    if numerical_cols is None:
        numerical_cols = ['averagetriangularity', 'a(m)', 'Ip(MA)', 'P_NBI(MW)',
                          'plasmavolume(m3)', 'P_TOTPNBIPohmPICRHPshi(MW)', 'q95',
                          'gasflowrateofmainspecies1022(es)', 'B(T)', 'P_ICRH(MW)', 'H(HD)',
                          'Meff']

    df = pd.read_csv(file_loc)
    # df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # just mix everything up at the start
    if filter_neped is not None:
        new_df = df[df['nepedheight1019(m3)'] < filter_neped]
        df = new_df.copy()
    if filter_control is not None:
        col, val, signal = filter_control
        if signal == 'above':
            new_df = df[df[col] > val]
        else:
            new_df = df[df[col] < val]
        df = new_df.copy()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if flags is not None:
        feature_space = df[numerical_cols + flags]
    else:
        feature_space = df[numerical_cols]

    target_space = df[target_col]
    joint_df = df[numerical_cols + target_col]
    target_error_df = df[['error_' + col for col in target_col]]
    feature_error_df = df[['error_' + col for col in numerical_cols]]

    if return_necessary:
        if return_numpy:
            return feature_space.to_numpy(), target_space.to_numpy(), feature_error_df.to_numpy(), target_error_df.to_numpy()
        return feature_space, target_space, feature_error_df, target_error_df
    return feature_space, target_space, df, joint_df, target_error_df, feature_error_df


def setup_tensors(features: pd.DataFrame, targets: pd.DataFrame, feat_err=None, tar_err=None) -> (torch.FloatTensor, torch.FloatTensor):
    # TODO: Error returns
    feature_scale = sk.preprocessing.StandardScaler() # scaling of features so that they all have mean of 0 and std 1
    feature_trans = feature_scale.fit_transform(features)
    feature_tensor = torch.tensor(feature_trans).float()
    target_tensor = torch.tensor(targets.values).float()

    return feature_tensor, target_tensor
