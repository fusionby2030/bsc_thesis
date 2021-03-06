"""
This runs the experiment for comparing size & depth vs performance of ANNs.

All arguments in the args below are variable, i.e., you can change them by passing in arguemtns via CLI
Accesible via:
    $ python3 ANN_UQ.py -lr 0.001 -batch_size 350

The architectures tested are found in main(), whereas the arguments are found in the if __name__ == '__main__':
If you run this, below are two files that will be created (this will also overwrite the existing experiment)

- a pickle file will be saved in the src/out/ANN directory, with args, and RMSE + MAE for each size of model tested
    - stored in dicts of format {'[10]': (RMSE_val, RMSE_std), '[10, 10]': (RMSE_val, RMSE_std)}}
        - [10] -> one hidden layer of 10 neuron width, [10, 10] -> 2 hidden layers, 10 neurons each
- a temporary model state dict will be saved wherever you run this program from!


To read the pickle file back and use it in some plotting see below:
    with open(filename, 'rb') as file:
        args = pickle.load(file)
        RMSE_dict = pickle.load(file)
        MAE_dict = pickle.load(file)

"""
from codebase.data import utils
from codebase.ANN.peanuts.models.utils import set_module_torch, save_load_torch
from codebase.ANN.peanuts.models.torch_ensembles import AverageTorchRegressor

import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm

import argparse
from itertools import chain

class PedFFNN(nn.Module):
    def __init__(self, **kwargs):
        super(PedFFNN, self).__init__()

        target_size = 1
        input_size = 12
        act_func = torch.nn.ELU()

        last_size = input_size

        self.hidden_layers = torch.nn.ModuleList()
        hidden_layer_sizes = kwargs['hidden_layer_sizes']

        for size in hidden_layer_sizes:
            self.hidden_layers.append(self._fc_block(last_size, size, act_func))
            last_size = size

        self.out = torch.nn.Linear(last_size, target_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)

        x = self.out(x)
        return x

    @staticmethod
    def _fc_block(in_c, out_c, act_func):
        block = torch.nn.Sequential(
            torch.nn.Linear(in_c, out_c),
            act_func
        )
        return block

    def predict(self, X):
        self.eval()
        pred = None

        if isinstance(X, torch.Tensor):
            pred = self.forward(X)

        elif isinstance(X, np.ndarray):
            X = torch.Tensor(X)
            pred = self.forward(X)

        else:
            msg = 'The type of input to ensemble should be a torch.tensor or np.ndarray'
            raise ValueError(msg)

        return pred


from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error


def cross_validate_torch(control_tensor, target_tensor, **kwargs):
    RMSE_list = []
    MAE_list = []
    cv = RepeatedKFold(n_splits=kwargs['n_splits'], n_repeats=kwargs['n_repeats'])

    for cv_id, (train, test) in enumerate(cv.split(control_tensor)):
        train_dataset = utils.ANNtorchdataset(control_tensor[train], target_tensor[train])
        test_dataset = utils.ANNtorchdataset(control_tensor[test], target_tensor[test])

        train_loader = torch.utils.data.DataLoader(train_dataset, kwargs['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, kwargs['batch_size'], shuffle=True)

        # model = PedFFNN(hidden_layer_sizes=kwargs['hidden_layer_sizes'])
        base_model = PedFFNN
        model = AverageTorchRegressor(estimator=base_model, n_estimators=kwargs['n_estimators'], estimator_args={'hidden_layer_sizes': kwargs['hidden_layer_sizes']})
        # optimizer = set_module_torch.set_optimizer(model, 'Adam', lr=kwargs['lr'])
        model.set_optimizer('Adam', lr=kwargs['lr'])
        criterion = nn.MSELoss()

        # best_MSE =  np.inf

        model.fit(train_loader=train_loader, test_loader=test_loader, epochs=200)

        model = AverageTorchRegressor(estimator=base_model, n_estimators=kwargs['n_estimators'], estimator_args={'hidden_layer_sizes': kwargs['hidden_layer_sizes']})
        model._decide_n_outputs(test_loader)

        save_load_torch.load(model)
        predictions = model.predict(control_tensor[test])
        RMSE_list.append(mean_squared_error(target_tensor[test], predictions, squared=False))
        MAE_list.append(mean_absolute_error(target_tensor[test], predictions))
        # print(cv_id)
    avg_RMSE, avg_MAE = np.mean(RMSE_list), np.mean(MAE_list)
    std_RMSE, std_MAE = np.std(RMSE_list), np.std(MAE_list)

    return (avg_RMSE, std_RMSE), (avg_MAE, std_MAE)


def main(control_tensor, target_tensor, **kwargs):

    # make a list hidden layer sizes lists to be tested
    # run them through CV
    exp_list_smoke_test = [[25],
                           [25, 25],
                           [25, 25, 25],
                           [25, 25, 25, 25]]

    exp_long = [
        [[i] for i in range(10, 500, 25)],
        [[i, i] for i in range(25, 1000, 50)],
        [[i, i, i] for i in range(25, 1000, 50)],
        [[i, i, i, i] for i in range(25, 1000, 50)],
        [[i, i, i, i, i] for i in range(25, 1000, 50)],
        [[i, i, i, i, i, i] for i in range(25, 1000, 50)]]

    exp_long = list(chain.from_iterable(exp_long))

    RMSE_dict = {}
    MAE_dict = {}
    iterator = tqdm(exp_long, position=0, leave=False)
    for exp in iterator:
        # print('Hidden Sizes: ', exp)
        iterator.set_description(str(exp))
        RMSE_info, MAE_info = cross_validate_torch(control_tensor, target_tensor, torch_model=PedFFNN, **kwargs, hidden_layer_sizes=exp)
        RMSE_dict[str(exp)] = RMSE_info
        MAE_dict[str(exp)] = MAE_info
        # print('RMSE {:.4}, +- {:.4}'.format(RMSE_info[0], RMSE_info[1]))
        # print('\n')
        iterator.set_postfix(RMSE=RMSE_info[0])
    return RMSE_dict, MAE_dict


import pickle
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", help='batch size during training/validation', type=int, default=396)
    parser.add_argument('-lr', help='learning rate', type=float, default=0.004)
    parser.add_argument('-n_splits', help='number of folds in CV', type=int, default=5)
    parser.add_argument('-n_repeats', help='number of repeats of CV', type=int, default=2)
    parser.add_argument('-n_estimators', help='Number of ANNs in ensemble, 1 is default ANN',type=int, default=1)

    args_namespace = parser.parse_args()
    args = vars(args_namespace)
    # make data and tensors
    control_space, target_space, _, _ = utils.process_data()
    control_tensors, target_tensors = utils.setup_tensors(control_space, target_space)

    # pass to main
    # relevant ANN kwargs get passed as kwargs
    # TODO: argparse - need for batch size, learning rate,
    # args = {'batch_size': 396, 'lr': 0.004, 'n_splits': 5, 'n_repeats': 2, 'n_estimators': 1}
    RMSE_dict, MAE_dict = main(control_tensors, target_tensors, **args)

    with open('ANN_performance_vs_size_exp.pickle', 'wb') as file:
        pickle.dump(args, file)
        pickle.dump(RMSE_dict, file)
        pickle.dump(MAE_dict, file)
