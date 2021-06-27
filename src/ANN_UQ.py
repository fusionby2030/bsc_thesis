"""
This runs the experiment for testing out a ANN and recieving the predictions and uncert. in preds from said ANN.

All arguments in the args below are variable, i.e., you can change them by passing in arguemtns via CLI
Accesible via:
    $ python3 ANN_UQ.py -lr 0.001 -batch_size 350

to run the smoke test:
    $ python3 ANN_UQ --smoke_test

If you run this, below are two files that will be created (this will also overwrite the existing experiment)

- pickle file with the following dictionaries:
    - args = {'batch_size': 396, 'lr': 0.004, 'n_splits': 5, 'n_repeats': 5, 'n_estimators': 15,
            'scalex':StandardScaler, 'scaley':None , 'scale_args':(None, None),
            'hidden_layer_sizes': [483, 415, 254]}
    - results = {'y_vals': np.array(list_yvals), 'preds': np.array(list_preds), 'uncert': np.array(list_uncert),
               'score': scores}

- a temporary model state dict will be saved wherever you run this program from!

Results file is stored in same directory
To read the pickle file:
    with open(filename, 'rb') as file:
        args = pickle.load(file)
        results = pickle.load(file)
"""
from codebase.data import utils
from codebase.ANN.peanuts.models.utils import save_load_torch
from codebase.ANN.peanuts.models.torch_ensembles import AverageTorchRegressor

import numpy as np
import torch.nn as nn
import torch

from sklearn.preprocessing import StandardScaler

import pickle
import argparse



class ANNtorchdataset(torch.utils.data.Dataset):
    def __init__(self, controls, targets, scaley=None, scalex=None, train_scalers=None, scale_args=None):
        if train_scalers is None:
            if scalex is not None:
                if scale_args[0] is not None:
                    self.scalex = scalex(**scale_args[0])
                else:
                    self.scalex = scalex()
                controls = self.scalex.fit_transform(controls)
                controls = torch.tensor(controls).float()
            if scaley is not None:
                if scale_args[1] is not None:
                    self.scaley = scaley(**scale_args[1])
                else:
                    self.scaley = scaley()
                targets = self.scaley.fit_transform(targets)
                targets = torch.tensor(targets).float()
        else:
            self.scalex, self.scaley = train_scalers
            if self.scalex is not None:
                controls = self.scalex.transform(controls)
            if self.scaley is not None:
                targets = self.scaley.transform(targets)
            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets).float()
            if not isinstance(controls, torch.Tensor):
                controls = torch.tensor(controls).float()
        self.inputs = controls
        self.outputs = targets

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, item):
        return self.inputs[item], self.outputs[item]

    def return_scalars(self):
        try:
            return self.scalex, self.scaley
        except AttributeError:
            try:
                return self.scalex, None
            except AttributeError:
                return None



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
from collections import OrderedDict
from tqdm import tqdm

def cross_validate_torch(control_tensor, target_tensor, **kwargs):
    control_tensor = control_tensor.float()
    target_tensor = target_tensor.float()
    ordered_preds = OrderedDict((index, []) for index in range(0, len(control_tensor)))
    ordered_error = OrderedDict((index, []) for index in range(0, len(control_tensor)))
    ordered_yvals = OrderedDict((index, []) for index in range(0, len(control_tensor)))
    error_dict = {'MAE': [], 'RMSE': []}
    cv = RepeatedKFold(n_splits=kwargs['n_splits'], n_repeats=kwargs['n_repeats'])
    iterator = tqdm(enumerate(cv.split(control_tensor)))
    for cv_id, (train, test) in iterator:
        train_dataset = ANNtorchdataset(control_tensor[train], target_tensor[train], scaley=kwargs['scaley'], scalex=kwargs['scalex'], scale_args=kwargs['scale_args'])
        test_dataset = ANNtorchdataset(control_tensor[test], target_tensor[test], train_scalers=train_dataset.return_scalars())

        train_loader = torch.utils.data.DataLoader(train_dataset, kwargs['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, kwargs['batch_size'], shuffle=True)

        # model = PedFFNN(hidden_layer_sizes=kwargs['hidden_layer_sizes'])
        base_model = PedFFNN
        model = AverageTorchRegressor(estimator=base_model, n_estimators=kwargs['n_estimators'],
                                      estimator_args={'hidden_layer_sizes': kwargs['hidden_layer_sizes']}).double()
        # optimizer = set_module_torch.set_optimizer(model, 'Adam', lr=kwargs['lr'])
        model.set_optimizer('Adam', lr=kwargs['lr'])

        # best_MSE =  np.inf

        model.fit(train_loader=train_loader, test_loader=test_loader, epochs=200)

        model = AverageTorchRegressor(estimator=base_model, n_estimators=kwargs['n_estimators'],
                                      estimator_args={'hidden_layer_sizes': kwargs['hidden_layer_sizes']})

        model._decide_n_outputs(test_loader)
        save_load_torch.load(model)
        predictions, std_preds = model.predict(control_tensor[test], return_std=True)
        if kwargs['scaley'] is not None:
            predictions = test_dataset.scaley.inverse_transform(predictions)

        # std_preds = predictions

        for index, std, mean_pred, yval in zip(test, std_preds, predictions, target_tensor[test]):
            ordered_error[index].append(std.item())
            ordered_preds[index].append(mean_pred.item())
            ordered_yvals[index].append(yval.item())

        RMSE = mean_squared_error(target_tensor[test], predictions, squared=False)
        MAE = mean_absolute_error(target_tensor[test], predictions)
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


def main(control_tensor, target_tensor, **kwargs):
    results = cross_validate_torch(control_tensor, target_tensor, torch_model=PedFFNN, **kwargs)

    return results

def generate_default_params():
    args = {'batch_size': 396, 'lr': 0.004, 'n_splits': 5, 'n_repeats': 5, 'n_estimators': 15,
            'scalex':StandardScaler, 'scaley':None , 'scale_args':(None, None),
            'hidden_layer_sizes': [483, 415, 254]
            }
    return args



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", help='batch size during training/validation', type=int, default=396)
    parser.add_argument('-lr', help='learning rate', type=float, default=0.004)
    parser.add_argument('-n_splits', help='number of folds in CV', type=int, default=5)
    parser.add_argument('-n_repeats', help='number of repeats of CV', type=int, default=5)
    parser.add_argument('-n_estimators', help='Number of ANNs in ensemble, 1 is default ANN',type=int, default=15)
    parser.add_argument('-hidden_layer_sizes', help='Sizes of hidden layers in ANN', type=list, default=[483, 415, 254])
    parser.add_argument('--smoke_test', help='Smoke Test, quickly check if it works', action="count", default=0)
    args_namespace = parser.parse_args()
    args = vars(args_namespace)
    # make data and tensors

    torch.manual_seed(42)
    np.random.seed(42)
    control_space, target_space, _, _ = utils.process_data()
    control_tensors, target_tensors = utils.setup_tensors(control_space, target_space)
    # TODO: argparse - need for batch size, learning rate,
    try:
        if args['smoke_test'] >= 1:
            smoke_args = {'n_estimators': 2, 'hidden_layer_sizes': [10, 10], 'n_splits': 2, 'n_repeats': 2}
            params = generate_default_params()
            params.update(smoke_args)
            results = main(control_tensors, target_tensors, **params)
            print('PASSED SMOKE TEST')
        else:
            params = generate_default_params()
            params.update(args)
            results = main(control_tensors, target_tensors, **params)
            with open('{}_layer_UQ_{}_est.pickle'.format(len(params['hidden_layer_sizes']), params['n_estimators']), 'wb') as file:
                pickle.dump(params, file)
                pickle.dump(results, file)
    except Exception as exc:
        raise exc
