"""
This runs the experiment for comparing size & depth vs performance of ANNs.

The architectures tested are found in main(), whereas the arguments are found in the if __name__ == '__main__':
If you run this, below are two files that will be created (this will also overwrite the existing experiment)

- a pickle file will be saved in the src/out/ANN directory, with args, and RMSE + MAE for each size of model tested
    - stored in dicts of format {'[10]': (RMSE_val, RMSE_std), '[10, 10]': (RMSE_val, RMSE_std)}}
        - [10] -> one hidden layer of 10 neuron width, [10, 10] -> 2 hidden layers, 10 neurons each
- a temporary model state dict will be saved wherever you run this program from!

I recommend running it from the src/ directory, i.e., cd into src/ and run the program from there
To read the pickle file back and use it in some plotting see below:
    with open(filename, 'rb') as file:
        args = pickle.load(file)
        RMSE_dict = pickle.load(file)
        MAE_dict = pickle.load(file)


TODO:
argparse integration
some bash stuff to offload to triton

Testing
    - Same model should return same RMSE
"""
from codebase.data import utils
from codebase.ANN.peanuts.models.utils import save_load_torch
from codebase.ANN.peanuts.models.torch_ensembles import AverageTorchRegressor

import numpy as np
import torch.nn as nn
import torch

import nni

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


def cross_validate_torch(control_tensor, target_tensor, **kwargs):
    control_tensor = control_tensor.float()
    target_tensor = target_tensor.float()
    RMSE_list = []
    MAE_list = []

    cv = RepeatedKFold(n_splits=kwargs['n_splits'], n_repeats=kwargs['n_repeats'])

    for cv_id, (train, test) in enumerate(cv.split(control_tensor)):
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
        predictions = model.predict(control_tensor[test])
        if kwargs['scaley'] is not None:
            predictions = test_dataset.scaley.inverse_transform(predictions)

        RMSE_list.append(mean_squared_error(target_tensor[test], predictions, squared=False))
        MAE_list.append(mean_absolute_error(target_tensor[test], predictions))
        nni.report_intermediate_result({'default':-np.log10(RMSE_list[-1]), 'RMSE': RMSE_list[-1]})
        # print(cv_id)
    avg_RMSE, avg_MAE = np.mean(RMSE_list), np.mean(MAE_list)
    std_RMSE, std_MAE = np.std(RMSE_list), np.std(MAE_list)

    return (avg_RMSE, std_RMSE), (avg_MAE, std_MAE)


def main(control_tensor, target_tensor, **kwargs):
    kwargs['hidden_layer_sizes'] = [kwargs['hl_s1'], kwargs['hl_s2'], kwargs['hl_s3'], kwargs['hl_s4'], kwargs['hl_s5']]
    RMSE_info, MAE_info = cross_validate_torch(control_tensor, target_tensor, torch_model=PedFFNN, **kwargs)

    scores = {'RMSE': RMSE_info, 'MAE': MAE_info}
    nni.report_final_result({'default':-np.log10(RMSE_info[0]), 'RMSE':RMSE_info[0]})
    return scores

def generate_default_params():
    args = {'batch_size': 396, 'lr': 0.004, 'n_splits': 5, 'n_repeats': 2, 'n_estimators': 1,
            'scalex':StandardScaler, 'scaley':None , 'scale_args':(None, None),
            'hl_s1': 400,
            'hl_s2': 400,
            'hl_s3': 400,
            'hl_s4': 400,
            'hl_s5': 400,
            }
    return args


import pickle
from sklearn.preprocessing import PowerTransformer, StandardScaler, QuantileTransformer
if __name__ == '__main__':
    # make data and tensors

    torch.manual_seed(42)
    np.random.seed(42)
    control_space, target_space, _, _ = utils.process_data()
    control_tensors, target_tensors = utils.setup_tensors(control_space, target_space)
    # TODO: argparse - need for batch size, learning rate,
    try:
        updated_params = nni.get_next_parameter()
        params = generate_default_params()
        params.update(updated_params)
        results = main(control_tensors, target_tensors, **params)
    except Exception as exc:
        raise exc

    # print(results)
    """
    with open('./out/ANN/trial_performance_vs_size_exp.pickle', 'wb') as file:
        pickle.dump(args, file)
        pickle.dump(RMSE_dict, file)
        pickle.dump(MAE_dict, file)
    """
    """
    To read the pickle back: 
    with open(filename, 'rb') as file:
        args = pickle.load(file)
        RMSE_dict = pickle.load(file)
        MAE_dict = pickle.load(file)
    """
