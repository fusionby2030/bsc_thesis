import abc
import logging

import torch 
import torch.nn as nn 
import numpy as np


class BaseTorchModel(nn.Module):
    """
    NOTE: THIS IS NOT TO BE USED ON ITS OWN UNLESS YOU KNOW WHAT YOU ARE DOING
    use the classes from torch_ensembles

    This is the base Feed Forward Torch Model, and can accept any other torch model.
    You can define your own model how you want with your specific hidden layers and such and inputs,
    but if you want to have your input and target layers to be data specific, then specify in your
    base_estimator model that you pass to the ensemble with
    input_size = estimator_args['input_size']
    target_size = estimator_args['target_size']
    and have the __init__ of your model take kwargs **estimator_args

    Example for a 1 hidden layer FFNN with hidden size 30:

        class PedFFNN(nn.Module):
            def __init__(self, **estimator_args):
                hidden_size = 30
                target_size = estimator_args['target_size']
                input_size = estimator_args['input_size']

                self.layer_1 = torch.nn.Linear(input_size, hidden_size)
                self.layer_2 = torch.nn.Linear(hidden_size, hidden_size)
                self.out = torch.nn.Linear(hidden_size, target_size)

                self.act_func = torch.nn.ReLU()

            def forward(self, x):
                x = self.act_func(self.layer_1(x))
                x = self.act_func(self.layer_2(x))
                out = self.out(x)
                return out

    One can even try out adding hidden size as a kwarg, and replace hidden_size =30 with hidden_size = estimator_args['hidden_size']
    Moral of the story, is that one can define arbitrarily complex models given any number of kwargs and just pass them as they please at the initalization

    """
    def __init__(self, estimator, n_estimators, estimator_args=None):
        super(BaseTorchModel, self).__init__()
        self.base_estimator_ = estimator
        self.n_estimators = n_estimators
        self.estimator_args = estimator_args

        self.estimators_ = nn.ModuleList()
        self.logger = logging.getLogger()
        self.use_scheduler_ = False

    def __len__(self):
        """
        Return the number of base estimators in the ensemble
        """
        return len(self.estimators_)

    def __getitem__(self, index):
        """
        Return the index -th base estimator in the ensemble
        """
        return self.estimators_[index]

    def _decide_n_outputs(self, train_loader):
        """
        Clever way of getting the input and output dimensions to be set later
        """
        for _, (control, target) in enumerate(train_loader):
            if len(target.size()) == 1:
                n_outputs = 1
            else:
                n_outputs = target.size(1)

            if len(control.size()) == 1:
                n_inputs = 1
            else:
                n_inputs = control.size(1)
            break
        self.n_outputs, self.n_inputs = n_outputs, n_inputs
        return n_outputs, n_inputs

    def _make_estimator(self):
        """
        Configure a copy of the self.base_estimator_
        """

        if self.estimator_args is None:
            estimator_args = {'target_size': self.n_outputs, 'input_size': self.n_inputs}
            estimator = self.base_estimator_(**estimator_args)
        else:
            estimator = self.base_estimator_(**self.estimator_args)

        return estimator

    @abc.abstractmethod
    def set_optimizer(self, optimizer_name, **kwargs):
        """
        Setting the optimizer
        """

    @abc.abstractmethod
    def set_scheduler(self, scheduler_name, **kwargs):
        """
        Set the learning rate scheduler
        """

    @abc.abstractmethod
    def forward(self, x):
        """
        Forward pass of the ensemble
        """

    @abc.abstractmethod
    def fit(self, train_loader, epochs=150, log_interval=25, test_loader=None, save_model=True, save_dir=None):
        """
        training stage of the ensemble
        """

    @torch.no_grad()
    def predict(self, X, return_numpy=False, return_std=False):
        self.eval()
        pred = None

        if isinstance(X, torch.Tensor):
            pred = self.forward(X, return_std)

        elif isinstance(X, np.ndarray):
            X = torch.Tensor(X)
            pred = self.forward(X, return_std)

        else:
            msg = ('The type of input to ensemble should be a torch.tensor or np.ndarray')
            raise ValueError(msg)


        return pred




class BaseTorchRegressor(BaseTorchModel):
    """
    Base class for all ensemble regressors
    """

    @torch.no_grad()
    def evaluate(self, test_loader):
        self.eval()
        mse = 0.0
        criterion = nn.MSELoss()

        for _, (control, target) in enumerate(test_loader):
            output = self.forward(control)
            mse += criterion(output, target)
        return float(mse) / len(test_loader)









