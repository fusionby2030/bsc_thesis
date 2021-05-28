"""

PEDFFNN shit



    Setup Experiment
        - set BS, LR (static)
        - Variable hidden layer sizes, e.g., [100, 50] is two hidden layers with first layer width 100, second layer width 50
        - list of lists ([10], [50, 10], [150, 50, 10]], first experiment is 1 hidden layer
    CV Evaluation
        - takes in model, and tensors
        - 5 splits, 5 repeats
            - Calls fitting procedure from codebase
                - Fits over epochs, saving best performing time step
            - Loads best model -> make predictions across left out set
            - Save predictions in some ordered dict (see RF experiment)
        - Average all predictions (and uncerts?)
        Save predictions & Uncerts for each model in ordered dict with name of the hidden layer sizes

INTO SCRIPT

Testing
    - Same model should return same RMSE
"""
from codebase.data import utils
from codebase.ANN.peanuts.models.utils import set_module_torch


import numpy as np
import torch.nn as nn 


class PedFFNN(nn.Module):
    def __init__(self, **kwargs):
        super(PedFFNN_params, self).__init__()

        target_size = 12
        input_size = 1
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

    def _fc_block(self, in_c, out_c, act_func):
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
            msg = ('The type of input to ensemble should be a torch.tensor or np.ndarray')
            raise ValueError(msg)


        return pred


from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

def cross_validate_torch(control_tensor, target_tensor, n_splits=5, n_repeats=5, **kwargs):
    RMSE_list = []
    MAE_list = []
    cv = RepeatedKFold(n_splits=5, n_repeats=5)
    
    for cv_id, train, test in enumerate(cv.split(control_tensor)):
        train_dataset = utils.ANNtorchdataset(control_tensor[train], target_tensor[train])
        test_dataset = utils.ANNtorchdataset(control_tensor[test], target_tensor[test])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, kwargs['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, kwargs['batch_size'], shuffle=True)

        model = PedFFNN(hidden_layer_sizes = kwargs['hidden_layer_sizes'])
        
        optimizer = set_module_torch.set_optimizer(model, 'Adam', lr=kwargs['lr'])
        criterion = nn.MSELoss()
        
        best_MSE =  np.inf
        for _ in range(200):

            model.train()
            for batch_idx, (control, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model.forward(control)
                loss = criterion(output, target)

                loss.backward()
                optimizer.step()
            
            model.eval()
            mse = 0.0
            for test_batch_idx, (control, target) in enumerate(test_loader):
                output = model.forward(control)
                mse += criterion(output, target).detach().numpy()

            mse /= len(test_loader)
            
            # Very modest early stopping method, i.e., save when the best validation MSE occurs 
            if mse < best_MSE:
                best_MSE = mse 
                # save a general state dict of the model 
                torch.save(model.save_dict(), 'temporary_model.pth')
        # Make predictions from best point of model (early stopped)
        best_model = PedFFNN(hidden_layer_sizes = kwargs['hidden_layer_sizes'])
        best_model.load_state_dict(torch.load('temporary_model.pth'))
        best_model.eval()
        
        predictions = best_model.predict(control_tensor[test])
        RMSE_list.append(mean_squared_error(target_tensor[test], predictions, squared=False))
        MAE_list.append(mean_absolute_error(target_tensor[test], predictions))
    
    avg_RMSE, avg_MAE = np.mean(RMSE_list), np.mean(MAE_list)
    std_RMSE, std_MAE = np.std(RMSE_list), np.std(MAE_list)
    
    return (avg_RMSE, std_RMSE), (avg_MAE, std_MAE)

        

    

def main(control_tensor, target_tensor, **kwargs):

    # make a list hidden layer sizes lists to be tested 
    # run them through CV 

    exp_list_smoke_test = [[25], [100], [25, 25], [50, 25], [100, 25], [100, 100], [100, 100, 100]]
    RMSE_dict = {}
    MAE_dict = {}
    for exp in exp_list_smoke_test:
        RMSE_info, MAE_info = cross_validate_torch(control_tensor, target_tensor, kwargs, hidden_layer_sizes = exp)
        RMSE_dict[str(exp)] = RMSE_info
        MAE_dict[str(exp)] = MAE_info
    
    return RMSE_dict, MAE_dict



if __name__ == '__main__':
    # make data and tensors
    control_space, target_space, _, _ = utils.process_data()
    control_tensors, target_tensors = utils.setup_tensors(control_space, target_space)

    # pass to main
    # relevant ANN kwargs get passed as kwargs

    main(control_tensors, target_tensors, batch_size=396, lr=0.0013)

    # TODO: argparse - need for batch size, learning rate, 
