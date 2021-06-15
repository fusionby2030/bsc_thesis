"""
TBD some functions that do the cross validate thing

If I have a
"""

import torch
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from peanuts.models.utils import set_module_torch
from peanuts.data.data_loading import ANNtorchdataset
import torch.nn as nn
import numpy as np


def cross_validate_torch(control_tensor, target_tensor, torch_model=None, **kwargs):
    RMSE_list = []
    MAE_list = []
    cv = RepeatedKFold(n_splits=kwargs['n_splits'], n_repeats=kwargs['n_repeats'])

    for cv_id, (train, test) in enumerate(cv.split(control_tensor)):
        train_dataset = ANNtorchdataset(control_tensor[train], target_tensor[train])
        test_dataset = ANNtorchdataset(control_tensor[test], target_tensor[test])

        train_loader = torch.utils.data.DataLoader(train_dataset, kwargs['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, kwargs['batch_size'], shuffle=True)
        if torch_model is None:
            from peanuts.models import PedFFNN
            model = PedFFNN
        else:
            model = torch_model(**kwargs)

        optimizer = set_module_torch.set_optimizer(model, 'Adam', lr=kwargs['lr'])
        criterion = nn.MSELoss()

        best_MSE = np.inf
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
                torch.save(model.state_dict(), 'temporary_model.pth')
        # Make predictions from best point of model (early stopped)
        best_model = torch_model(**kwargs)
        best_model.load_state_dict(torch.load('temporary_model.pth'))
        best_model.eval()
        with torch.no_grad():
            predictions = best_model.predict(control_tensor[test])
            RMSE_list.append(mean_squared_error(target_tensor[test], predictions, squared=False))
            MAE_list.append(mean_absolute_error(target_tensor[test], predictions))

    avg_RMSE, avg_MAE = np.mean(RMSE_list), np.mean(MAE_list)
    std_RMSE, std_MAE = np.std(RMSE_list), np.std(MAE_list)

    return (avg_RMSE, std_RMSE), (avg_MAE, std_MAE)
