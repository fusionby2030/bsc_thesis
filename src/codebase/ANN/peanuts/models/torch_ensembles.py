import torch
import torch.nn as nn

from ._base import BaseTorchRegressor
from .utils import set_module_torch
from .utils import save_load_torch


class AverageTorchRegressor(BaseTorchRegressor):
    """
      In fusion-based ensemble, predictions from all base estimators are
      first aggregated as an average output. After then, the training loss is
      computed based on this average output and the ground-truth. The training
      loss is then back-propagated to all base estimators simultaneously.
    """

    def forward(self, x, return_std=False):
        # average the outputs
        outputs = [estimator(x) for estimator in self.estimators_]
        pred = sum(outputs) / len(outputs)
        if return_std:
            return pred, torch.std(torch.stack(outputs), dim=0).numpy()
        return pred

    def set_optimizer(self, optimizer_name, **kwargs):
        self.optimizer_name = optimizer_name
        self.optimizer_args = kwargs


    def set_scheduler(self, scheduler_name, **kwargs):
        self.scheduler_name = scheduler_name
        self.scheduler_args = kwargs
        self.use_scheduler_ = True

    def fit(self, train_loader, epochs=150, log_interval=25, test_loader=None, save_model=True, save_dir=None, nni=False):
        # Create the base estimator and set attributes
        self.n_outputs, self.n_inputs = self._decide_n_outputs(train_loader)

        for _ in range(self.n_estimators):
            self.estimators_.append(self._make_estimator())


        optimizer = set_module_torch.set_optimizer(self, self.optimizer_name, **self.optimizer_args)

        if self.use_scheduler_:
            self.scheduler_ = set_module_torch.set_scheduler(optimizer, self.scheduler_name, **self.scheduler_args)

        criterion = nn.MSELoss()

        best_mse = float("inf")
        total_iters = 0

        for epoch in range(epochs):
            self.train()
            for batch_idx, (control, target) in enumerate(train_loader):

                # control, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.forward(control)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        msg = "Epoch: {:03d} | Loss {:.5f}"
                        self.logger.info(msg.format(epoch, loss))

                total_iters += 1

            if test_loader:
                self.eval()
                with torch.no_grad():
                    mse = 0.0
                    for _, (control, target) in enumerate(test_loader):
                        output = self.forward(control)
                        mse += criterion(output, target)

                    mse /= len(test_loader)

                    if mse < best_mse:
                        best_mse = mse
                        if save_model:
                            save_load_torch.save(self, save_dir, self.logger)

                    msg = (
                        "Epoch: {:03d} | Validation MSE: {:.5f} |"
                        " Historical Best: {:.5f}"
                    )
                    self.logger.info(msg.format(epoch, mse, best_mse))

            if hasattr(self, "scheduler_"):
                self.scheduler_.step()


    def predict(self, X, return_numpy=False, return_std=False):
        return super().predict(X, return_numpy, return_std)