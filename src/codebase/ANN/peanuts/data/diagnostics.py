from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import torch
import logging

logger = logging.getLogger()

def cross_validate_test(model, dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    scaler = dataset.target_scaler
    for _, (controls, targets) in enumerate(loader):
        output, error_bars = model.predict(controls, return_std=True)
        output = output.detach().numpy()

    scaled_targets = scaler.inverse_transform(targets)
    scaled_output = scaler.inverse_transform(output)

    RMSE = mean_squared_error(y_pred=scaled_output, y_true=scaled_targets, squared=False)

    return (scaled_targets, scaled_output, error_bars), RMSE

def plot_predictions(scaled_targets, scaled_output, target_params, RMSE, error_bars=None):
    # TODO: Add the additional parts

    if len(target_params) == 1:
        fig = plt.figure(figsize=(10, 8))
        plt.scatter(scaled_targets, scaled_targets, label='Truth')
        plt.scatter(scaled_targets, scaled_output, label='Predicted')
        if error_bars is not None:
            plt.errorbar(scaled_targets, scaled_output, error_bars.flatten(), ecolor='green', fmt='none', alpha=0.5)
        plt.legend()
        plt.xlabel('Target {}'.format(target_params))
        plt.ylabel('Predicted {}'.format(target_params))
    else:
        scaled_targets = scaled_targets.transpose()
        scaled_ouptut = scaled_output.transpose()
        fig, axs = plt.subplots(nrows=len(target_params), ncols=1, figsize=(15, 15))
        axs = axs.ravel()
        i = 0
        for target in scaled_targets:
            axs[i].scatter(target, target, label='Truth')
            axs[i].scatter(target, scaled_ouptut[i], label='Prediction')
            # TODO: different RMSE for each output
            axs[i].set(title=target_params[i] + ' RMSE: {:.3}'.format(RMSE), ylabel='Real {}'.format(target_params[i]), xlabel='Predicted')

            if error_bars is not None:
                axs[i].errorbar(target, scaled_ouptut[i], error_bars[:, i], ecolor='green', fmt='none', alpha=0.5)

            i += 1

    plt.suptitle('Predictions vs Actual with {}, {:.3}'.format('Average Ensemble', RMSE))
    plt.show()


def plot_results(net, dataset, config=None):
    logger.info('Plotting Predictions')

    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    scaler = dataset.target_scaler
    for _, (controls, targets) in enumerate(loader):
        output, error_bars = net.predict(controls, return_std=True)
        output = output.detach().numpy()

    scaled_targets = scaler.inverse_transform(targets)
    scaled_output = scaler.inverse_transform(output)

    if config == None:
        target_params = ''
    else:
        target_params = config['target_params']

    RMSE = mean_squared_error(y_pred=scaled_output, y_true=scaled_targets, squared=False)

    if len(target_params) == 1:
        fig = plt.figure(figsize=(10, 8))
        plt.scatter(scaled_targets, scaled_targets, label='Truth')
        plt.scatter(scaled_targets, scaled_output, label='Predicted')
        if error_bars is not None:
            plt.errorbar(scaled_targets, scaled_output, error_bars.flatten(), ecolor='green', fmt='none', alpha=0.5)
        plt.legend()
        plt.xlabel('Target {}'.format(target_params))
        plt.ylabel('Predicted {}'.format(target_params))
        plt.title('RMSE {:.3}'.format(RMSE))
    else:
        scaled_targets = scaled_targets.transpose()
        scaled_ouptut = scaled_output.transpose()
        fig, axs = plt.subplots(nrows=len(target_params), ncols=1, figsize=(15, 15))
        axs = axs.ravel()
        i = 0
        for target in scaled_targets:
            axs[i].scatter(target, target, label='Truth')
            axs[i].scatter(target, scaled_ouptut[i], label='Prediction')
            # TODO: different RMSE for each output
            axs[i].set(title=target_params[i] + ' RMSE: {:.3}'.format(RMSE), ylabel='Real {}'.format(target_params[i]), xlabel='Predicted')

            if error_bars is not None:
                axs[i].errorbar(target, scaled_ouptut[i], error_bars[:, i], ecolor='green', fmt='none', alpha=0.5)

            i += 1

    if isinstance(net.base_estimator_, type):
        base_estimator_name = net.base_estimator_.__name__
    else:
        base_estimator_name = net.base_estimator_.__class__.__name__

    plt.suptitle("{} with base model {} with {} estimators".format(type(net).__name__, base_estimator_name, net.n_estimators,))
    plt.show()


    # plot_predictions(scaled_targets, scaled_output, target_params, RMSE=RMSE, error_bars=error_bars)