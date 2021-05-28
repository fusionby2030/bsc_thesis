"""

INTO CODEBASE:


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


from sklearn.model_selection import RepeatedKFold
def cross_validate_torch(n_splits=5, n_repeats=5):
    cv = RepeatedKFold(n_splits=5, n_repeats=1)

def main(control_tensor, target_tensor, **kwargs):


    pass



if __name__ == '__main__':
    # make data and tensors
    control_space, target_space, _, _ = utils.process_data()
    control_tensors, target_tensors = utils.setup_tensors(control_space, target_space)

    # pass to main
    # relevant ANN kwargs get passed as kwargs

    main(control_tensors, target_tensors)

    # reason for making data first is
    # (a) it is reproducible, (b) so it is not done every god damn time a CV experiment happens
    pass
