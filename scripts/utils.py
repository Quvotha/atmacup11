import copy
from decimal import Decimal
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader


def should_stop_earlier(metrics, *, patience: int = 10, min_delta: float = 1e-4,
                        greater_is_better: bool = False) -> bool:
    """Check whether metric stops improving or not.

    Early stopping is to be detected if metric shows no improvement during a certain number of iteration.

    Parameters
    ----------
    metrics : Iterable of metric value
        1st element is metric of 1st epoch, 2nd one is that of 2nd epoch, 3rd is ...
    patience : int, optional
        If latest `patience` epoch's metrics show no improvement, early stopping is detected. 
        Should be integer greater than 1. By default 10.
    min_delta : float, optional
        Metric is to be treated as getting improved if a quantity of metric's improvement is bigger than `min_delta`.
        Otherwise treated as that there are no improvement. Should be float greater than or equal to 0. 
        By default 1e-4.
    greater_is_better : bool, optional
        If a metiric's type is loss, set False. If that is score, set True. By default False.

    Returns
    -------
    Early stipping signal: bool
        Return True if training should be stopped, otherwise False.
    """
    if not (isinstance(min_delta, float) and min_delta >= 0.):
        raise ValueError('`min_delta` shuold be float >= 0.0 but {} given'
                         .format(min_delta))
    if not (isinstance(patience, int) and patience > 1):
        raise ValueError('`patience` shuold be integer >= 2 but {} given'
                         .format(patience))
    if len(metrics) < patience:
        return False
    sign = 1 if greater_is_better else -1
    improvement_quantities = [sign * (metrics[i + 1] - metrics[i]) for i in range(len(metrics) - 1)]
    max_improvement_quantity = max(improvement_quantities[-patience:])
    max_improvement_quantity_decimal = Decimal(max_improvement_quantity) \
        .quantize(Decimal(str(min_delta)))
    min_delta_decimal = Decimal(min_delta).quantize(Decimal(str(min_delta)))
    return max_improvement_quantity_decimal <= min_delta_decimal


def train_model(model, dataloaders: Dict[str, DataLoader], *,
                criterion, optimizer, device: torch.device = torch.device('cpu'),
                num_epochs: int = 100, is_inception=False, log_func: Optional[callable] = None,
                patience: int = 5, min_delta: float = 0.00001) -> tuple:
    """Training model with early stopping framework.

    Parameters
    ----------
    model :
        Model to be trained.
    dataloaders : Dict[str, DataLoader]
        Should have 2 keys: 'train' and 'val. dataloaders['train'] should be a dataloader of 
        training set and dataloaders['val'] be an one of evaluation set.
    criterion : [type]
        Instance of evaluator, such as `torch.nn.MSELoss`.
    optimizer : [type]
        Instance of optimizer, such as `torch.optim.SGD`.
    device : torch.device, optional
        Device where calculation will be performet. By default `torch.device('cpu')`.
    num_epochs : int, optional
        Number of epochs, by default 100
    is_inception : bool, optional
        Set True if `model` is inception, by default False
    log_func : Optional[callable], optional
        Logging function, such as `Logger.info`. If None is given, `print` will be used. 
        By default None
    patience : int, optional
        See `should_stop_earlier`'s documentation, by default 5
    min_delta : float, optional
        See `should_stop_earlier`'s documentation, by default 0.00001

    Returns
    -------
    (model, training loss history, validation loss history): tuple
        `model` is model instance of which weight is updatetd by training.
        `training loss history` and `validation loss history` are epoch loss histories.

    Refference
    ----------
    https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    """
    log_func = log_func or print
    best_model_wts = copy.deepcopy(model.state_dict())
    minimum_loss = np.inf
    train_losses, eval_losses = [], []  # save loss history here
    early_stopping_signal = False  # switch to True if early stopping is detected

    for epoch in range(num_epochs):
        if early_stopping_signal:
            break
        log_func('Epoch {}/{}'.format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels.reshape(outputs.size()))
                        loss2 = criterion(aux_outputs, labels.reshape(aux_outputs.size()))
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels.reshape(outputs.size()))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            log_func('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model if performance is inprove on this epoch
            if phase == 'val' and epoch_loss < minimum_loss:
                best_model_wts = copy.deepcopy(model.state_dict())
                minimum_loss = epoch_loss
            # save loss history
            if phase == 'val':
                eval_losses.append(epoch_loss)
            else:
                train_losses.append(epoch_loss)
            # check whether training should be stopped
            if phase == 'val':
                early_stopping_signal = should_stop_earlier(
                    eval_losses, patience=patience, min_delta=min_delta)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_losses, eval_losses


def predict_by_model(model, dataloader: DataLoader, device: torch.device = torch.device('cpu')):
    """Predict 

    Parameters
    ----------
    model : [type]
        Model to be used for prediction
    dataloader : DataLoader
        Supply features to be used for prediction
    device : torch.device, optional
        Device where calculation will be performet. By default `torch.device('cpu')`.

    Returns
    -------
    prediction : np.ndarray

    Refference
    ----------
    https://www.kaggle.com/yasufuminakama/moa-pytorch-nn-starter
    """

    model.eval()
    preds = []

    for inputs in dataloader:

        inputs = inputs.to(device)

        with torch.no_grad():
            pred = model(inputs)

        preds.append(pred.detach().cpu().numpy())

    preds = np.concatenate(preds)

    return preds
