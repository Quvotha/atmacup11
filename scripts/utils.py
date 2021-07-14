from decimal import Decimal


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
