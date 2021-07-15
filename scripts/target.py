from typing import Union


def soring_date2target(sorting_date: Union[int, float]) -> int:
    """Convert `sorting_date` to `target`.

    If `sorting_date` <= 1600 then `target` is 0.
    If 1600 < `sorting_date` <= 1700 then `target` is 1.
    If 1700 < `sorting_date` <= 1800 then `target` is 2.
    If 1800 < `sorting_date` then `target` is 3.

    Parameters
    ----------
    sorting_date: int
        Value of `sorting_date`

    Returns
    -------
    target: int
        0, 1, 2 or 3
    """
    if sorting_date <= 1600:
        return 0
    elif sorting_date <= 1700:
        return 1
    elif sorting_date <= 1800:
        return 2
    else:
        return 3
