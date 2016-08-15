

def normalise(X, axis=None):
    """
    :param X: matrix like object
    :param axis: dimenison to normalise in
    :return: Returns a demeaned and variance normalized matrix
    """
    if not axis:
        s = X.shape
        if s(0) > 1:
            axis = 0
        else:
            axis = 1
    # ddof set to 1 to match default behaviour of matlab std
    return (X - X.mean(axis=axis)) / X.std(axis=axis, ddof=1)
