import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import recall_score, f1_score, accuracy_score


def scorePerformance(ecgId_pred, ecgId_true):
    """
    the user classification task is scored according to macro averaged recall,
    with an adjustment for chance level. All performances are clipped at 0.0, so that zero indicates chance
    or worse performance, and 1.0 indicates perfect performance.
    :param ecgId_pred: 1D int32 numpy array. containing the predicted user ID label for each window.
    :param ecgId_true: 1D int32 numpy array. containing the true user ID label for each window.
    :retturn tuple(f1_score, accuracy, recall)
    """

    numElmts = None

    # Input checking
    if ecgId_true is not None:
        assert isinstance(ecgId_pred, np.ndarray)
        assert len(ecgId_pred.shape) == 1
        assert ecgId_pred.dtype == np.int32

        assert isinstance(ecgId_true, np.ndarray)
        assert len(ecgId_true.shape) == 1
        assert ecgId_true.dtype == np.int32

        assert len(ecgId_pred) == len(ecgId_true)
        if numElmts is not None:
            assert (len(ecgId_pred) == numElmts) and (len(ecgId_true) == numElmts)
        else:
            numElmts = len(ecgId_pred)

    if numElmts is None:
        return (0.0, 0.0, 0.0)

    return (
        f1_score(ecgId_true, ecgId_pred, average="weighted"),
        accuracy_score(ecgId_true, ecgId_pred),
        recall_score(ecgId_true, ecgId_pred, average="weighted"),
    )


def example():
    ecgId_pred = np.random.randint(low=0, high=32, size=(480,), dtype=np.int32)
    ecgId_true = np.random.randint(low=0, high=32, size=(480,), dtype=np.int32)

    print(scorePerformance(ecgId_true, ecgId_pred))
