import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import recall_score, f1_score, accuracy_score


def scorePerformance(treeId_pred, treeId_true):
    """
    the tree classification task is scored according to macro averaged recall,
    with an adjustment for chance level. All performances are clipped at 0.0, so that zero indicates chance
    or worse performance, and 1.0 indicates perfect performance.
    :param treeId_pred: 1D int32 numpy array. containing the predicted tree ID label for each window.
    :param treeId_true: 1D int32 numpy array. containing the true tree ID label for each window.
    :retturn tuple(f1_score, accuracy, recall)
    """

    numElmts = None

    # Input checking
    if treeId_true is not None:
        assert isinstance(treeId_pred, np.ndarray)
        assert len(treeId_pred.shape) == 1
        assert treeId_pred.dtype == np.int32

        assert isinstance(treeId_true, np.ndarray)
        assert len(treeId_true.shape) == 1
        assert treeId_true.dtype == np.int32

        assert len(treeId_pred) == len(treeId_true)
        if numElmts is not None:
            assert (len(treeId_pred) == numElmts) and (len(treeId_true) == numElmts)
        else:
            numElmts = len(treeId_pred)

    if numElmts is None:
        return (0.0, 0.0, 0.0)

    return (
        f1_score(treeId_true, treeId_pred, average="weighted"),
        accuracy_score(treeId_true, treeId_pred),
        recall_score(treeId_true, treeId_pred, average="weighted"),
    )


def example():
    treeId_pred = np.random.randint(low=0, high=32, size=(480,), dtype=np.int32)
    treeId_true = np.random.randint(low=0, high=32, size=(480,), dtype=np.int32)

    print(scorePerformance(treeId_true, treeId_pred))
