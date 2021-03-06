import torch
from tqdm import tqdm
import numpy as np
import scoring_function as scoreF


def get_scoring_func_param_index(target_labels):
    scoring_func_param_index = [
        None if target_labels.count("treeid") == 0 else target_labels.index("treeid")
    ]
    return scoring_func_param_index


def update_prediction_data(score_index, y, outputs, ID):
    score_param_index = score_index

    treeId_true, treeId_pred = ID

    if score_param_index[0] is not None:
        i = score_param_index[0]
        _, pred_classes = torch.max(outputs[i], dim=1)
        if y is not None:
            treeId_true.extend(y[i].view(-1).tolist())
        treeId_pred.extend(pred_classes.view(-1).tolist())

    ID = [treeId_true, treeId_pred]

    return ID


def evaluate(
    model, device, dataloader, targets, criterion=None, weight=None, labeled=False
):

    model.eval()

    score_param_index = get_scoring_func_param_index(targets)
    if (weight is None) and (criterion is not None):
        weight = [1.0] * len(criterion)
    if criterion is not None:
        assert len(weight) == len(criterion)

    treeId_pred, treeId_true = None, None

    # labeled = dataloader.dataset.labeled

    if score_param_index[0] is not None:
        treeId_pred, treeId_true = [], [] if labeled else None

    valid_loss = 0
    valid_n_iter = 0

    with torch.no_grad():
        # for data in tqdm(dataloader):
        for data in dataloader:
            if labeled:
                x, y = data
                if not isinstance(y, (list, tuple)):
                    y = [y]
                x, y = x.to(device), [t.to(device) for t in y]
            else:
                x = data
                y = None
                x = x.to(device)

            outputs = model(x)
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]

            if labeled and (criterion is not None):
                loss = [
                    w * (c(o, gt) if o.size(1) == 1 else c(o, gt.squeeze(1)))
                    for c, o, gt, w in zip(criterion, outputs, y, weight)
                ]
                loss = sum(loss)

            ID = update_prediction_data(
                score_param_index, y, outputs, [treeId_true, treeId_pred]
            )
            treeId_true, treeId_pred = ID

            if labeled and (criterion is not None):
                valid_loss += loss.item()
            valid_n_iter += 1

    # mean loss
    mean_loss = valid_loss / max(valid_n_iter, 1)

    # metrics
    if treeId_pred is not None:
        treeId_pred = np.array(treeId_pred, dtype=np.int32)
    if treeId_true is not None:
        treeId_true = np.array(treeId_true, dtype=np.int32)

    if labeled:
        metrics = scoreF.scorePerformance(treeId_pred, treeId_true)
    else:
        return (treeId_pred,)

    preds = treeId_pred
    return mean_loss, metrics, preds

