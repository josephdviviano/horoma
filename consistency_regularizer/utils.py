from collections import OrderedDict
from pathlib import Path
from tensorboardX import SummaryWriter
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, top_k=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    if len(res) == 1:
        res = res[0]

    return res


def save_checkpoint(model, epoch, filename, optimizer=None, reg_model=None):
    data = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }
    if optimizer is not None:
        data['optimizer'] = optimizer.state_dict()
    if reg_model is not None:
        data['reg_model'] = reg_model.state_dict()
    torch.save(data, filename)


def load_checkpoint(model, path, optimizer=None, reg_model=None, device=None):
    resume = torch.load(path, map_location=device)

    if ('module' in list(resume['state_dict'].keys())[0]) \
            and not (isinstance(model, torch.nn.DataParallel)):
        new_state_dict = OrderedDict()
        for k, v in resume['state_dict'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(resume['state_dict'])

    data_return = [resume['epoch'], model]

    if optimizer is not None:
        if 'optimizer' in resume:
            optimizer.load_state_dict(resume['optimizer'])
        data_return.append(optimizer)

    if reg_model is not None:
        if 'reg_model' in resume:
            if ('module' in list(resume['reg_model'].keys())[0]) \
                    and not (isinstance(reg_model, torch.nn.DataParallel)):
                new_state_dict = OrderedDict()
                for k, v in resume['reg_model'].items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v

                reg_model.load_state_dict(new_state_dict)
            else:
                reg_model.load_state_dict(resume['reg_model'])
        data_return.append(reg_model)

    return data_return


def set_path(path):
    path_dir = '/'.join(path.split('/')[:-1])
    if not Path(path_dir).exists():
        Path(path_dir).mkdir(parents=True)
    return path


def set_tensorborad_writer(tf_board_path):
    if not tf_board_path.endswith('/'):
        tf_board_path += '/temp'
    else:
        tf_board_path += 'temp'
    set_path(tf_board_path)
    writer = SummaryWriter(tf_board_path)
    return writer

