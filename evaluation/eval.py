import os
import sys
import argparse

from torchvision import transforms

import numpy as np
from joblib import load  # You can use Pickle or the serialization technique of your choice

sys.path.append("../")
from utils.dataset import HoromaDataset  # Import your own pytorch Dataset here


def eval_model(model_path, dataset_dir, split):
    '''
    # MODIFY HERE #
    This function is meant to be an example

    '''

    # # SETUP MODEL # #
    # Load your best model
    print("\nLoading model from ({}).".format(model_path))
    model = load(model_path)

    # # SETUP DATASET # #
    # Load requested dataset
    """ IMPORTANT # of example per splits.
    "train" = 150700
    "train_overlapped" = 544027
    "valid" = 499
    "valid_overlapped" = 1380
    "test" = 483

    Files available the test folder:
        test_regions_id.txt
        test_x.dat
        test_y.txt
        train_overlapped_regions_id.txt
        train_overlapped_x.dat
        train_overlapped_y.txt
        train_regions_id.txt
        train_x.dat
        train_y.txt
        valid_overlapped_regions_id.txt
        valid_overlapped_x.dat
        valid_overlapped_y.txt
        valid_regions_id.txt
        valid_x.dat
        valid_y.txt

    You need to load the right one according to the `split`.
    """
    dataset = HoromaDataset(
        dataset_dir,
        split,
        transforms=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    )

    # # INFERENCE # #
    # Use model on dataset to predict the class
    prediction = model.predict(dataset)

    # We return -1 if the cluster has no label...
    map_labels = np.concatenate([dataset.map_labels, np.array([''])])

    pred = map_labels[prediction]

    # # PREDICTIONS # #
    # Return the predicted classes as a numpy array of shape (nb_exemple, 1)
    """ Example:
    [['ES']
     ['EN']
     ['ES']]
    """
    return pred


if __name__ == "__main__":

    # Put your group name here
    group_name = "b2phot1"

    model_path = "/rap/jvb-000-aa/COURS2019/etudiants/submissions/b2phot1/model/best_model.pkl"
    # model_path should be the absolute path on shared disk to your best model.
    # You need to ensure that they are available to evaluators on Helios.

    #########################
    # DO NOT MODIFY - BEGIN #
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", type=str, default="/rap/jvb-000-aa/COURS2019/etudiants/data/horoma/",
                        help="Absolute path to the dataset directory.")
    parser.add_argument("-s", "--dataset_split", type=str, choices=['valid', 'test', 'train'], default="valid",
                        help="Which split of the dataset should be loaded from `dataset_dir`.")
    parser.add_argument("-r", "--results_dir", type=str, default="./",
                        help="Absolute path to where the predictions will be saved.")
    parser.add_argument('-m', '--model', type=str, default=model_path,
                        help='Absolute path to the best model.')
    args = parser.parse_args()

    model_path = args.model

    # Arguments validation
    if group_name is "b1phut_N":
        print("'group_name' is not set.\nExiting ...")
        exit(1)

    if model_path is None or not os.path.exists(model_path):
        print("'model_path' ({}) does not exists or unreachable.\nExiting ...".format(model_path))
        exit(1)

    if args.dataset_dir is None or not os.path.exists(args.dataset_dir):
        print("'dataset_dir' does not exists or unreachable..\nExiting ...")
        exit(1)

    y_pred = eval_model(model_path, args.dataset_dir, args.dataset_split)

    assert type(y_pred) is np.ndarray, "Return a numpy array"
    assert len(y_pred.shape) == 1, "Make sure ndim=1 for y_pred"

    results_fname = os.path.join(args.results_dir, "{}_pred_{}.txt".format(group_name, args.dataset_split))

    print('\nSaving results to ({})'.format(results_fname))
    np.savetxt(results_fname, y_pred, fmt='%s')
    # DO NOT MODIFY - END #
    #######################
