'''
Module for Breiman's experiments using our implementation of random forests.
Looks for data in the BreimanDatasets subdirectory.
'''
from glob import glob
from parse_arff import Database
from math import log
from utils import log2
from random_forest import RandomForest
from evaluation import evaluate_model
from functools import partial
from random import shuffle
from statistics import mean
from multiprocessing import Pool



def single_evaluation(dataset_path):
    print('Performing a single evaluation of Breiman\'s accuracy experiments (using Erickson/Andersen code)')

    db = Database()
    db.read_data(dataset_path)
    shuffle(db.data)

    # 10% prune, 10% test, 80% training
    train_and_test, prune = next(db.k_fold(10))
    train, test = next(db.k_fold(9))

    # a) Random forest, with F = 1, ntrees=100
    forest_a = RandomForest(train, 1, 100)
    pre_error_a = 1 - evaluate_model(lambda _: forest_a, None, test)
    forest_a.prune(prune.data)
    post_error_a = 1 - evaluate_model(lambda _: forest_a, None, test)

    # b) Random forest with F = floor(log2(n_attributes) + 1), ntrees = 100
    F = int(log2(len(db.ordered_attributes) - 1) + 1)
    forest_b = RandomForest(train, F, 100)
    pre_error_b = 1 - evaluate_model(lambda _: forest_b, None, test)
    forest_b.prune(prune.data)
    post_error_b = 1 - evaluate_model(lambda _: forest_b, None, test)

    return min(pre_error_a, pre_error_b), min(post_error_a, post_error_b)


if __name__ == '__main__':
    datasets = glob('./BreimanDatasets/*.arff')
    for dataset_path in datasets:
        print(dataset_path)

        trials = 100

        inputs = [dataset_path] * trials
        pool = Pool()
        outputs = pool.map(single_evaluation, inputs)
        outputs = list(zip(*outputs))
        print(outputs)
        pre_errors = outputs[0]
        post_errors = outputs[1]
        before = mean(pre_errors) * 100
        after = mean(post_errors) * 100

        print('After {} trials, average error is pre={}% post={}%'.format(trials, before, after))
