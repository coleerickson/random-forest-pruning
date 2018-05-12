from decision_tree import RandomTree
from collections import Counter
from random import sample
from utils import mode, first

class RandomForest:
    ''' An ensemble of random trees. '''

    def __init__(self, database, attribute_subset_size, num_trees, max_depth=None):
        ''' Trains `num_trees` random trees and bootstraps the dataset for each tree. '''
        self.trees = []
        # print("Training a random forest with {} trees".format(num_trees))
        for i in range(num_trees):
            # print("{:5d} / {:5d} trees trained".format(i, num_trees))

            num_examples = len(database.data)
            example_indices = list(range(num_examples))
            sampled_indices = sample(example_indices, num_examples)
            weights = [0] * num_examples
            for i in sampled_indices:
                weights[i] += 1

            self.trees.append(RandomTree(database, attribute_subset_size, max_depth=max_depth, weights=weights))

    def predict(self, example):
        ''' Gets each tree's prediction for the example; returns the mode. '''
        return mode(tree.predict(example) for tree in self.trees)

    def prune(self, pruning_examples):
        ''' Prunes every tree in the forest using the same pruning set. '''
        for tree in self.trees:
            tree.prune(pruning_examples)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='RandomForest')

    parser = ArgumentParser()
    parser.add_argument('dataset_path', help='.arff file name')

    args = parser.parse_args()

    from parse_arff import Database
    database = Database()
    database.read_data(args.dataset_path)
    print(database)


    from random import shuffle
    shuffle(database.data)

    # Split the dataset into pruning, training, and testing
    etc, pruning = first(database.k_fold(6))
    training, testing = first(etc.k_fold(5))

    # Evaluate the forest before and after pruning
    from evaluation import evaluate_model
    f = RandomForest(training, 1, 10)
    evaluate_model(lambda _: f, None, testing)
    f.prune(pruning.data)
    evaluate_model(lambda _: f, None, testing)
