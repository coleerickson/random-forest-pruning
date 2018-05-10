from decision_tree import RandomTree
from collections import Counter
from random import sample
from utils import mode, first

class RandomForest:
    ''' An ensemble of random trees. '''

    def __init__(self, database, attribute_subset_size, num_trees, max_depth=None):
        self.trees = []
        # print("Training a random forest with {} trees".format(num_trees))
        for i in range(num_trees):
            # debug_print("{:5d} / {:5d} trees trained".format(i, num_trees))

            num_examples = len(database.data)
            example_indices = list(range(num_examples))
            sampled_indices = sample(example_indices, num_examples)
            weights = [0] * num_examples
            for i in sampled_indices:
                weights[i] += 1
            
            self.trees.append(RandomTree(database, attribute_subset_size, max_depth=max_depth, weights=weights))

    def predict(self, example):
        ''' Gets each tree's prediction for the example; returns the most commonly occurring one. '''
        return mode(tree.predict(example) for tree in self.trees)

    def prune(self, pruning_examples):
        for tree in self.trees:
            tree.prune(pruning_examples)

    def __str__(self):
        return "TODO" # TODO

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

    # from cProfile import run as profile
    # profile("RandomTree(database, 16, max_depth=10)")
    # RandomTree(database, 5, max_depth=10)

    from random import shuffle
    shuffle(database.data)
    
    # 10% of 100% is 10%
    etc, pruning = first(database.k_fold(6))
    # 11% of 90% is 10%
    training, testing = first(etc.k_fold(5))
    # 80, 10, 10 ! wow ! 
    
    from evaluation import evaluate_model
    f = RandomForest(training, 1, 10)
    evaluate_model(lambda _: f, None, testing)
    f.prune(pruning.data)
    evaluate_model(lambda _: f, None, testing)
    
    # from evaluation import k_fold
    # print(k_fold(lambda db: RandomForest(db, 1, 20), database, 10))
