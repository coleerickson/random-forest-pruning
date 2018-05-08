from decision_tree import RandomTree
from collections import Counter

def mode(l):
    return Counter(l).most_common(1)[0][0]

class RandomForest:
    ''' An ensemble of random trees. '''

    def __init__(self, database, attribute_subset_size, num_trees, max_depth=None):
        self.trees = []
        for _ in range(num_trees):
            self.trees.append(RandomTree(database, attribute_subset_size, max_depth=max_depth))

    def predict(self, example):
        ''' Gets each tree's prediction for the example; returns the most commonly occurring one. '''
        return mode(tree.predict(example) for tree in self.trees)

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

    from evaluation import k_fold
    print(k_fold(lambda db: RandomForest(db, 5, 5, max_depth=6), database, 10))
