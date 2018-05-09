from collections import defaultdict
from math import log
from random import sample
from pprint import pformat

DEBUG = False
if DEBUG:
    debug_print = print
else:
    debug_print = lambda _: None

def log2(x):
    ''' Returns the base 2 logarithm of `x`. '''
    return log(x, 2)

def inner(x, y):
    ''' Returns the inner product (dot product) of vectors `x` and `y`, where `x` and `y` are represented as lists. '''
    return sum(xi * yi for (xi, yi) in zip(x, y))

def first(l):
    for x in l:
        return x

def _get_class_from_example_with_weight(example_with_weight):
    return example_with_weight[0][-1]

def entropy(indices, examples, weights):
    '''
    Returns the entropy of `examples`. Entropy is defined in terms of the true class of the example. When counted, each of the `examples` is multiplied by the factor at the corresponding index in `weights`.
    '''
    total_weights = sum(weights)

    class_weights = defaultdict(lambda: 0)
    for index in indices:
        example = examples[index]
        weight = weights[index]
        klass = example[-1]
        class_weights[klass] += weight

    class_ratios = [klass_weight / total_weights for klass, klass_weight in class_weights.items()]

    entropy_terms = [-ratio * log2(ratio) for ratio in class_ratios]
    entropy = sum(entropy_terms)
    return entropy


def information_gain(database, weights, attribute):
    '''
    Computes the information gain of `database` by splitting on `attribute`. The examples in the database are reweighted by `weights`.
    '''
    total_entropy = entropy(range(len(database.data)), database.data, weights)
    gain = total_entropy

    # Compute split entropy
    attr_index = database.ordered_attributes.index(attribute)
    example_indices_by_attr_value = defaultdict(list)
    for example_index, example in enumerate(database.data):
        attr_value = example[attr_index]
        example_indices_by_attr_value[attr_value].append(example_index)

    for attr_value, example_indices in example_indices_by_attr_value.items():
        gain -= entropy(example_indices, database.data, weights) * len(example_indices) / len(database)

    return gain


class DecisionTree:
    '''
    A classifier. The decision tree induction algorithm is ID3 which recursively greedily maximizes information gain. An attribute only ever appears once on a given path.
    '''
    class Node:
        def __init__(self, database, weights, attributes, max_depth, attribute_selector, depth=1, parent=None):
            self.database = database
            self.parent = parent
            def info_gain(x): return information_gain(database, weights, x)
            self.best_attribute = max(attribute_selector(attributes), key=info_gain)
            other_attributes = [attr for attr in attributes if attr != self.best_attribute]

            self.desc = []
            self.desc.append('  ' * depth + self.best_attribute)
            if len(other_attributes) == 0 or (max_depth is not None and depth >= max_depth):
                # For each value of the best attribute, determine the majority class. In
                # `self.predictions`, map that attribute value to the majority class.
                self.predictions = {}
                attr_index = database.ordered_attributes.index(self.best_attribute)
                for attr_value in range(len(database.attributes[self.best_attribute])):
                    filtered_indices = [index for index, ex in enumerate(database.data) if ex[attr_index] == attr_value]
                    filtered_data = [database.data[i] for i in filtered_indices]
                    filtered_weights = [weights[i] for i in filtered_indices]

                    neg_examples = [ex[-1] == 0 for ex in filtered_data]

                    weighted_neg = inner(neg_examples, filtered_weights)
                    prediction = int(weighted_neg < (sum(filtered_weights) / 2))
                    self.predictions[attr_value] = prediction
                    self.desc.append('  ' * depth + 'attr_value {} for {} => {}'.format(attr_value, self.best_attribute, prediction))
            else:
                self.predictions = {}
                attr_index = database.ordered_attributes.index(self.best_attribute)
                for attr_value in range(len(database.attributes[self.best_attribute])):
                    self.desc.append('  ' * depth + 'attr_value {} for {}'.format(attr_value, self.best_attribute))
                    self.predictions[attr_value] = DecisionTree.Node(database, weights, other_attributes, max_depth, attribute_selector, depth=depth + 1, parent=self)
            self.desc = '\n'.join(self.desc)

            self.total_predictions = 0
            self.incorrect_predictions = 0

        def predict(self, example):
            # TODO don't store database, just store bestattrindex
            attr_index = self.database.ordered_attributes.index(self.best_attribute)
            prediction = self.predictions[example[attr_index]]

            if isinstance(prediction, DecisionTree.Node):
                prediction = prediction.predict(example)

            # For pruning, keep track of misclassifications
            self.total_predictions += 1
            if prediction != example[-1]:
                self.incorrect_predictions += 1

            return prediction

        def leaves(self):
            # If we are at a leaf node, return
            if not isinstance(first(self.predictions.values()), DecisionTree.Node):
                return [self]
            else:
                leaves = []
                for subtree in self.predictions.values():
                    leaves.extend(subtree.leaves())
                return leaves

        def prune(self):

        def __str__(self):
            return self.desc

    def __init__(self, database, max_depth=None, weights=None, attribute_selector=lambda attributes: attributes):
        ''' Learns/creates the decision tree by selecting the attribute that maximizes information gain. '''
        if weights is None:
            weights = [1] * len(database.data)
        self.root = DecisionTree.Node(database, weights, database.ordered_attributes[:-1], max_depth, attribute_selector)

    def predict(self, example):
        ''' Returns the predicted class of `example` based on the attribute that maximized information gain at training time. '''
        return self.root.predict(example)

    def prune(self, prune_data):
        ''' Mutates the tree, performing reduced-error pruning. Uses bottom-up algorithm described in Elomaa and Kaariainen. '''
        # TODO clear misclassification counts

        for example in prune_data:
            self.predict(example)

        markers = set(self.leaves())
        while True:
            markers = set(marker.parent for marker in markers)
            for marker in markers:


    def leaves(self):
        return self.root.leaves()

    def __str__(self):
        return str(self.root)

def RandomTree(database, attribute_subset_size, max_depth=None, weights=None):
    if attribute_subset_size > len(database.ordered_attributes) - 1:
        raise Exception("Attempted to create tree with larger random attribute subset than number of attributes. {} > {}".format(attribute_subset_size, len(database.ordered_attributes) - 1))
    def attribute_selector(attributes):
        if len(attributes) <= attribute_subset_size:
            return attributes
        else:
            return sample(attributes, attribute_subset_size)

    return DecisionTree(database, max_depth=max_depth, weights=weights, attribute_selector=attribute_selector)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='DecisionTree')

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
    #
    # from evaluation import k_fold
    # print(k_fold(lambda db: RandomTree(db, 1, max_depth=4), database, 10))

    # for leaf in RandomTree(database, 1, max_depth=4).leaves():
    #     print(leaf)
    leaves = set(RandomTree(database, 1, max_depth=4).leaves())
    print(len(leaves))
    from pprint import pprint
    pprint([(i, str(leaf)) for i, leaf in enumerate(leaves)])

    parents = []
    for leaf in leaves:
        parents.append(leaf.parent)
    print(len(parents))
    print(len(set(parents)))
