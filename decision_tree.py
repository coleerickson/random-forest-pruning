'''
An implementation of decision trees and random trees.
When run as the main module, takes a single argument: the path to an ARFF file, and then returns the accuracy of
a single random tree trained on the dataset, both before and after pruning.
'''

from collections import defaultdict, Counter
from math import log
from random import sample
from pprint import pformat
from utils import first, mode, log2, inner

DEBUG = False
def debug_print(*args):
    if DEBUG:
        print(*args)

def entropy(indices, examples, weights):
    '''
    Returns the entropy of `examples`. Entropy is defined in terms of the true class of the example. When counted,
    each of the `examples` is multiplied by the factor at the corresponding index in `weights`.
    '''
    total_weights = 0
    for weight in weights:
        total_weights += weight

    class_weights = defaultdict(lambda: 0) # TODO could use a counter
    for index in indices:
        example = examples[index]
        weight = weights[index]
        klass = example[-1]
        class_weights[klass] += weight

    class_ratios = [klass_weight / total_weights for klass, klass_weight in class_weights.items()]

    entropy_terms = [-ratio * log2(ratio) for ratio in class_ratios]
    entropy = 0
    for entropy_term in entropy_terms:
        entropy += entropy_term
    return entropy


def information_gain(database, indices, weights, attribute):
    '''
    Computes the information gain of `database` by splitting on `attribute`. The examples in the database are
    reweighted by `weights`.
    '''
    total_entropy = entropy(range(len(database.data)), database.data, weights)
    gain = total_entropy

    # Compute split entropy
    attr_index = database.ordered_attributes.index(attribute)
    example_indices_by_attr_value = defaultdict(list)
    for index in indices:
        example = database.data[index]
        attr_value = example[attr_index]
        example_indices_by_attr_value[attr_value].append(index)

    for attr_value, example_indices in example_indices_by_attr_value.items():
        gain -= entropy(example_indices, database.data, weights) * len(example_indices) / len(database)

    return gain

class DecisionTree:
    '''
    A classifier. Uses ID3, recursively greedily maximizing information gain. One subtree per value of
    nominal-valued attributes. An attribute only ever appears once on a given path.
    '''
    class Node:
        def __init__(self, database, indices, weights, attributes, max_depth, attribute_selector, depth=1):
            '''
            Recursive constructor for nodes of the decision tree.
            database - the Database that contains the training data
            indices - a list of the indices into the database that serve as the training data for this node
            weights - a list of weights, one for each entry in the database
            attributes - the attributes being considered for use as splitting critera in this subtree
            max_depth - the maximum depth of the tree
            attribute_selector - a function that takes `attributes` as a parameter and returns some subset of it
                In a normal decision tree, this is the identity function
                In a random tree, this randomly samples `attributes`
            depth - the depth of the tree so far
            '''
            self.database = database

            # This state is useful for pruning
            self.prune_classes = Counter()
            for prune_class in database.attributes[database.ordered_attributes[-1]]:
                self.prune_classes[prune_class] = 0
            self.indices = indices

            # If no attributes remain, then we should create a leaf node and record the majority class of the
            # associated examples. We also do this if we have reached maximum depth.
            if len(attributes) == 0 or (max_depth is not None and depth >= max_depth):
                examples = [database.data[i] for i in indices]
                assert len(examples) != 0
                self.best_attribute = None
                class_counts = Counter()
                for weight, example in zip(weights, examples):
                    klass = example[-1]
                    class_counts[klass] += weight
                self.majority_class = mode(class_counts)
                debug_print('  ' * depth + str(len(indices)) + " Leaf with majority class: " + str(self.majority_class))
                return

            # Otherwise we should select the best attribute based on information gain
            def info_gain(x): return information_gain(database, indices, weights, x)
            selected_attrs = attribute_selector(attributes)
            if len(selected_attrs) == 1:
                self.best_attribute = selected_attrs[0]
            else:
                self.best_attribute = max(selected_attrs, key=info_gain)
            other_attributes = [attr for attr in attributes if attr != self.best_attribute]
            debug_print('  ' * depth + str(len(indices)) + ' ' + self.best_attribute)

            # Then we should consider each of the attribute values for this best attribute, and recursively create subtrees where appropriate
            self.predictions = {}
            attr_index = database.ordered_attributes.index(self.best_attribute)
            for attr_value in range(len(database.attributes[self.best_attribute])):
                indices_with_value = [index for index in indices if database.data[index][attr_index] == attr_value]
                debug_print('  ' * depth + str(len(indices)) + ' attr_value {} for {}'.format(attr_value, self.best_attribute))
                if len(indices_with_value) == 0:
                    # If there are no examples with this attribute value, then create a leaf node with the majority class from examples with indices `indices`
                    self.predictions[attr_value] = DecisionTree.Node(database, indices, weights, [], max_depth, attribute_selector, depth=depth + 1)

                else:
                    # Otherwise recursively create another node based off the examples with this attribute value
                    self.predictions[attr_value] = DecisionTree.Node(database, indices_with_value, weights, other_attributes, max_depth, attribute_selector, depth=depth + 1)


        def predict(self, example):
            ''' Traverses the subtree and returns the predicted class of the example. '''
            if self.best_attribute is None:
                assert self.majority_class is not None
                return self.majority_class

            attr_index = self.database.ordered_attributes.index(self.best_attribute)
            prediction = self.predictions[example[attr_index]]

            if isinstance(prediction, DecisionTree.Node):
                prediction = prediction.predict(example)
            else:
                assert False

            return prediction

        def is_leaf(self):
            return self.best_attribute is None

        def prune_classify(self, example):
            '''
            Used as part of pruning. Takes an example from the pruning set and updates the error
            counts for each node in the subtree.
            '''
            prune_class = example[-1]
            self.prune_classes[prune_class] += 1

            if not self.is_leaf():
                attr_index = self.database.ordered_attributes.index(self.best_attribute)
                attr_value = example[attr_index]
                child = self.predictions[attr_value]
                child.prune_classify(example)

        def prune(self):
            '''
            Mutates the tree, performing reduced-error pruning. Uses the bottom-up algorithm described
            in Elomaa and Kaariainen 2001 (see Table 1 and section 2.4), extended to handle non-binary
            nominal-valued attributes. Returns the number of misclassifications.
            '''
            majority_prune_class = self.prune_classes.most_common(1)[0][0]

            total = sum(self.prune_classes.values())
            correct = self.prune_classes[majority_prune_class]
            my_error = total - correct
            if self.is_leaf():
                return my_error
            else:
                children_error = sum(child.prune() for child in self.predictions.values())
                if children_error < my_error:
                    return children_error
                else:
                    # Prune
                    del self.predictions
                    self.best_attribute = None

                    # See section 2 of Elomaa et al.:
                    # "Neither is it obvious whether the training set or pruning set is used to decide
                    # the labels of the leaves that result from pruning." Here we make the same choice
                    # as Elomaa et al.: Use the pruning set.
                    self.majority_class = majority_prune_class

                    correct = self.prune_classes[majority_prune_class]
                    total = sum(self.prune_classes.values())
                    return min(correct, total - correct)

        def size(self):
            if self.is_leaf():
                return 1
            else:
                return sum(child.size() for child in self.predictions.values())


    def __init__(self, database, max_depth=None, weights=None, attribute_selector=lambda attributes: attributes):
        ''' Learns/creates the decision tree by selecting the attribute that maximizes information gain. '''
        if weights is None:
            weights = [1] * len(database.data)
        indices = list(range(len(database.data)))
        self.root = DecisionTree.Node(database, indices, weights, database.ordered_attributes[:-1], max_depth, attribute_selector)

    def predict(self, example):
        '''
        Returns the predicted class of `example` based on the attribute that maximized information gain
        at training time.
        '''
        return self.root.predict(example)

    def prune(self, pruning_examples):
        ''' Given a pruning set, performs reduced-error pruning on the tree. See DecisionTree.Node.prune '''
        for example in pruning_examples:
            self.root.prune_classify(example)
        self.root.prune()

def RandomTree(database, attribute_subset_size, max_depth=None, weights=None):
    '''
    Returns a decision tree. The attribute to split on is chosen from a random subset of the attributes at the node.
    The size of this random subset is `attribute_subset_size`.
    '''
    if attribute_subset_size > len(database.ordered_attributes) - 1:
        raise Exception(
            "Attempted to create tree with larger random attribute subset than number of attributes. {} > {}".format(
                attribute_subset_size,
                len(database.ordered_attributes) - 1
            )
        )
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
    parser.add_argument('-r', '--random', help='the random attribute subset size (F); if not supplied, then the tree is not random')
    parser.add_argument('-d', '--max_depth', help='an upper bound on the depth of the tree')
    parser.add_argument('-v', '--verbose', help='if supplied, prints the decision tree during induction', action='store_true')

    args = parser.parse_args()
    DEBUG = bool(args.verbose)
    max_depth = int(args.max_depth) if args.max_depth is not None else None
    attribute_subset_size = int(args.random) if args.random is not None else None

    # Parse the database
    from parse_arff import Database
    database = Database()
    database.read_data(args.dataset_path)
    print(database)

    # Shuffle it
    from random import shuffle
    shuffle(database.data)

    # The code here is hacky. We split the database by using the `k_fold` function. We do so twice to
    # obtain three datasets: `pruning`, `training`, `testing`.
    etc, pruning = first(database.k_fold(6))
    training, testing = first(etc.k_fold(5))


    if max_depth is not None:
        print('Using a maximum depth of ' + str(max_depth))
    else:
        print('Training the tree to full size')
    
    if attribute_subset_size is not None:
        print('Training a random tree with F = ' + str(attribute_subset_size))
        t = RandomTree(training, attribute_subset_size, max_depth=max_depth)
    else:
        print('Training a non-random decision tree')
        t = DecisionTree(training, max_depth=max_depth)

    # The code here is again hacky. It prints the size and accuracy of a random tree trained on the above
    # datasets before and after pruning.
    from evaluation import evaluate_model
    print('Tree size before pruning: ', t.root.size(), '\n\t', sep='', end='')
    evaluate_model(lambda _: t, None, testing)
    t.prune(pruning.data)
    print('Tree size after pruning: ', t.root.size(), '\n\t', sep='', end='')
    evaluate_model(lambda _: t, None, testing)
