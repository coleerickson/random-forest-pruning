from collections import defaultdict, Counter
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
    result = 0
    for (xi, yi) in zip(x, y):
        result += xi * yi
    return result

def first(l):
    for x in l:
        return x

def mode(l):
    return Counter(l).most_common(1)[0][0]
    
def _get_class_from_example_with_weight(example_with_weight):
    return example_with_weight[0][-1]

def entropy(indices, examples, weights):
    '''
    Returns the entropy of `examples`. Entropy is defined in terms of the true class of the example. When counted, each of the `examples` is multiplied by the factor at the corresponding index in `weights`.
    '''
    total_weights = 0
    for weight in weights:
        total_weights += weight

    class_weights = defaultdict(lambda: 0)
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
    Computes the information gain of `database` by splitting on `attribute`. The examples in the database are reweighted by `weights`.
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

# def random_class(db):
#     return choice(db.attributes[db.ordered_attributes[-1]])

class DecisionTree:
    '''
    A classifier. The decision tree induction algorithm is ID3 which recursively greedily maximizes information gain. An attribute only ever appears once on a given path.
    '''
    class Node:
        
        def __init__(self, database, indices, weights, attributes, max_depth, attribute_selector, depth=1, parent=None):
            self.database = database
            self.parent = parent
            
            self.total_predictions = 0
            self.incorrect_predictions = 0

            if len(attributes) == 0:
                examples = [database.data[i] for i in indices]
                assert len(examples) != 0
                self.best_attribute = None
                self.majority_class = mode([ex[-1] for ex in examples])
                debug_print('  ' * depth + str(len(indices)) + " Leaf with majority class: " + str(self.majority_class))
                return
            
            def info_gain(x): return information_gain(database, indices, weights, x)
            selected_attrs = attribute_selector(attributes)
            if len(selected_attrs) == 1:
                self.best_attribute = selected_attrs[0]
            else:
                self.best_attribute = max(selected_attrs, key=info_gain)
            other_attributes = [attr for attr in attributes if attr != self.best_attribute]

            debug_print('  ' * depth + str(len(indices)) + ' ' + self.best_attribute)

            # if len(other_attributes) == 0 or (max_depth is not None and depth >= max_depth):
            #     # For each value of the best attribute, determine the majority class. In
            #     # `self.predictions`, map that attribute value to the majority class.
            #     self.predictions = {}
            #     attr_index = database.ordered_attributes.index(self.best_attribute)
            #     for attr_value in range(len(database.attributes[self.best_attribute])):
            #         # filtered_indices = [index for index, ex in enumerate(database.data) if ex[attr_index] == attr_value]
            #         filtered_indices = [index for index in indices if database.data[index] == attr_value]
            #         filtered_data = [database.data[i] for i in filtered_indices]
            #         filtered_weights = [weights[i] for i in filtered_indices]
                        
            #         neg_examples = [ex[-1] == 0 for ex in filtered_data]
                        
            #         weighted_neg = inner(neg_examples, filtered_weights)
            #         prediction = int(weighted_neg < (sum(filtered_weights) / 2))
            #         self.predictions[attr_value] = prediction
            #         self.desc.append('  ' * depth + 'attr_value {} for {} => {}'.format(attr_value, self.best_attribute, prediction))
            # else:
            self.predictions = {}
            attr_index = database.ordered_attributes.index(self.best_attribute)
            for attr_value in range(len(database.attributes[self.best_attribute])):
                indices_with_value = [index for index in indices if database.data[index][attr_index] == attr_value]
                debug_print('  ' * depth + str(len(indices)) + ' attr_value {} for {}'.format(attr_value, self.best_attribute))
                if len(indices_with_value) == 0:
                    # If there are no examples with this attribute value, then create a leaf node with the majority class from examples with indices `indices`
                    self.predictions[attr_value] = DecisionTree.Node(database, indices, weights, [], max_depth, attribute_selector, depth=depth + 1, parent=self)

                else:
                    self.predictions[attr_value] = DecisionTree.Node(database, indices_with_value, weights, other_attributes, max_depth, attribute_selector, depth=depth + 1, parent=self)


        def predict(self, example):
            # TODO don't store database, just store bestattrindex
            if self.best_attribute is None:
                assert self.majority_class is not None
                return self.majority_class
                
            attr_index = self.database.ordered_attributes.index(self.best_attribute)
            prediction = self.predictions[example[attr_index]]

            if isinstance(prediction, DecisionTree.Node):
                prediction = prediction.predict(example)
            else:
                assert False

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
            pass

    def __init__(self, database, max_depth=None, weights=None, attribute_selector=lambda attributes: attributes):
        ''' Learns/creates the decision tree by selecting the attribute that maximizes information gain. '''
        if weights is None:
            weights = [1] * len(database.data)
        indices = list(range(len(database.data)))
        self.root = DecisionTree.Node(database, indices, weights, database.ordered_attributes[:-1], max_depth, attribute_selector)

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
                pass

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
    from evaluation import k_fold
    print(k_fold(lambda db: DecisionTree(db, max_depth=6), database, 10))

    # for leaf in RandomTree(database, 1, max_depth=4).leaves():
    #     print(leaf)
    # tree = RandomTree(database, 1, max_depth=4)
    # leaves = set(tree.leaves())
    # print(len(leaves))
    # from pprint import pprint
    # pprint([(i, str(leaf)) for i, leaf in enumerate(leaves)])

    # parents = []
    # for leaf in leaves:
    # parents.append(leaf.parent)
    # print(len(parents))
    # print(len(set(parents)))
