from math import log

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


def entropy(examples, weights):
    '''
    Returns the entropy of `examples`. Entropy is defined in terms of the true class of the example. When counted, each of the `examples` is multiplied by the factor at the corresponding index in `weights`.
    '''
    negative_examples = [example[-1] == 0 for example in examples]
    positive_examples = [example[-1] == 1 for example in examples]

    if sum(negative_examples) == 0 or sum(positive_examples) == 0:
        return 0

    weighted_neg = inner(negative_examples, weights)
    weighted_pos = inner(positive_examples, weights)

    total_weights = sum(weights)

    neg_ratio = float(weighted_neg / total_weights)
    pos_ratio = 1 - neg_ratio

    return -neg_ratio * log2(neg_ratio) - pos_ratio * log2(pos_ratio)


def information_gain(database, weights, attribute):
    '''
    Computes the information gain of `database` by splitting on `attribute`. The examples in the database are reweighted by `weights`.
    '''
    total_entropy = entropy(database.data, weights)
    gain = total_entropy
    attr_index = database.ordered_attributes.index(attribute)

    # Computes split entropy
    for attr_level in range(len(database.attributes[attribute])):
        filtered_indices = [index for index, ex in enumerate(database.data) if ex[attr_index] == attr_level]

        filtered_data = [database.data[i] for i in filtered_indices]
        filtered_weights = [weights[i] for i in filtered_indices]

        gain -= entropy(filtered_data, filtered_weights) * len(filtered_data) / len(database)

    return gain


class DecisionTree:
    '''
    A classifier. The decision tree induction algorithm is ID3 which recursively greedily maximizes information gain. An attribute only ever appears once on a given path.
    '''
    class Node:
        def __init__(self, database, weights, attributes, max_depth=None, depth=0):
            self.database = database
            def info_gain(x): return information_gain(database, weights, x)
            self.best_attribute = max(attributes, key=info_gain)
            other_attributes = [attr for attr in attributes if attr != self.best_attribute]

            debug_print('  ' * depth + self.best_attribute)
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
                    debug_print('  ' * depth + 'attr_value {} for {} => {}'.format(attr_value, self.best_attribute, prediction))
            else:
                self.predictions = {}
                attr_index = database.ordered_attributes.index(self.best_attribute)
                for attr_value in range(len(database.attributes[self.best_attribute])):
                    debug_print('  ' * depth + 'attr_value {} for {}'.format(attr_value, self.best_attribute))
                    self.predictions[attr_value] = DecisionTree.Node(database, weights, other_attributes, max_depth=max_depth, depth=depth + 1)

        def predict(self, example):
            # TODO don't store database, just store bestattrindex
            attr_index = self.database.ordered_attributes.index(self.best_attribute)
            prediction = self.predictions[example[attr_index]]
            if isinstance(prediction, DecisionTree.Node):
                return prediction.predict(example)
            else:
                return prediction

    def __init__(self, database, max_depth=None, weights=None):
        ''' Learns/creates the decision stump by selecting the attribute that maximizes information gain. '''
        if weights is None:
            weights = [1] * len(database.data)
        self.root = DecisionTree.Node(database, weights, database.ordered_attributes[:-1], max_depth=max_depth)

    def predict(self, example):
        ''' Returns the predicted class of `example` based on the attribute that maximized information gain at training time. '''
        self.root.predict(example)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='ParseARFF')

    parser = ArgumentParser()
    parser.add_argument('dataset_path', help='.arff file name')

    args = parser.parse_args()

    from parse_arff import Database
    database = Database()
    database.read_data(args.dataset_path)
    print(database)
    #
    # print("training")
    # tree = DecisionTree(database, max_depth=3)
    # print("prediction")
    # tree.predict(database.data[0])

    from evaluation import k_fold
    print(k_fold(lambda db: DecisionTree(db, max_depth = 5), database, 5))
