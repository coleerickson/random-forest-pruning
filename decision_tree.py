from math import log

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

    def __init__(self, database, weights):
        ''' Learns/creates the decision stump by selecting the attribute that maximizes information gain. '''
        self.database = database

        # Select the attribute that maximizes information gain on the training set
        attributes = database.ordered_attributes[:-1]

        def info_gain(x): return information_gain(database, weights, x)
        self.best_attribute = max(attributes, key=info_gain)

        # For each value of the best attribute, determine the majority class. In
        # `self.predictions`, map that attribute value to the majority class.
        self.predictions = {}
        attr_index = database.ordered_attributes.index(self.best_attribute)
        for attr_level in range(len(database.attributes[self.best_attribute])):
            filtered_indices = [index for index, ex in enumerate(database.data) if ex[attr_index] == attr_level]
            filtered_data = [database.data[i] for i in filtered_indices]
            filtered_weights = [weights[i] for i in filtered_indices]

            neg_examples = [ex[-1] == 0 for ex in filtered_data]

            weighted_neg = inner(neg_examples, filtered_weights)
            self.predictions[attr_level] = int(weighted_neg < (sum(filtered_weights) / 2))

        print(self.best_attribute)

    def predict(self, example):
        ''' Returns the predicted class of `example` based on the attribute that maximized information gain at training time. '''
        attr_index = self.database.ordered_attributes.index(self.best_attribute)
        return self.predictions[example[attr_index]]

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='ParseARFF')

    parser = ArgumentParser()
    parser.add_argument('dataset_path', help='.arff file name')

    args = parser.parse_args()

    from parse_arff import Database
    db = Database()
    db.read_data(args.dataset_path)
    print(db)
