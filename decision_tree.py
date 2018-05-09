from collections import defaultdict, Counter
from math import log
from random import sample
from pprint import pformat
from utils import first, mode, log2, inner

DEBUG = False
if DEBUG:
    debug_print = print
else:
    debug_print = lambda _: None

    
def _get_class_from_example_with_weight(example_with_weight):
    return example_with_weight[0][-1]

def entropy(indices, examples, weights):
    '''
    Returns the entropy of `examples`. Entropy is defined in terms of the true class of the example. When counted, each of the `examples` is multiplied by the factor at the corresponding index in `weights`.
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

            # This state is useful for pruning
            self.prune_classes = Counter()
            for prune_class in database.attributes[database.ordered_attributes[-1]]:
                self.prune_classes[prune_class] = 0
            self.parent = parent
            self.indices = indices            

            # If no attributes remain, then we should create a leaf node and record the majority class of the associated examples
            if len(attributes) == 0:
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
                    self.predictions[attr_value] = DecisionTree.Node(database, indices, weights, [], max_depth, attribute_selector, depth=depth + 1, parent=self)

                else:
                    # Otherwise recursively create another node based off the examples with this attribute value
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

            return prediction

        def is_leaf(self):
            return self.best_attribute is None

        def leaves(self):
            # If we are at a leaf node, return
            if not isinstance(first(self.predictions.values()), DecisionTree.Node):
                return [self]
            else:
                leaves = []
                for subtree in self.predictions.values():
                    leaves.extend(subtree.leaves())
                return leaves

        def prune_classify(self, example):
            # TODO use a bestattrindex
            prune_class = example[-1]
            self.prune_classes[prune_class] += 1
            
            if not self.is_leaf():
                attr_index = self.database.ordered_attributes.index(self.best_attribute)
                attr_value = example[attr_index]
                child = self.predictions[attr_value]
                child.prune_classify(example)
                
        def prune(self):
            ''' Mutates the tree, performing reduced-error pruning. Uses the bottom-up algorithm described in Elomaa and Kaariainen 2001 (see Table 1 and section 2.4), extended to handle non-binary nominal-valued attributes. Returns the number of misclassifications'''            
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

                    # See section 2 of Elomaa et al.: "Neither is it obvious whether the training set or pruning set is used to decide the labels of the leaves that result from pruning." Here we make the same choice as Elomaa to use the pruning set.
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
        ''' Returns the predicted class of `example` based on the attribute that maximized information gain at training time. '''
        return self.root.predict(example)

    def prune(self, pruning_examples):
        for example in pruning_examples:
            self.root.prune_classify(example)
        self.root.prune()
     
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

 
    # from evaluation import k_fold
    # print(k_fold(lambda db: DecisionTree(db, max_depth=6), database, 10))

    from random import shuffle
    shuffle(database.data)
    
    # 10% of 100% is 10%
    etc, pruning = first(database.k_fold(6))
    # 11% of 90% is 10%
    training, testing = first(etc.k_fold(5))
    # 80, 10, 10 ! wow ! 
    
    from evaluation import evaluate_model
    t = RandomTree(training, 1)
    print(t.root.size())
    evaluate_model(lambda _: t, None, testing)
    t.prune(pruning.data)
    print(t.root.size())
    evaluate_model(lambda _: t, None, testing)
    
    # for leaf in RandomTree(database, 1, max_depth=4).leaves():
    #     print(leaf)
    # tree = RandomTree(database, 1, max_depth=4)
    # leaves = set(tree.leaves())
    # print(len(leaves))
    # from pprint import pprint
    # pprint([(i, str(leaf)) for i, leaf in enumerate(leaves)])
