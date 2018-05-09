# A file to run Breiman's experiments
from glob import glob
from parse_arff import Database
from math import log
#from random_forest import RandomForest
#from evaluation import evaluate_model
from sklearn.ensemble import RandomForestClassifier
from functools import partial

def log2(x):
    ''' Returns the base 2 logarithm of `x`. '''
    return log(x, 2)



def single_evaluation(dataset_fn,impurity_split):
    ntrees = 100
#    print('iter')
    # 1) Set aside a random 10% of the dataset
    db = Database()
    db.read_data(dataset_fn)
    train, test = next(db.k_fold(10))

    # 2a) Random forest, with F = 1, ntrees=100

    train_features = [ex[:-1] for ex in train.data]
    train_classes = [ex[-1] for ex in train.data]

    test_features = [ex[:-1] for ex in test.data]
    test_classes = [ex[-1] for ex in test.data]

    k = int(log2(len(db.ordered_attributes)- 1) + 1)
    rf = partial(RandomForestClassifier,n_estimators=ntrees,min_impurity_decrease=impurity_split)

    rf_a = rf(max_features=1)
    rf_a.fit(train_features,train_classes)
    error_a = 1 - rf_a.score(test_features,test_classes)

    rf_b = rf(max_features=k)
    rf_b.fit(train_features,train_classes)
    error_b = 1 - rf_b.score(test_features,test_classes)

    return min(error_a,error_b)


def evaluate_dataset(dataset_fn,impurity_split):
    n_iters = 25
    return [single_evaluation(dataset_fn,impurity_split) for _ in range(n_iters)]


if __name__ == '__main__':
    datasets = glob('./EnsembleData/*.arff')
    for dataset_fn in datasets:
        print(dataset_fn)
#        all_errors = []
#        for e in range(10):
#            print(e)
#            all_errors.append(evaluate_dataset(dataset_fn,e/60))

        all_errors = [evaluate_dataset(dataset_fn,e/60) for e in range(10)]
        for index,errors in enumerate(all_errors):
            print(index/60,sum(errors)/len(errors),'a')
        print()
        # do something w i l d
