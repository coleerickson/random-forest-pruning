# A file to run Breiman's experiments
from glob import glob
from parse_arff import Database
from random_forest import RandomForest
from evaluation import evaluate_model

def log2(x):
    ''' Returns the base 2 logarithm of `x`. '''
    return log(x, 2)

def single_evaluation(dataset_fn):
    ntrees = 100
    # 1) Set aside a random 10% of the dataset
    db = Database()
    db.read_data(dataset_fn)
    train, test = next(db.k_fold(10))

    # 2a) Random forest, with F = 1, ntrees=100
    rf_a = lambda db: RandomForest(db,1,ntrees)
    # 2b) Random forest, wth F = int(log_2 (M) + 1)), ntrees=100
    rf_b = lambda db: RandomForest(db,int(log2(len(db.attributes)) + 1),ntrees)
    # 3) Test error = lower error between the two

    print('learning tree')
    error_a = evaluate_model(rf_a,train,test)
    print('learning tree2')
    error_b = evaluate_model(rf_b,train,test)
    return min(error_a,error_b)


def evaluate_dataset(dataset_fn):
    return sum(single_evaluation(dataset_fn) for _ in range(100)) / 100


if __name__ == '__main__':
    datasets = glob('./datasets/breast-cancer.arff')
    for dataset_fn in datasets:
        print(dataset_fn)
        print(evaluate_dataset(dataset_fn))
        print()
        # do something w i l d
