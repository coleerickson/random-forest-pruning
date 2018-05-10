# Random Forest Implementation and Pruning Study

Nathan Andersen, Cole Erickson

Included in this repository are an implementation of random forests and the code used in conducting an experiment to determine the effect of pruning in random forests.

## Running

To run our experiment, run the following in bash:

```sh
$ python3 breiman_erickson_andersen.py
```

The program runs several trials in parallel. Each trial's output will be interleaved in stdout, so don't expect it to mean much. However, the final output will indicate the average error percentage across all the trials. The last line of output should read something like "After 25 trials, average error is pre=X% post=Y%." Here, X is the average error percentage of the forests before they were pruned and Y is the average error percentage afterward.

The program looks for datasets in the `./BreimanDatasets/` directory. To modify the parameters, edit the `trials` variable and the number of trees, which is hardcoded to be `100`.

You can also run a more comprehensive experiment that makes use of scikit-learn's implementation of random forests in `breiman.py`, which is hardcoded to consider the ARFF files in the `./EnsembleData` directory.

Both of these experiments consider forests with 100 trees, reporting the minimum error rate when the tree is trained using F=1 (completely random attribute selection) and an F logarithmic in the number of attributes.

## Using Our Code

Several of our modules could be used for further research. In the `decision_tree` module, the `DecisionTree` class implements the ID3 decision tree induction algorithm. `RandomTree` reuses most of `DecisionTree`, but considers a randomly sampled subset of the attributes at each node during induction. The `random_forest` module contains a `RandomForest` class.

Useful functions for evaluating models are available in the `evaluation` module.

Beware training trees to maximum depth and considering all the attributes at each node. Large trees and high-dimensional data can dramatically increase runtime, which is dominated by the information gain (and therefore also entropy) computations. We have done our best to optimize these computations, but anyone interested in performance should look elsewhere for an efficient implementation--or tackle the task of using FFI to call into C for these computations.

## What's Missing

This implementation lacks the ability to handle unknown data and real-valued attributes.