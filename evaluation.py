def hold_one_out(classifier, database):
    '''
    Returns the percentage accuracy of a hold-one-out validation on the
    entire database.
    '''
    ps = []
    for train, test in database.hold_one_out():
        ps.append(evaluate_model(classifier, train, test))

    return sum(ps) / len(ps)

def k_fold(classifier, database, k):
    '''
    Returns the percentage accuracy of a hold-one-out validation on the
    entire database.
    '''
    ps = []
    for train, test in database.k_fold(k):
        ps.append(evaluate_model(classifier, train, test))

    return sum(ps) / len(ps)

def evaluate_model(model, train, test):
    '''
    Creates a classifier and returns its accuracy (in [0, 1]) on the test data
    model -- The classifier constructor to be applied to the training set
    train -- The database of examples to train on
    test -- The database of examples to test on
    '''
    model_instance = model(train)

    num_correctly_classified = 0
    for example in test.data:
        predicted_class = model_instance.predict(example)
        true_class = example[-1]
        # print('predicted: {}, true: {}'.format(predicted_class, true_class))
        if predicted_class == true_class:
            num_correctly_classified += 1

    print('Correctly classified: {} / {}'.format(num_correctly_classified, len(test.data)))

    return num_correctly_classified / len(test.data)
