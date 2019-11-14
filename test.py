import pickle
import project_part2 as project_part2
import sys

DATA_DIR = "./" + sys.argv[1] + "/"

## Read the data sets...

### Read the Training Data
train_file = DATA_DIR + 'train.pickle'
train_mentions = pickle.load(open(train_file, 'rb'))

### Read the Training Labels...
train_label_file = DATA_DIR + 'train_labels.pickle'
train_labels = pickle.load(open(train_label_file, 'rb'))

### Read the Dev Data... (For Final Evaluation, we will replace it with the Test Data)
dev_file = DATA_DIR + 'dev.pickle'
dev_mentions = pickle.load(open(dev_file, 'rb'))

### Read the Parsed Entity Candidate Pages...
fname = DATA_DIR + 'parsed_candidate_entities.pickle'
parsed_entity_pages = pickle.load(open(fname, 'rb'))

### Read the Mention docs...
mens_docs_file = DATA_DIR + "men_docs.pickle"
men_docs = pickle.load(open(mens_docs_file, 'rb'))

## Result of the model...
result = project_part2.disambiguate_mentions(
    train_mentions, train_labels, dev_mentions, men_docs, parsed_entity_pages)

## Here, we print out sample result of the model for illustration...
for key in list(result)[:5]:
    print('KEY: {} \t VAL: {}'.format(key, result[key]))


## We will be using the following function to compute the accuracy...
def compute_accuracy(result, data_labels):
    assert set(list(result.keys())) - set(list(data_labels.keys())) == set()
    TP = 0.0
    for id_ in result.keys():
        if result[id_] == data_labels[id_]['label']:
            print("OK - %s: %s, %s" % (data_labels[id_]['mention'],
                                       data_labels[id_]['label'], result[id_]))
            TP += 1
        else:
            print(
                "NOT - %s: %s, %s" % (data_labels[id_]['mention'],
                                      data_labels[id_]['label'], result[id_]))
    assert len(result) == len(data_labels)
    return TP / len(result)


### Read the Dev Labels... (For Final Evaluation, we will replace it with the Test Data)
dev_label_file = DATA_DIR + 'dev_labels.pickle'
dev_labels = pickle.load(open(dev_label_file, 'rb'))

accuracy = compute_accuracy(result, dev_labels)
print("Accuracy = ", accuracy)
