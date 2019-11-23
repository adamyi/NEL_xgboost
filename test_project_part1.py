import pickle
import project_part1 as project_part1

fname = './Data/sample_documents.pickle'
documents = pickle.load(open(fname, "rb"))

print(documents)

## Step- 1. Construct the index...
index = project_part1.InvertedIndex()

index.index_documents(documents)
#index.index_documents({1: "Donald Trump, Donald Trump, Trump."})

## Test cases
Q = 'New York Times Trump travel'
DoE = {'New York Times': 0, 'New York': 1, 'New York City': 2}
doc_id = 3

## 2. Split the query...
query_splits = index.split_query(Q, DoE)
print(query_splits)

## 3. Compute the max-score...
result = index.max_score_query(query_splits, doc_id)

print(result)
