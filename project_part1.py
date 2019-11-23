# Adam Yi <z5231521@cse.unsw.edu.au>

from collections import Counter
from itertools import combinations, chain
from math import log
import spacy
import copy

nlp = spacy.load('en_core_web_sm')

LAMBDA = 0.4


def tokenize(doc):
    return doc


def entitize(doc):
    return list(doc.ents)


def compute_tf(collection, counter=None):
    if counter is None:
        counter = Counter()
    for w in collection:
        counter[w] += 1
    return counter


def add_tf(did, tf, idx):
    for c in tf:
        if c not in idx:
            idx[c] = {}
        idx[c][did] = tf[c]


def calc_norm_tf(idx, func):
    res = {}
    for c, v in idx.items():
        res[c] = {}
        for t, f in v.items():
            res[c][t] = func(f)
    return res


def calc_idf(tf, docs):
    ret = {}
    for c, v in tf.items():
        ret[c] = 1.0 + log(docs / (1.0 + len(v)))
    return ret


def calc_tf_idf(tf, idf, t, did):
    if t not in tf:
        return 0
    if did not in tf[t]:
        return 0
    if t not in idf:
        return 0
    return tf[t][did] * idf[t]


class InvertedIndex:
    def __init__(self):
        self.tf_tokens = {}
        self.tf_entities = {}
        self.tf_norm_tokens = {}
        self.tf_norm_entities = {}

        self.idf_tokens = {}
        self.idf_entities = {}

    def index_documents(self, documents):
        for did, content in documents.items():
            doc = nlp(content)
            ents = list(map(lambda x: x.text, doc.ents))
            ents_filtermap = set(
                map(lambda x: (x.start_char, x.end_char - x.start_char),
                    filter(lambda x: len(x) == 1, doc.ents)))
            tokens = list(
                map(
                    lambda x: x.text,
                    filter(
                        lambda x: not x.is_stop and not x.is_punct and (
                            x.idx, len(x)) not in ents_filtermap, doc)))
            add_tf(did, compute_tf(tokens), self.tf_tokens)
            add_tf(did, compute_tf(ents), self.tf_entities)
        # print(self.tf_tokens)
        # print(self.tf_entities)
        self.tf_norm_tokens = calc_norm_tf(
            self.tf_tokens, lambda x: 1.0 + log(1.0 + log(x)))
        self.tf_norm_entities = calc_norm_tf(
            self.tf_entities, lambda x: 1.0 + log(x))
        self.idf_tokens = calc_idf(self.tf_tokens, len(documents))
        self.idf_entities = calc_idf(self.tf_entities, len(documents))

    def split_query(self, Q, DoE):
        ret = []
        knownentities = DoE.keys()
        tokens = list(filter(lambda x: x != "", Q.split(" ")))
        tc = compute_tf(tokens)
        pe = []
        tids = range(len(tokens))
        for i in range(1, len(tokens) + 1):
            pe += list(
                filter(lambda x: " ".join(x) in knownentities,
                       combinations(tokens, i)))
        pe = list(set(pe))
        for i in range(len(pe) + 1):
            es = combinations(pe, i)
            for j in es:
                c = compute_tf(chain.from_iterable(j))
                cont = True
                tcc = copy.copy(tc)
                tcc.subtract(c)
                if tcc.most_common()[-1][1] >= 0:
                    ret.append((list(map(lambda x: " ".join(x), j)),
                                list(tcc.elements())))
        return ret

    def max_score_query(self, query_splits, doc_id):
        ret = 0
        chosen = {"tokens": [], "entities": []}
        for split in query_splits:
            s1 = sum(
                map(
                    lambda x: calc_tf_idf(self.tf_norm_entities, self.
                                          idf_entities, x, doc_id), split[0]))
            s2 = sum(
                map(
                    lambda x: calc_tf_idf(self.tf_norm_tokens, self.idf_tokens,
                                          x, doc_id), split[1]))
            score = s1 + LAMBDA * s2
            # print(split[0], s1, split[1], s2, score)
            if score > ret:
                ret = score
                chosen = {"tokens": split[1], "entities": split[0]}
        return (ret, chosen)
        ## Output should be a tuple (max_score, {'tokens': [...], 'entities': [...]})
