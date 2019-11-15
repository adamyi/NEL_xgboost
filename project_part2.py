import numpy as np
import xgboost as xgb

from collections import Counter
from itertools import combinations, chain
from math import log
import copy
import math
import spacy

nlp = spacy.load('en_core_web_sm')
np.set_printoptions(threshold=np.inf)

tf_similarity_cache = {}


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


def get_tf(tf, t, did):
    if t not in tf:
        return 0
    if did not in tf[t]:
        return 0
    return tf[t][did]


def get_df(tf, t):
    if t not in tf:
        return 0
    return len(tf[t])


def get_idf(idf, t):
    if t not in idf:
        return 0
    return idf[t]


def list_to_map(lst):
    return {k: v for v, k in enumerate(list(set(lst)))}


def extract_tf(tf, idf, did):
    ret = {}
    for k, v in tf.items():
        if did in v:
            ret[k] = v[did] * idf[k]
    return ret


def tf_similarity(tf1, idf1, did1, tf2, idf2, did2):
    if (did1, did2) in tf_similarity_cache:
        return tf_similarity_cache[(did1, did2)]
    etf1 = extract_tf(tf1, idf1, did1)
    etf2 = extract_tf(tf2, idf2, did2)
    allkeys = list(set(list(etf1.keys()) + list(etf2.keys())))
    keymap = {v: k for k, v in enumerate(allkeys)}
    vec1 = np.zeros(len(allkeys))
    vec2 = np.zeros(len(allkeys))
    for k, v in etf1.items():
        vec1[keymap[k]] = v
    for k, v in etf2.items():
        vec2[keymap[k]] = v
    #print(vec1)
    #print(vec2)
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    cos = dot / (norm1 * norm2)
    tf_similarity_cache[(did1, did2)] = cos
    return cos


class InvertedIndex:
    def __init__(self):
        self.tf = {}
        self.tf_entities = {}
        self.tf_norm = {}

        self.idf = {}

        self.pos_set = []
        self.ent_set = []

    def index_documents(self, documents):
        for did, doc in documents.items():
            add_tf(did, compute_tf([v[2] for v in doc]), self.tf)
            add_tf(did, compute_tf([v[4] for v in doc]), self.tf_entities)
            self.pos_set.extend([v[3] for v in doc])
            self.ent_set.extend([v[4] for v in doc])
        self.tf_norm = calc_norm_tf(self.tf, lambda x: 1.0 + log(1.0 + log(x)))
        self.idf = calc_idf(self.tf, len(documents))
        self.pos_set = list_to_map(self.pos_set)
        self.ent_set = list_to_map(self.ent_set)


def transform_data(features, groups, labels=None):
    xgb_data = xgb.DMatrix(data=features, label=labels)
    xgb_data.set_group(groups)
    return xgb_data


def one_hot_encoding(enummap, enumvalues):
    ret = np.zeros(len(enummap))
    for v in enumvalues:
        if v in enummap:
            ret[enummap[v]] += 1
    return ret


def gen_feature_space(mentions, men_docs_nlp, tfidx, men_tfidx):
    feature_space = []
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    for k, v in mentions.items():
        nlpmen = men_docs_nlp[mentions[k]['doc_title']]
        span = nlpmen.char_span(mentions[k]['offset'],
                                mentions[k]['offset'] + mentions[k]['length'])
        if span is None:
            for t in nlpmen:
                if t.idx >= mentions[k]['offset']:
                    span = nlpmen[t.i:t.i + 1]
        tokens = [t for t in span]
        poss = [t.pos_ for t in tokens]
        entities = [t for t in span.ents]
        ents = [t.label_ for t in entities]
        sid = tokens[0].i
        eid = tokens[-1].i
        sent = tokens[0].sent
        present = nlpmen[max([sent.start - 1, 0])].sent
        shared_feature_vector = []
        shared_feature_vector.extend(one_hot_encoding(tfidx.pos_set, poss))
        shared_feature_vector.extend(one_hot_encoding(tfidx.ent_set, ents))
        for candidate in v["candidate_entities"]:
            tf = max([0] +
                     [get_tf(tfidx.tf, t.lemma_, candidate) for t in tokens])
            df = max([0] + [get_df(tfidx.tf, t.lemma_) for t in tokens])
            ttfidf = sum([
                calc_tf_idf(tfidx.tf_norm, tfidx.idf, t.lemma_, candidate)
                for t in tokens
            ])
            etfidf = sum([
                calc_tf_idf(tfidx.tf_norm, tfidx.idf, t.lemma_, candidate)
                for t in entities
            ])
            #atf = sum([
            #    math.sqrt(1.0 - min([abs(i - sid), abs(i - eid)]) / len(nlpmen)
            #              ) * get_tf(tfidx.tf_norm, t.lemma_, candidate) *
            #    get_idf(tfidx.idf, t.lemma_) * 2 for i, t in enumerate(nlpmen)
            #    if not (sid <= i <= eid)
            #])
            atf = sum([
                calc_tf_idf(tfidx.tf_norm, tfidx.idf, t.lemma_, candidate)
                for t in sent
            ])
            etf = sum(
                [(1.0 -
                  min([abs(t.idx - sid), abs(t.idx - eid)]) / len(nlpmen)) *
                 get_tf(tfidx.tf_entities, t.ent_type_, candidate)
                 for t in nlpmen])
            title = [t.lemma_ for t in tokenizer(candidate.replace('_', ' '))]
            title_tfidf = sum([
                get_idf(tfidx.idf, t.lemma_) for t in tokens
                if t.lemma_ in title
            ])
            title_rtfidf = sum([
                calc_tf_idf(men_tfidx.tf_norm, men_tfidx.idf, t,
                            mentions[k]['doc_title']) for t in title
            ])
            tfs = tf_similarity(tfidx.tf_norm, tfidx.idf, candidate,
                                men_tfidx.tf, men_tfidx.idf,
                                mentions[k]['doc_title'])
            n_nums = len([c for c in mentions[k]['mention'] if c.isdigit()])
            n_nums_2 = len([c for c in candidate if c.isdigit()])
            all_caps = int(mentions[k]['mention'].isupper())
            n_caps = len([c for c in mentions[k]['mention'] if c.isupper()])
            #idf = min([10] + [get_idf(tfidx.idf, t.lemma_) for t in tokens])
            feature_vector = [
                n_nums,
                n_nums_2,
                all_caps,
                n_caps,
                #tf,
                #df,
                ttfidf,
                etfidf,
                #etf,
                atf,
                title_tfidf,
                title_rtfidf,
                tfs
            ]
            # feature_vector = [title_rtfidf, title_tfidf, atf, tf, df, ttfidf]
            # feature_vector.extend(shared_feature_vector)
            feature_space.append(feature_vector)
    return np.array(feature_space)


def gen_train_labels(train_mentions, train_labels):
    train_label = []
    for k, v in train_mentions.items():
        label = train_labels[k]['label']
        for candidate in v["candidate_entities"]:
            if label == candidate:
                train_label.append(1)
            else:
                train_label.append(0)
    return np.array(train_label)


def gen_train_groups(train_mentions):
    train_groups = []
    for k, v in train_mentions.items():
        train_groups.append(len(v["candidate_entities"]))
    return np.array(train_groups)


def disambiguate_mentions(train_mentions, train_labels, dev_mentions, men_docs,
                          parsed_entity_pages):
    # pos_set = set([v[3] for v in parsed_entity_pages])
    # ent_set = set([v[4] for v in parsed_entity_pages])
    # print(pos_set)
    # print(ent_set)
    print("Building inverted index...")
    index = InvertedIndex()
    index.index_documents(parsed_entity_pages)
    print(index.pos_set)
    print(index.ent_set)
    print(len(index.pos_set))
    print(len(index.ent_set))
    print("Running SpaCy on men_docs corpus...")
    men_docs_nlp = {k: nlp(v) for k, v in men_docs.items()}
    print("Building inverted index...")
    mindex = InvertedIndex()
    mindex.index_documents({
        docid: [(v.idx, v.text, v.lemma_, v.pos_, v.ent_type_) for v in doc]
        for docid, doc in men_docs_nlp.items()
    })
    print("Generating feature space...")

    train_data = gen_feature_space(train_mentions, men_docs_nlp, index, mindex)
    train_label = gen_train_labels(train_mentions, train_labels)

    print("train_data shape:", train_data.shape)
    print("train_label shape:", train_label.shape)
    # print(train_data)
    print(train_label[:5])
    print(sum(train_label))

    ## Form Groups...

    #idxs = np.where(train_label == 1)[0]
    #train_groups = np.append(np.delete(idxs, 0), len(train_label)) - idxs
    train_groups = gen_train_groups(train_mentions)
    print(len(train_groups))
    print(sum(train_groups))

    xgboost_train = transform_data(train_data, train_groups, train_label)

    ## Parameters for XGBoost, you can fine-tune these parameters according to your settings...

    param = {
        'max_depth': 5,
        'eta': 0.05,
        'objective': 'rank:pairwise',
        'min_child_weight': 0.01,
        'n_estimators': 5000,
        'lambda': 100
    }

    param = {
        'learning_rate': 0.1,
        'n_estimators': 200,
        'max_depth': 5,
        'min_child_weight': 1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'rank:pairwise',
    }

    print("Training...")
    ## Train the classifier...
    classifier = xgb.train(param, xgboost_train, num_boost_round=4900)
    print("Training complete.")

    print("Parsing eval data...")
    eval_data = gen_feature_space(dev_mentions, men_docs_nlp, index, mindex)
    eval_groups = gen_train_groups(dev_mentions)

    xgboost_test = transform_data(eval_data, eval_groups)

    print("Evaluating...")
    preds = classifier.predict(xgboost_test)

    print("Generating result...")
    ret = {}
    idx = 0
    it = iter(dev_mentions.items())
    for iter_, group in enumerate(eval_groups):
        grouppreds = preds[idx:idx + group]
        data = next(it)
        ret[data[0]] = data[1]['candidate_entities'][np.argmax(grouppreds)]
        idx += group
    return ret
