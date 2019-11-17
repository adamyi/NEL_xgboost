import numpy as np
import xgboost as xgb

from collections import Counter
from itertools import combinations, chain
from math import log
import copy
import spacy

nlp = spacy.load('en_core_web_sm')
np.set_printoptions(threshold=np.inf)

tf_similarity_cache = {}


def norm_ent_type(lemma, et):
    if lemma.isdigit():
        return "NUM"
    return et.split('-')[-1]


def compute_K(dl, avdl):
    k1 = 1.2
    k2 = 100
    b = 0.75
    R = 0.0
    return k1 * ((1 - b) + b * (float(dl) / float(avdl)))


def score_BM25(n, f, qf, r, N, dl, avdl):
    k1 = 1.2
    k2 = 100
    b = 0.75
    R = 0.0
    K = compute_K(dl, avdl)
    first = log(
        ((r + 0.5) / (R - r + 0.5)) / ((n - r + 0.5) / (N - n - R + r + 0.5)))
    second = ((k1 + 1) * f) / (K + f)
    third = ((k2 + 1) * qf) / (k2 + qf)
    return first * second * third


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
        self.tf_norm = {}
        self.entities = {}

        self.idf = {}

        self.pos_set = []
        self.ent_set = []
        self.avglen = 0
        self.doclen = {}

    def index_documents(self, documents):
        for did, doc in documents.items():
            self.doclen[did] = len(doc)
            self.avglen += len(doc)
            self.pos_set.extend([v[3] for v in doc])
            self.ent_set.extend([norm_ent_type(v[2], v[4]) for v in doc])
        self.pos_set = list_to_map(self.pos_set)
        self.ent_set = list_to_map(list(set(self.ent_set) - set(["O"])))
        self.avglen /= len(documents)
        for ent_type in self.ent_set.keys():
            self.entities[ent_type] = InvertedIndex()
        for did, doc in documents.items():
            add_tf(did, compute_tf([v[2] for v in doc]), self.tf)
            for ent_type in self.ent_set.keys():
                add_tf(
                    did,
                    compute_tf([
                        v[2] for v in doc
                        if norm_ent_type(v[2], v[4]) == ent_type
                    ]), self.entities[ent_type].tf)
        self.tf_norm = calc_norm_tf(self.tf, lambda x: 1.0 + log(1.0 + log(x)))
        self.idf = calc_idf(self.tf, len(documents))
        for k, v in self.entities.items():
            v.tf_norm = calc_norm_tf(v.tf, lambda x: 1.0 + log(1.0 + log(x)))
            v.idf = calc_idf(v.tf, len(documents))


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
        nxtsent = nlpmen[min([sent.end + 1, len(sent) - 1])].sent
        sents = list(set([present, sent, nxtsent]))
        proxnouns = []
        allnouns = []
        for s in sents:
            for t in s:
                if t.pos_ in ["PROPN", "NOUN", "NUM"]:
                    proxnouns.append(t)
        for t in nlpmen:
            if t.pos_ in ["PROPN", "NOUN", "NUM"]:
                allnouns.append(t)
        #shared_feature_vector = []
        #shared_feature_vector.extend(one_hot_encoding(tfidx.pos_set, poss))
        #shared_feature_vector.extend(one_hot_encoding(tfidx.ent_set, ents))
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
            stf = sum([
                calc_tf_idf(tfidx.tf_norm, tfidx.idf, t.lemma_, candidate)
                for t in sent
            ])
            ntf = sum([
                calc_tf_idf(tfidx.tf_norm, tfidx.idf, t.lemma_, candidate)
                for t in proxnouns
            ])
            antf = sum(
                [(1.0 - min([abs(t.i - sid), abs(t.i - eid)]) / len(nlpmen)) *
                 calc_tf_idf(tfidx.tf_norm, tfidx.idf, t.lemma_, candidate)
                 for t in allnouns])
            bm25 = sum([
                score_BM25(
                    n=len(tfidx.tf[t.lemma_]),
                    f=get_tf(tfidx.tf, t.lemma_, candidate),
                    qf=1,
                    r=0,
                    N=len(tfidx.doclen),
                    dl=tfidx.doclen[candidate],
                    avdl=tfidx.avglen) for t in sent if t.lemma_ in tfidx.tf
            ])
            atf = sum(
                [(1.0 - min([abs(t.i - sid), abs(t.i - eid)]) / len(nlpmen)) *
                 calc_tf_idf(tfidx.tf_norm, tfidx.idf, t.lemma_, candidate)
                 for t in nlpmen])
            atf_entities = []
            for ent_type, ent_idx in tfidx.entities.items():
                if ent_type not in men_tfidx.entities:
                    atf_entities.append(0)
                else:
                    atf_entities.append(
                        sum([
                            (1.0 - min([abs(t.i - sid),
                                        abs(t.i - eid)]) / len(nlpmen)) *
                            calc_tf_idf(ent_idx.tf_norm, ent_idx.idf, t.lemma_,
                                        candidate) for t in nlpmen
                            if norm_ent_type(t.lemma_, t.ent_type_) == ent_type
                        ]))
            #etf = sum(
            #    [(1.0 -
            #      min([abs(t.idx - sid), abs(t.idx - eid)]) / len(nlpmen)) *
            #     get_tf(tfidx.tf_entities, t.ent_type_, candidate)
            #     for t in nlpmen])
            title_tokens = nlp(candidate.replace('_', ' '))
            title = [t.lemma_ for t in title_tokens]
            title_root = [t.lemma_ for t in title_tokens if t.dep_ == "ROOT"]
            title_tfidf = sum([
                get_idf(tfidx.idf, t.lemma_) for t in tokens
                if t.lemma_ in title
            ])
            title_rtfidf = sum([
                calc_tf_idf(men_tfidx.tf_norm, men_tfidx.idf, t,
                            mentions[k]['doc_title']) for t in title
            ])
            root_title_tfidf = sum([
                calc_tf_idf(men_tfidx.tf_norm, men_tfidx.idf, t,
                            mentions[k]['doc_title']) for t in title_root
            ])
            tfs = tf_similarity(tfidx.tf_norm, tfidx.idf, candidate,
                                men_tfidx.tf, men_tfidx.idf,
                                mentions[k]['doc_title'])
            tbm25 = sum([
                score_BM25(
                    n=len(men_tfidx.tf[t]),
                    f=get_tf(men_tfidx.tf, t, mentions[k]['doc_title']),
                    qf=1,
                    r=0,
                    N=len(men_tfidx.doclen),
                    dl=men_tfidx.doclen[mentions[k]['doc_title']],
                    avdl=men_tfidx.avglen) for t in title if t in men_tfidx.tf
            ])
            n_nums = len([c for c in mentions[k]['mention'] if c.isdigit()])
            n_nums_2 = len([c for c in candidate if c.isdigit()])
            all_caps = int(mentions[k]['mention'].isupper())
            n_caps = len([c for c in mentions[k]['mention'] if c.isupper()])
            #idf = min([10] + [get_idf(tfidx.idf, t.lemma_) for t in tokens])
            match_words = 0
            for i in range(min([len(tokens), len(title)])):
                if tokens[i].lemma_.lower() == title[i].lower():
                    match_words += get_idf(tfidx.idf, title[i])
            match_words /= len(tokens)
            all_match = int(" ".join(title).lower() == " ".join(
                [t.lemma_.lower() for t in tokens]))
            len_mention = len(tokens)
            len_title = len(title)
            feature_vector = [
                n_nums,
                n_nums_2,
                len_mention,
                len_title,
                all_caps,
                n_caps,
                #tf,
                #df,
                ttfidf,
                etfidf,
                #etf,
                atf,
                stf,
                ntf,
                antf,
                bm25,
                tbm25,
                title_tfidf,
                title_rtfidf,
                root_title_tfidf,
                tfs,
                match_words,
                all_match
            ]
            feature_vector.extend(atf_entities)
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
        docid: [(v.idx, v.text, v.lemma_, v.pos_, v.ent_type_)
                for v in doc]  #if not v.is_stop and not v.is_punct]
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
        'n_estimators': 50,
        'max_depth': 3,
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
    print(classifier.get_score(importance_type='gain'))

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
