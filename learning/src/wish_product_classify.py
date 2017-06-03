import pickle
import transaction
import numpy as np
from datetime import datetime
from sqlalchemy.orm import aliased, join
from sqlalchemy import update
from sqlalchemy import bindparam

from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import os
import os.path


def read_product(rootdir):
    # rootdir = "/opt/data/data/product-classify/train"
    # rootdir = "/opt/data/data/test"
    nameMap = {}
    labels = []
    data = []
    i = 0
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            if (filename.endswith(".txt")):
                full_path = os.path.join(parent, filename)
                cate = filename.split(".")[0]
                print "path:" + full_path + ",cate:" + cate
                text = open(full_path).read()
                texts = text.split('\n')
                for line in texts:
                    array = line.split("\t")
                    array.remove(array[0])
                    data += [str.join(" ", array)]
                    labels += [i]
                nameMap[i] = cate
                i += 1
    return data, labels, nameMap


def split_train_test(data, label):
    split_pos = int(0.8 * len(data))
    train_docs = data[:split_pos]
    train_target = label[:split_pos]
    test_docs = data[split_pos:]
    test_target = label[split_pos:]
    return train_docs, train_target, test_docs, test_target


def generate_model(train_data, tran_label):
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
    text_clf = text_clf.fit(train_data, tran_label)
    return text_clf


def text_classify():
    train_data, train_label, name_map = read_product("/opt/data/data/product-classify/second/test")
    test_data, test_label, name_map2 = read_product("/opt/data/data/product-classify/second/small-test")
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
    text_clf = text_clf.fit(train_data, train_label)
    predicted = text_clf.predict(test_data)
    print np.mean(predicted == test_label)

    # tfidf_transformer.fit_transform(count_vect.fit_transform(data))
    # X_train_counts = count_vect.fit_transform(train_data)
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # clf = MultinomialNB().fit(X_train_tfidf, train_label)
    # part_train = train_data[0:1000]
    # test = tfidf_transformer.fit_transform(count_vect.fit_transform(part_train))
    # result = clf.predict(test)
    # label.
    print ""


def load_second_cate(traindir, second_first_map):
    second_map = {}
    for parent, dirnames, filenames in os.walk(traindir):
        for dirname in dirnames:
            data, label, name_map = read_product(os.path.join(parent, dirname))
            model = generate_model(data, label)
            second_cate = {}
            second_cate["model"] = model
            second_cate["name_map"] = name_map
            second_map[dirname] = second_cate
            for i in name_map:
                second_first_map[name_map[i]] = dirname
    return second_map


def cate_clf():
    first_train_dir = "/opt/data/data/product-classify/train"
    first_test_dir = "/opt/data/data/product-classify/test"
    second_train_dir = "/opt/data/data/product-classify/second/train"
    second_test_dir = "/opt/data/data/product-classify/second/small-test"
    train_data, train_label, name_map = read_product(first_train_dir)
    test_data, test_label, name_map2 = read_product(first_test_dir)
    text_clf = generate_model(train_data, train_label)
    second_test_data, second_test_label, name_map3 = read_product(second_test_dir)
    second_first_map = {}
    second_map = load_second_cate("/opt/data/data/product-classify/second/classified", second_first_map)
    result = text_clf.predict(second_test_data)
    actual_first = []
    correct = 0
    for i in range(0, len(second_test_label)):
        second = name_map3[second_test_label[i]]
        first = second_first_map[second]
        cal_first = name_map[result[i]]
        if (first == cal_first):
            correct += 1
            # actual_first.append(first)
    print correct

    print "total:" + str(len(result))
    second_correct = 0
    for i in range(0, len(result)):
        print "predict:" + str(i)
        text = second_test_data[i]
        first_cate = name_map[result[i]]
        second_model = second_map[first_cate]
        second_result = second_model["model"].predict([text])
        second_cate = second_model["name_map"][second_result[0]]
        real_second_cate = name_map3[second_test_label[i]]
        if (second_cate == real_second_cate):
            second_correct += 1

    print second_correct
    #print np.mean(result_tag == second_test_lable_name)


cate_clf()
