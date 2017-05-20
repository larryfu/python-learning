#!/usr/bin/env python
# encoding: utf-8

import pickle
import transaction
import numpy as np
from datetime import datetime
from sqlalchemy.orm import aliased, join
from sqlalchemy import update
from zope.sqlalchemy import mark_changed
from sqlalchemy import bindparam

from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from ml_models.base import *
from haitao.models import DBSession, Category, Product, Brand

log = logging.getLogger(__name__)

ROOT_CATE_IDS = [867, 868, 874, 875]

def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return -1
            
def libsvm_pipeline_subset_train_test_process(ps, parent_id, reports):
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),\
                         ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)), ])
    
    sample = recursive_collect_sample(ps, parent_id)
    #pdb.set_trace()
    if len(sample) == 0: return
    sample = random_sample(sample)
    split_pos, train_docs, train_target, test_docs, test_target = split_train_test_sample_data(sample)
    _ = text_clf.fit(train_docs, train_target)
    predicted = text_clf.predict(test_docs)
    accuracy = np.mean(predicted == test_target)
    print accuracy
    print(metrics.classification_report(test_target, predicted))
    print(metrics.confusion_matrix(test_target, predicted))
    report = {
        'accuracy': accuracy,
        'prf1': metrics.classification_report(test_target, predicted),
        'cm': metrics.confusion_matrix(test_target, predicted)
    }
    reports.append(report)
        

def libsvm_pipeline_train_test_process():
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),\
                         ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)), ])
    samples = [collect_level_three_cate_sample()] #, collect_leaf_cate_sample()
    for sample in samples:
        sample = random_sample(sample)
        split_pos, train_docs, train_target, test_docs, test_target = split_train_test_sample_data(sample)
        _ = text_clf.fit(train_docs, train_target)
        predicted = text_clf.predict(test_docs)
        accuracy = np.mean(predicted == test_target)
        print accuracy
        print(metrics.classification_report(test_target, predicted))
        print(metrics.confusion_matrix(test_target, predicted))
        
def mnb_pipeline_train_test_process():
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()),])
    samples = [collect_level_three_cate_sample()] #, collect_leaf_cate_sample()
    for sample in samples:
        sample = random_sample(sample)
        split_pos, train_docs, train_target, test_docs, test_target = split_train_test_sample_data(sample)
        text_clf = text_clf.fit(train_docs, train_target)
        predicted = text_clf.predict(test_docs)
        accuracy = np.mean(predicted == test_target)
        print accuracy
        print(metrics.classification_report(test_target, predicted))
        print(metrics.confusion_matrix(test_target, predicted))
        
def mnb_train_test_process():
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    
    sample = collect_root_cate_sample()
    sample = random_sample(sample)
    split_pos, train_docs, train_target, test_docs, test_target = split_train_test_sample_data(sample)
    clf = train_sample_data(train_docs, train_target, count_vect, tfidf_transformer)
    test_predicted = test_product_categorization(clf, test_docs, count_vect, tfidf_transformer)
    accuracy = np.mean(test_predicted == test_target)
    return accuracy        

###########################################
### GRID SEARCHING FOR PARAMETER OPT    ###
###########################################
def libsvm_grid_search_pipeline_train_test_process(ps, parent_id):
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),\
                         ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)), ])
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5),}
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
        
    sample = recursive_collect_sample(ps, parent_id)
    if len(sample) == 0: return
    sample = random_sample(sample)
    split_pos, train_docs, train_target, test_docs, test_target = split_train_test_sample_data(sample)
    gs_clf = gs_clf.fit(train_docs, train_target)
    
    # save model into disk
    gs_clf_persistence = pickle.dumps(gs_clf)
    filename = path.join(path.dirname(path.abspath(__file__)), 'models/product_categorization/GsClf_ParentId_%s' % parent_id)
    with open(filename, 'w') as f:
        f.write(gs_clf_persistence)
    predicted = gs_clf.predict(test_docs)
    
    import itertools
    for d, t, p in itertools.izip(sample[split_pos:], test_target, predicted):
        if t != p: print(t, p, d)
    accuracy = np.mean(predicted == test_target)
    print accuracy
    print(metrics.classification_report(test_target, predicted))
    print(metrics.confusion_matrix(test_target, predicted))
    
    best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
    

###########################################
### ADD MORE FIELDS INTO FEATURE SPACE  ###
###########################################


###########################################
### LEARNING CURVE and VALIDATION CURVE ###
###########################################
from sklearn.learning_curve import validation_curve
from sklearn.learning_curve import learning_curve
from sklearn.svm import SVC
from sklearn.linear_model import Ridge

# train_sizes, train_scores, valid_scores = learning_curve(SVC(kernel='linear'), X, y, train_sizes=[50, 80, 110], cv=5)

###########################
###  HELPER FUNCTIONS   ###
###########################
def train_sample_data(train_docs, train_target, count_vect, tfidf_transformer):
    X_train_counts = count_vect.fit_transform(train_docs)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, train_target)
    return clf

def test_product_categorization(clf, test_docs, count_vect, tfidf_transformer): 
    X_test_counts = count_vect.transform(test_docs)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    test_predicted = clf.predict(X_test_tfidf)
    return test_predicted

def random_sample(sample):
    timestamp = (datetime.now() - datetime(1970, 1, 1)).total_seconds()
    np.random.seed(seed=int(timestamp))
    np.random.shuffle(sample)
    return sample
    
def split_train_test_sample_data(sample):
    split_pos = int(0.8 * len(sample))
    train_docs = np.array([e['data'] for e in sample[:split_pos]])
    train_target = np.array([e['target'] for e in sample[:split_pos]])
    test_docs = np.array([e['data'] for e in sample[split_pos:]])
    test_target = np.array([e['target'] for e in sample[split_pos:]])
    return split_pos, train_docs, train_target, test_docs, test_target
        
########################### 
### COLLECT SAMPLE DATA ###
###########################
def recursive_collect_sample(ps, parent_id):
    # extract related categories and products
    cates = DBSession.query(Category.id, Category.cate_id_path).filter(Category.parent_id==parent_id).all()
    
    # build feature sets
    product_cate_pairs, product_cate_set = [], set()
    for p in ps:
        idx = find_element_in_list(parent_id, p.cate_id_path)
        ### try to use cate_paths instead of subjects.
        if p.cate_paths and (idx != -1 or parent_id == 0): 
            # print(p.cate_paths)
            cate_path_str = " ".join(p.cate_paths.values())
            if cate_path_str in product_cate_set: continue
            product_cate_pair = {'data': cate_path_str, 'target': p.cate_id_path[idx + 1], 'pid': p.id, 'cate_id': p.category_id}
            product_cate_pairs.append(product_cate_pair)
            product_cate_set.add(cate_path_str)
    return product_cate_pairs
    
def collect_root_cate_sample(): 
    root_category = aliased(Category)
    ps = DBSession.query(Product.id, Product.subject, Product.attributes, Product.description, Product.cate_paths, Product.category_id,\
                         Brand.name.label('brand_name'), root_category.id.label('root_parent_id')).\
                   join(Category, Product.category).\
                   outerjoin(Brand, Product.brand).\
                   join(root_category, root_category.id == Category.root_parent_id).\
                   filter(root_category.id.in_(ROOT_CATE_IDS)).all()         
    product_cate_pairs = [] 
    for p in ps:
        if p.subject.get('en'):
            #,  p.description.get('en'), p.cate_paths.get('en'), p.brand_name or ''
            product_cate_pair = {'data': "%s" % (p.subject.get('en')), 
                                 'target': p.root_parent_id, 'pid': p.id, 'cate_id': p.category_id} 
            product_cate_pairs.append(product_cate_pair)
    return product_cate_pairs

def collect_level_two_cate_sample():
    root_category = aliased(Category)
    ps = DBSession.query(Product.id, Product.subject, Product.attributes, Product.description, Product.cate_paths, Product.category_id,\
                         Brand.name.label('brand_name'), Category.cate_id_path).\
                   join(Category, Product.category).\
                   outerjoin(Brand, Product.brand).\
                   join(root_category, root_category.id == Category.root_parent_id).\
                   filter(root_category.id.in_(ROOT_CATE_IDS)).all()
    product_cate_pairs = [] 
    for p in ps:
        if p.subject.get('en'):
            product_cate_pair = {'data': "%s" % (p.subject.get('en')), 
                                 'target': p.cate_id_path[1], 'pid': p.id, 'cate_id': p.category_id}
            product_cate_pairs.append(product_cate_pair)
    return product_cate_pairs

def collect_level_three_cate_sample(): 
    root_category = aliased(Category)
    ps = DBSession.query(Product.id, Product.subject, Product.category_id, Category.cate_id_path).\
                   join(Category, Product.category).\
                   join(root_category, root_category.id == Category.root_parent_id).\
                   filter(root_category.id.in_(ROOT_CATE_IDS)).all()  
    product_cate_pairs = [] 
    for p in ps:
        if p.subject.get('en'):
            product_cate_pair = {'data': p.subject.get('en'), 'target': p.cate_id_path[2], 'pid': p.id, 'cate_id': p.category_id}
            product_cate_pairs.append(product_cate_pair)
    return product_cate_pairs
    
def collect_leaf_cate_sample():
    root_category = aliased(Category)
    ps = DBSession.query(Product.id, Product.subject, Product.category_id).\
                   join(Category, Product.category).\
                   join(root_category, root_category.id == Category.root_parent_id).\
                   filter(root_category.id.in_(ROOT_CATE_IDS)).all()    
    product_cate_pairs = [] 
    for p in ps:
        if p.subject.get('en'):
            product_cate_pair = {'data': p.subject.get('en'), 'target': p.category_id, 'pid': p.id, 'cate_id': p.category_id}
            product_cate_pairs.append(product_cate_pair)
    return product_cate_pairs

def libsvm_pipeline_recursive_train_test_process(ps, reports, parent_id=0):
    print "########### %s ###########" % parent_id
    try:
        #libsvm_pipeline_subset_train_test_process(ps, parent_id, reports)
        libsvm_grid_search_pipeline_train_test_process(ps, parent_id)
    except ValueError as ex:
        log.error("parent_id=%s and %s" % (parent_id, ex.message))
        
    cates = DBSession.query(Category.id, Category.level).filter(Category.parent_id == parent_id).all()
    for cate in cates:
        if  cate.level <= 2:
            libsvm_pipeline_recursive_train_test_process(ps, reports, parent_id=cate.id)
    
def product_categorization_libsvm_prediction(parent_ids=[0]):
    for parent_id in parent_ids:
        # get the trained model firstly
        gs_clf_persistence = ""
        filename = path.join(path.dirname(path.abspath(__file__)), 'models/product_categorization/GsClf_ParentId_%s' % parent_id)
        with open(filename, 'r') as f:
            gs_clf_persistence = f.read()
        gs_clf = pickle.loads(gs_clf_persistence)
        
        # product list to be trained
        while True:
            ps = DBSession.query(Product.id, Product.subject).filter(Product.cateogry_labeled_by_hand == False).\
                           filter(Product.first_level_category_id_trained == False).limit(100).all()
            if not ps: break
            session = DBSession()
            update_data = []
            
            print('starting prediction ...')
            for p in ps:
                update_item = {'_id': p.id, '_first_level_category_id_trained': True, '_trained_first_level_category_id': None}
                if p.subject.get('en'): 
                    predicted_category_ids = gs_clf.predict([p.subject.get('en')])
                    update_item['_trained_first_level_category_id'] = predicted_category_ids[0]
                update_data.append(update_item)
            print('starting batch update...')
            update_category_id_stmt = update(Product).values({Product.trained_first_level_category_id: bindparam('_trained_first_level_category_id'),\
                                                              Product.first_level_category_id_trained: bindparam('_first_level_category_id_trained')}).\
                                                       where(Product.id == bindparam('_id'))
            session.execute(update_category_id_stmt, update_data)
            mark_changed(session)
            transaction.commit()
            
def product_categorization_level_two_libsvm_prediction(parent_ids=[0]):
    for parent_id in parent_ids:
        # get the trained model firstly
        gs_clf_persistence = ""
        filename = path.join(path.dirname(path.abspath(__file__)), 'models/product_categorization/GsClf_ParentId_%s' % parent_id)
        with open(filename, 'r') as f:
            gs_clf_persistence = f.read()
        gs_clf = pickle.loads(gs_clf_persistence)
        
        # product list to be trained
        while True:
            print('looping...')
            ps = DBSession.query(Product.id, Product.subject).filter(Product.cateogry_labeled_by_hand == False).\
                           filter(Product.trained_first_level_category_id == parent_id).filter(Product.second_level_category_id_trained == False).limit(10).all()
            if not ps: break
            session = DBSession()
            update_data = []
            print('starting prediction ...')
            for p in ps:
                update_item = {'_id': p.id, '_trained_second_level_category_id': None, '_second_level_category_id_trained': True}
                if p.subject.get('en'):
                    predicted_category_ids = gs_clf.predict([p.subject.get('en')])
                    update_item['_trained_second_level_category_id'] = predicted_category_ids[0]
                update_data.append(update_item) 
            print('starting level two batch update...')
            update_category_id_stmt = update(Product).values({Product.trained_second_level_category_id: bindparam('_trained_second_level_category_id'),\
                                                              Product.second_level_category_id_trained: bindparam('_second_level_category_id_trained')}).\
                                                       where(Product.id == bindparam('_id'))
            session.execute(update_category_id_stmt, update_data)
            mark_changed(session)
            transaction.commit()
            
def product_categorization_level_three_libsvm_prediction(parent_ids=[0]):
    for parent_id in parent_ids:
        try:
            # get the trained model firstly
            gs_clf_persistence = ""
            filename = path.join(path.dirname(path.abspath(__file__)), 'models/product_categorization/GsClf_ParentId_%s' % parent_id)
            with open(filename, 'r') as f:
                gs_clf_persistence = f.read()
            gs_clf = pickle.loads(gs_clf_persistence)
        
            # product list to be trained
            while True:
                print('looping...')
                ps = DBSession.query(Product.id, Product.subject).filter(Product.cateogry_labeled_by_hand == False).\
                               filter(Product.trained_second_level_category_id == parent_id).filter(Product.third_level_category_id_trained == False).limit(100).all()
                if not ps: break
                session = DBSession()
                update_data = []
                print('starting prediction ...')
                for p in ps:
                    update_item = {'_id': p.id, '_trained_third_level_category_id': None, '_third_level_category_id_trained': True}
                    if p.subject.get('en'):
                        predicted_category_ids = gs_clf.predict([p.subject.get('en')])
                        update_item['_trained_third_level_category_id'] = predicted_category_ids[0]
                    update_data.append(update_item) 
                print('starting level three batch update...')
                update_category_id_stmt = update(Product).values({Product.trained_third_level_category_id: bindparam('_trained_third_level_category_id'),\
                                                                  Product.third_level_category_id_trained: bindparam('_third_level_category_id_trained')}).\
                                                           where(Product.id == bindparam('_id'))
                session.execute(update_category_id_stmt, update_data)
                mark_changed(session)
                transaction.commit()
        except Exception as ex:
            log.error(ex.message)
            

def set_product_category_id():
    # level three category is leaf
    """
        UPDATE product
        SET category_id=product.trained_third_level_category_id
        WHERE product.trained_third_level_category_id IN
            (SELECT category.id AS category_id
             FROM category
             WHERE category.isleaf = true
             FOR UPDATE)
    """
    
    # level four category is leaf!!!
    leaf_category_subquery = DBSession.query(Category.parent_id, Category.id.label('category_id')).distinct(Category.parent_id).\
                                       order_by(Category.parent_id.desc(), Category.created_at.desc()).subquery()
    while True:
        ps = DBSession.query(Product.id, leaf_category_subquery.c.category_id).\
                       join(leaf_category_subquery, leaf_category_subquery.c.parent_id == Product.trained_third_level_category_id).\
                       filter(Product.trained_third_level_category_id != None).filter(Product.category_id == None).limit(1000).all()
        if not ps: break
        session = DBSession()
        update_data = []
        for p in ps:
            update_item = {'_id': p.id, '_category_id': p.category_id}
            update_data.append(update_item)            
        print('starting setting product category ids ...')               
        update_category_id_stmt = update(Product).values({Product.category_id: bindparam('_category_id')}).\
                                                  where(Product.id == bindparam('_id'))
        session.execute(update_category_id_stmt, update_data)
        mark_changed(session)
        transaction.commit()
 
def train_level_three_cates():
    all_level_two_cate_ids = []
    root_parent_ids = [867, 868, 869, 870, 871, 872, 873, 874, 875]
    for root_parent_id in root_parent_ids:
        cates = DBSession.query(Category.id).filter(Category.parent_id == root_parent_id).all()
        for cate in cates:
            all_level_two_cate_ids.append(cate.id)
    product_categorization_level_three_libsvm_prediction(all_level_two_cate_ids)   
    
if __name__ == '__main__':
    ### train
    reports = []
    ps = DBSession.query(Product.id, Product.subject, Product.cate_paths, Product.category_id, Category.cate_id_path).\
                   filter(Product.cateogry_labeled_by_hand == True).join(Category, Product.category).all()
    libsvm_pipeline_recursive_train_test_process(ps, reports, parent_id=0)
    
    
    ### prediction
    # product_categorization_libsvm_prediction()
    # product_categorization_level_two_libsvm_prediction([867, 868, 869, 870, 871, 872, 873, 874, 875])
    #train_level_three_cates()
    
    ### setting
    #set_product_category_id()