#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from math import log

from pyspark.sql.types import Row
from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


# PYSPARK_DRIVER_PYTHON=ipython pyspark
# PYSPARK_DRIVER_PYTHON=jupyter-notebook pyspark
"""
spark-submit \
    --master yarn \
    --num-executors 100 \
    --executor-memory 4G \
    --executor-cores 4 \
    --conf spark.default.parallelism=1000 \
    dsp/script/ctr/ctr_predict_model_train.py p p
""" 

def exectue_model_training(fit_feature_pipeline_model, to_train_model):
    subfix = "_finance" #"_without_placement_app"
    warehouse_location = "/user/hive/warehouse" 
    spark = SparkSession\
            .builder\
            .appName("CTRPredictor")\
            .config("spark.sql.warehouse.dir", warehouse_location)\
            .enableHiveSupport()\
            .getOrCreate() 

    # Load training data
    df = read_sample_from_hive(spark)
    print(df.take(5))

    # fix missing values
    df = handle_missing_values(df)
    print(df.count())

    # feature transform
    feature_model = get_feature_model(spark, df, subfix, fit_feature_pipeline_model)
    train_df, test_df = split_df_after_feature_transform(feature_model, df)
    print(train_df.show(10, False))

    # train model and print out metrics
    lr_model = get_trained_model(spark, train_df, subfix, to_train_model)
    predict_and_save_results(spark, lr_model, test_df, subfix)

    # close spark
    spark.stop()


def get_trained_model(spark, train_df, subfix, to_train_model=False):
    model_file = "hdfs://nameservice1/user/luwl/dsp/model/ctr-lr-model%s" % subfix
    if not to_train_model:
        print("LOADING THE TRAINED MODEL............")
        lr_model = LogisticRegressionModel.load(model_file)
        save_lr_model_weights(spark, lr_model, subfix)
        return lr_model
    else:
        print("TRY TO TRAIN THE MODEL............")
        # LogisticRegressionWithLBFGS
        lr = LogisticRegression() #weightCol="weight"
        paramGrid = ParamGridBuilder().addGrid(lr.elasticNetParam, [0.0]).addGrid(lr.regParam, [0.0]).addGrid(lr.maxIter, [100]).build()
        evaluator = BinaryClassificationEvaluator()
        cv = CrossValidator().setEstimator(lr).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(2)
        cv_model = cv.fit(train_df)
        lr_model = cv_model.bestModel

        # delete previous dir and save model into disk
        os.system("hadoop fs -rm -r %s" % model_file)
        lr_model.save(model_file)
        save_lr_model_weights(spark, lr_model, subfix)
        return lr_model


def get_feature_model(spark, df, subfix, fit_feature_pipeline_model=False):
    feature_list = [c for c in df.columns if c not in ["label", "weight"]]
    feature_pipeline_model_file = "hdfs://nameservice1/user/luwl/dsp/model/ctr-pipeline-model%s" % subfix
    if not fit_feature_pipeline_model:
        print("TRY TO LOADING FEATURES PIPELINE MODEL...")
        feature_pipeline_model = PipelineModel.load(feature_pipeline_model_file)
        save_feature_onehot_encode(spark, feature_pipeline_model, subfix, feature_list)
        return feature_pipeline_model
    else:
        print("TRY TO FITTING FEATURES PIPELINE MODEL...")
        encode_feature_list, feature_stage_list = [], []
        for xi in feature_list:
            # Use StringIndexer and OneHotEncoder to convert categorical variables to 0/1
            catIndexer = StringIndexer(inputCol=xi, outputCol="idx_%s" % xi, handleInvalid="skip")
            catEncoder = OneHotEncoder(inputCol="idx_%s" % xi, outputCol="enc_%s" % xi)
            feature_stage_list.append(catIndexer)
            feature_stage_list.append(catEncoder)
            encode_feature_list.append("enc_%s" % xi)
        va = VectorAssembler(inputCols=encode_feature_list, outputCol="features")
        feature_stage_list.append(va)
        print(feature_stage_list)
        feature_pipeline = Pipeline(stages=feature_stage_list)
        feature_pipeline_model = feature_pipeline.fit(df)
        os.system("hadoop fs -rm -r %s" % feature_pipeline_model_file)
        feature_pipeline_model.save(feature_pipeline_model_file)
        save_feature_onehot_encode(spark, feature_pipeline_model, subfix, feature_list)
        return feature_pipeline_model


def save_feature_onehot_encode(spark, feature_pipeline_model, subfix, feature_list):
    ## save one hot encode table
    table_name = "ctr_lr_feature_onehot_encoder%s" % subfix
    feature_index, feature_onehot_encodes = 1, []
    for idx, feature_name in enumerate(feature_list):
        stage_model = feature_pipeline_model.stages[2 * idx]
        # if size of labels is N,
        # only keep N - 1 because 000...000 represents the Nth category       
        for feature_value in stage_model.labels[:-1]:
            feature_onehot_encodes.append((feature_index, feature_value, feature_name))
            feature_index += 1
    feature_df = spark.sparkContext.parallelize(feature_onehot_encodes).map(lambda t: Row(index=t[0], value=t[1], name=t[2])).toDF()
    save_df_to_hive(spark, feature_df, table_name)


def save_lr_model_weights(spark, lr_model, subfix):
    # Shows the best parameters
    # lr_model
    # Shows the weights of best model
    print(lr_model.coefficients)
    table_name = "ctr_lr_weight%s" % subfix
    all_model_coefficients = [coeff.item() for coeff in lr_model.coefficients]
    all_model_coefficients.insert(0, lr_model.intercept)
    index_weight_tuples = list(enumerate(all_model_coefficients))
    coeff_df = spark.sparkContext.parallelize(index_weight_tuples).map(lambda t: Row(index=t[0], weight=t[1])).toDF()
    save_df_to_hive(spark, coeff_df, table_name)
    # select percentile_approx(weight, array(0.1, 0.3, 0.5, 0.7, 0.95)) from ctr_lr_weight;
    # [-6.24, -3.17, -0.54, 0.55, 2.83]


def predict_and_save_results(spark, lr_model, test_df, subfix):
    prediction = lr_model.transform(test_df).select("label", "probability")
    prob_and_label = prediction.rdd.map(lambda row: (row.probability[1].item(), row.label))
    prob_and_label.take(10)

    # select percentile_approx(prob, array(0.01, 0.25, 0.5, 0.75, 0.85, 0.99, 0.9999)) from ctr_label_prob;
    # [2.42E-4, 0.0027, 0.0053, 0.0107, 0.0179, 0.086, 0.3347]
    table_name = "ctr_label_prob%s" % subfix
    prob_and_label_df = prob_and_label.map(lambda t: Row(prob=t[0], label=t[1])).toDF()
    save_df_to_hive(spark, prob_and_label_df, table_name)

    print("start to calculate log loss")
    calculate_log_loss(prob_and_label)


def calculate_log_loss(prob_and_label):   
    # Compute log-loss on the test set 
    log_rdd = prob_and_label.map(lambda t: log(t[0], 2) if t[1] == 1 else log(1 - t[0], 2))
    log_rdd.cache()
    ll = - 1.0  * log_rdd.sum()/log_rdd.count()
    print("log-loss: %s" % ll)
    

def split_df_after_feature_transform(feature_model, df):
    # Use randomSplit with weights and seed to get training, test data sets
    weights = [0.8, 0.2]
    seed = 42
    df_train, df_test = df.randomSplit(weights, seed)

    train_df = feature_model.transform(df_train).select("label", "features").cache() #"weight",
    test_df = feature_model.transform(df_test).select("label",  "features").cache() #"weight",
    df.unpersist()
    df_train.unpersist()
    df_test.unpersist()
    return train_df, test_df


## missing values
def handle_missing_values(df):
    # Drop rows containing null values in the DataFrame
    # df = df.dropna()

    # replace NULL into NA default category
    df = df.fillna("NA")
    return df


def save_df_to_hive(spark, df, table_name):
    os.system("hive -e 'drop table ad.%s'" % table_name)
    temp_table_name =  "%s_temp" % table_name
    df.createOrReplaceTempView(temp_table_name)
    spark.sql("create table ad.%s as select * from %s" % (table_name, temp_table_name))


def read_sample_from_hive(spark):
    # http://spark.apache.org/docs/latest/sql-programming-guide.html#hive-tables
    # province, city, 
    ad_sample_sql = """
        select label, gender, age, placement_id, media_app,
            media_app_industry, placement_type, connection_type, carrier 
        from ad.ctr_predict_dataset where ad_pindustry = '21474836486'
    """
    # 21474836486 - 金融
    # 21474836523 - 游戏

    # over sampling for 10% 
    # ad_sample_sql = """
    #     select label, case when label == 1 then 11.0 else 1.0 end as weight, ad_pindustry, ad_industry, 
    #         gender, age, placement_id, media_app, media_app_industry, placement_type, connection_type, carrier 
    #     from ad.ctr_predict_dataset where 
    # """

    print(ad_sample_sql)
    ad_sample = spark.sql(ad_sample_sql)
    return ad_sample


if __name__ == "__main__":
    fit_feature_pipeline_model = False
    if sys.argv[1] == "t":
        fit_feature_pipeline_model = True
        print("TRY TO FIT THE PIPELINE MODEL.......")
    elif sys.argv[1] == "p":
        fit_feature_pipeline_model = False
        print("LOADING THE FITTED PIPELINE MODEL....")
    else:
        print("args should only be t/p for fit/transform!!!")

    to_train_model = False
    if sys.argv[2] == "t":
        to_train_model = True
        print("TRY TO TRAIN THE MODEL.......")
    elif sys.argv[2] == "p":
        to_train_model = False
        print("LOADING THE TRAINED MODEL....")
    else:
        print("args should only be t/p for train/predct!!!")

    ### train and test the models
    exectue_model_training(fit_feature_pipeline_model, to_train_model)