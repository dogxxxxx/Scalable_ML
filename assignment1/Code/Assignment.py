from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Assignment1") \
    .config("spark.local.dir","/fastdata/acp21kc") \
    .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel('ERROR')
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, MultilayerPerceptronClassifier, DecisionTreeClassifier, RandomForestClassifier
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.sql import functions as F
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
import time

df_train = spark.read.load("../Data/XOR_Arbiter_PUFs/5xor_128bit/train_5xor_128dim.csv", format="csv", inferSchema="true", header="false")
df_test = spark.read.load("../Data/XOR_Arbiter_PUFs/5xor_128bit/test_5xor_128dim.csv", format="csv", inferSchema="true", header="false")
train_ncolumns = len(df_train.columns)

df_train = df_train.withColumnRenamed('_c128', 'label')
schemaNames_train = df_train.schema.names

df_test = df_test.withColumnRenamed('_c128', 'label')


df_train = df_train.withColumn('label',
    F.when(df_train['label']==-1,0).
    otherwise(df_train['label']))

df_test = df_test.withColumn('label',
    F.when(df_test['label']==-1,0).
    otherwise(df_test['label']))

train_data_sample = df_train.sample(False, 0.1, seed=123)
test_data_sample = df_test.sample(False, 0.1, seed=123)

vecAssembler = VectorAssembler(inputCols = schemaNames_train[0:train_ncolumns-1], outputCol = 'features') 
evaluator = MulticlassClassificationEvaluator\
      (labelCol="label", predictionCol="prediction", metricName="accuracy")


# Logistic Regression
# create LR model
lr = LogisticRegression(labelCol="label", featuresCol="features")
lr_stages = [vecAssembler, lr]
# create pipeline
lr_pipeline = Pipeline(stages=lr_stages)
# create param lists in order to tune params
lr_paramgrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01, 0.001]).addGrid(lr.maxIter, [10, 20, 30]).addGrid(lr.elasticNetParam, [0, 0.5, 1]).build()
# create cross validation model
lr_cross_validation = CrossValidator(estimator=lr_pipeline, estimatorParamMaps=lr_paramgrid, evaluator=MulticlassClassificationEvaluator(), numFolds=3)
# train cv model
lr_cvmodel = lr_cross_validation.fit(train_data_sample)
# get the best model
lr_best_model = lr_cvmodel.bestModel
# extract the params from the best model
lr_param_dict = lr_best_model.stages[-1].extractParamMap()
# create a dictionary to store the params
lr_sane_dict = {}
for k, v in lr_param_dict.items():
  lr_sane_dict[k.name] = v
  
# prediction by the cv model
lr_predictions = lr_cvmodel.transform(test_data_sample)
# create evaluator
lr_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
# get the accuracy of the best model
lr_accuracy = lr_evaluator.evaluate(lr_predictions)
# since this part only need to be printed once, so I only print it while using 5 cores
if sc.master == 'local[5]':
  print('-------------------------------------------------------------')
  print('Logistic Regression:')
  print("Logistic Regression Accuracy = %g " % lr_accuracy)
  print('best regparam: ', lr_sane_dict['regParam'])
  print('best maxIter: ', lr_sane_dict['maxIter'])
  print('best elasticNetParam: ', lr_sane_dict['elasticNetParam'])


# NN
# create mpc (NN) model
mpc = MultilayerPerceptronClassifier(labelCol="label", featuresCol="features")
nn_stages = [vecAssembler, mpc]
# create pipeline
nn_pipeline = Pipeline(stages=nn_stages)
# create lists of params to tune in cv model
nn_paramgrid = ParamGridBuilder().addGrid(mpc.stepSize, [0.03, 0.3, 0.09]).addGrid(mpc.maxIter, [10, 15, 20]).addGrid(mpc.layers, [[len(df_train.columns)-1, 20, 5, 2], [len(df_train.columns)-1, 30, 10, 2], [len(df_train.columns)-1, 15, 3, 2]]).build()
# create cv model
nn_cross_validation = CrossValidator(estimator=nn_pipeline, estimatorParamMaps=nn_paramgrid, evaluator=MulticlassClassificationEvaluator(), numFolds=3)
# train cv model
nn_cvmodel = nn_cross_validation.fit(train_data_sample)
# get the best model
nn_best_model = nn_cvmodel.bestModel
# extract the params in best model
nn_param_dict = nn_best_model.stages[-1].extractParamMap()
# create a dict to store the params values and names
nn_sane_dict = {}
for k, v in nn_param_dict.items():
  nn_sane_dict[k.name] = v

# prediction of the cv model
nn_predictions = nn_cvmodel.transform(test_data_sample)
# create an evaluator
nn_evaluator = MulticlassClassificationEvaluator\
      (labelCol="label", predictionCol="prediction", metricName="accuracy")
# get the accuracy of the best model
nn_accuracy = nn_evaluator.evaluate(nn_predictions)
if sc.master == 'local[5]':  
  print('-------------------------------------------------------------')
  print('NN:')
  print("NN Accuracy = %g " % nn_accuracy)
  print('best maxIter: ', nn_sane_dict['maxIter'])
  print('best layers: ', nn_sane_dict['layers'])
  print('best stepSize: ', nn_sane_dict['stepSize'])


#random forest
# create Random Forest model
rf = RandomForestClassifier(labelCol="label", featuresCol="features")
rf_stages = [vecAssembler, rf]
# create pipeline
rf_pipeline = Pipeline(stages=rf_stages)
# create lists of params in order to tune params
rf_paramgrid = ParamGridBuilder().addGrid(rf.numTrees, [20, 30, 40]).addGrid(rf.maxBins, [16, 32, 48]).addGrid(rf.maxDepth, [3, 5, 7]).build()
# create cv model
rf_cross_validation = CrossValidator(estimator=rf_pipeline, estimatorParamMaps=rf_paramgrid, evaluator=MulticlassClassificationEvaluator(), numFolds=3)
rf_cvmodel = rf_cross_validation.fit(train_data_sample)
# get the best model
rf_best_model = rf_cvmodel.bestModel
rf_param_dict = rf_best_model.stages[-1].extractParamMap()
# create a dict to store the params values and names
rf_sane_dict = {}
for k, v in rf_param_dict.items():
  rf_sane_dict[k.name] = v

# prediction of the rf model
rf_predictions = rf_cvmodel.transform(test_data_sample)
# evaluate the rf best model, no need to create a new evaluator since I found I've created a evaluator in the beginning
rf_accuracy = evaluator.evaluate(rf_predictions)
if sc.master == 'local[5]':
  print('-------------------------------------------------------------')
  print('Random Forest model:')
  print("Random Forest Accuracy = %g " % rf_accuracy)
  print('best maxBins: ', rf_sane_dict['maxBins'])
  print('best maxDepth: ', rf_sane_dict['maxDepth'])
  print('best numTrees: ', rf_sane_dict['numTrees'])

# work on the whole data

# create logistic regression model
# get the time at the start of training
lr_full_trainstart = time.time()
# create lr model and pass the best params by using the dict create before
lr_full = LogisticRegression(labelCol="label", featuresCol="features", regParam=lr_sane_dict['regParam'], maxIter=lr_sane_dict['maxIter'], elasticNetParam=lr_sane_dict['elasticNetParam'])
lr_full_stages = [vecAssembler, lr_full]
# create pipeline
lr_full_pipeline = Pipeline(stages=lr_full_stages)
# train the model
lr_full_model = lr_full_pipeline.fit(df_train)
# time of the end of training
lr_full_trainend = time.time()

# time of the start of testing
lr_full_teststart = time.time()
# prediction of the full lr model
lr_full_prediction = lr_full_model.transform(df_test)
# create two evaluators in order to get accuracy and auc
lr_full_Multievaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
lr_full_Binaryevaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
# get acuracy and auc
lr_full_accuracy = lr_full_Multievaluator.evaluate(lr_full_prediction)
lr_full_auc = lr_full_Binaryevaluator.evaluate(lr_full_prediction)
# time of the end of testing
lr_full_testend = time.time()
# get the total time of both training and testing
lr_time_for_training = lr_full_trainend - lr_full_trainstart
lr_time_for_testing = lr_full_testend - lr_full_teststart
# print the results using 5 cores and 10 cores separately
print('-------------------------------------------------------------')
if sc.master == 'local[5]':
  print('Logistic Regression full model on 5 cores:')
if sc.master == 'local[10]':
  print('Logistic Regression full model on 10 cores:')
print('Accuracy of lr full model: ', lr_full_accuracy)
print('Area under curve of lr full model: ', lr_full_auc)
print('time for training: ', lr_time_for_training, 'seconds')
print('time for testing: ', lr_time_for_testing, 'seconds')



# create NN model
# training start time
nn_full_trainstart = time.time()
# pass the best params to the full nn model
nn_full = MultilayerPerceptronClassifier(labelCol="label", featuresCol="features", maxIter=nn_sane_dict['maxIter'], layers=nn_sane_dict['layers'], stepSize=nn_sane_dict['stepSize'])
# create pipeline
nn_full_stages = [vecAssembler, nn_full]
nn_full_pipeline = Pipeline(stages=nn_full_stages)
# train model
nn_full_model = nn_full_pipeline.fit(df_train)
# training end time
nn_full_trainend = time.time()

# test start time
nn_full_teststart = time.time()
# prediction by nn model
nn_full_prediction = nn_full_model.transform(df_test)
# two evaluators to get auc and accuracy
nn_full_Multievaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
nn_full_Binaryevaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
# results
nn_full_accuracy = nn_full_Multievaluator.evaluate(nn_full_prediction)
nn_full_auc = nn_full_Binaryevaluator.evaluate(nn_full_prediction)
# test end time
nn_full_testend = time.time()
# time for training and testing
nn_time_for_training = nn_full_trainend - nn_full_trainstart
nn_time_for_testing = nn_full_testend - nn_full_teststart
# results using 5 cores and 10 cores
print('-------------------------------------------------------------')
if sc.master == 'local[5]':
  print('NN full model on 5 cores:')
if sc.master == 'local[10]':
  print('NN full model on 10 cores:')
print('Accuracy of nn full model: ', nn_full_accuracy)
print('Area under curve of lr full model: ', nn_full_auc)
print('time for training: ', nn_time_for_training, 'seconds')
print('time for testing: ', nn_time_for_testing, 'seconds')


# create Random Forest model
# training start time
rf_full_trainstart = time.time()
# passing the best params to rf model
rf_full = RandomForestClassifier(labelCol="label", featuresCol="features", maxBins=rf_sane_dict['maxBins'], maxDepth=rf_sane_dict['maxDepth'], numTrees=rf_sane_dict['numTrees'])
# pipeline
rf_full_stages = [vecAssembler, rf_full]
rf_full_pipeline = Pipeline(stages=rf_full_stages)
# train model
rf_full_model = rf_full_pipeline.fit(df_train)
# training end time
rf_full_trainend = time.time()

# testing start time
rf_full_teststart = time.time()
# prediction by rf model
rf_full_prediction = rf_full_model.transform(df_test)
# two evaluators to get auc and accuracy
rf_full_Multievaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
rf_full_Binaryevaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
# get results
rf_full_accuracy = rf_full_Multievaluator.evaluate(rf_full_prediction)
rf_full_auc = rf_full_Binaryevaluator.evaluate(rf_full_prediction)
# testing end time
rf_full_testend = time.time()
# training and testing time
rf_time_for_training = rf_full_trainend - rf_full_trainstart
rf_time_for_testing = rf_full_testend - rf_full_teststart
# results under 5 cores and 10 cores
print('-------------------------------------------------------------')
if sc.master == 'local[5]':
  print('Random Forest full model on 5 cores:')
if sc.master == 'local[10]':
  print('Random Forest full model on 10 cores:')
print('Accuracy of nn full model: ', rf_full_accuracy)
print('Area under curve of lr full model: ', rf_full_auc)
print('time for training: ', rf_time_for_training, 'seconds')
print('time for testing: ', rf_time_for_testing, 'seconds')



















