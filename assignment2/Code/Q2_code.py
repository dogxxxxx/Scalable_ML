from pyspark.sql import SparkSession
import pyspark
spark = SparkSession.builder \
        .master("local[10]") \
        .appName("Q2") \
        .config("spark.local.dir","/fastdata/acp21kc") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR") 
import pyspark.sql.functions as F
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyspark.sql import DataFrame
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors
from prettytable import PrettyTable
import numpy as np

logFile2 = spark.read.csv("../Data/ml-25m/ratings.csv", header='true').cache()
q2data = logFile2.withColumn("rating", logFile2["rating"].cast("double")).withColumn("userId", logFile2["userId"].cast("long")).withColumn("movieId", logFile2["movieId"].cast("long")).withColumn("timestamp", logFile2["timestamp"].cast("long"))
tags = spark.read.csv("../Data/ml-25m/tags.csv", header='true').cache()

# split the training data into five pieces
split = q2data.randomSplit([0.2,0.2,0.2,0.2,0.2], seed=123)


# first version of ALS
als = ALS(userCol="userId", itemCol="movieId", seed=123, coldStartStrategy="drop")
# second version of ALS
# the default regParam is 1, here I want to know the results if I reduce the regParam, which means more likely to overfit.
# Also, I want to increase the rate of increase of the confidence matrix since the rating from 1 to 5 is quite precise.
als_second = ALS(userCol="userId", itemCol="movieId", seed=123,regParam=0.2, alpha=10, coldStartStrategy="drop")
# create a list to use different splits to train and test
a = [[0,1,2,3,4],[0,1,2,4,3],[0,1,3,4,2],[0,2,3,4,1],[1,2,3,4,0]]

# create lists to store the rmse
first_hot_rmse_list = []
second_hot_rmse_list = []
first_cool_rmse_list = []
second_cool_rmse_list = []
# create a list to store itemFactor
movie_fac = []
# create a rmse table
rmse_table = PrettyTable(['RMSE','first split','second split','third split','fourth split','fifth split'])
for i in range(len(split)):
  # combine four splits as training set
  training = split[a[i][0]].union(split[a[i][1]]).union(split[a[i][2]]).union(split[a[i][3]])
  # calculate the amount of ratings post by a user
  training_count = training.select('userId').groupBy('userId').count()
  # calculate the amount of users in order to get the amount of hot users and cool users
  train_user_amount = training_count.count()
  # create a dataframe for hot users
  max_count = training_count.groupBy('userId').agg(F.max('count')).sort('max(count)', ascending=False).limit(int(0.1*train_user_amount))
  # create a dataframe for cool users
  min_count = training_count.groupBy('userId').agg(F.min('count')).sort('min(count)', ascending=True).limit(int(0.1*train_user_amount))
  # fit two als models
  als_model = als.fit(training)
  als_second_model = als_second.fit(training)
  # create test data of hot users
  hot_test_data = split[a[i][4]].join(max_count,split[a[i][4]].userId ==  max_count.userId,"leftsemi")
  # create test data of cool users
  cool_test_data = split[a[i][4]].join(min_count,split[a[i][4]].userId ==  min_count.userId,"leftsemi")
  
  # predictions of hot users in test data in first model
  hot_predictions = als_model.transform(hot_test_data)
  # predictions of hot users in test data in second model
  second_hot_predictions = als_second_model.transform(hot_test_data)
  # predictions of cool users in test data in first model
  cool_predictions = als_model.transform(cool_test_data)
  # predictions of cool users in test data in secpnd model
  second_cool_predictions = als_second_model.transform(cool_test_data)
  evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
  
  # calculate rmse of hot users in first model and append it
  
  hot_rmse = evaluator.evaluate(hot_predictions)
  first_hot_rmse_list.append(hot_rmse)
  # calculate rmse of cool users in first model and append it
  cool_rmse = evaluator.evaluate(cool_predictions)
  first_cool_rmse_list.append(cool_rmse)
  # calculate rmse of hot users in second model and append it
  second_hot_rmse = evaluator.evaluate(second_hot_predictions)
  second_hot_rmse_list.append(second_hot_rmse)
  # calculate rmse of cool users in second model and append it
  second_cool_rmse = evaluator.evaluate(second_cool_predictions)
  second_cool_rmse_list.append(second_cool_rmse) 
  
  # generate recommendation for movieID for Question B
  moviefac = als_model.itemFactors
  movie_fac.append(moviefac)

# complete the RMSE table
rmse_table.add_row(['1st version hot', first_hot_rmse_list[0],first_hot_rmse_list[1],first_hot_rmse_list[2],first_hot_rmse_list[3],first_hot_rmse_list[4]])  
rmse_table.add_row(['1st version cool', first_cool_rmse_list[0],first_cool_rmse_list[1],first_cool_rmse_list[2],first_cool_rmse_list[3],first_cool_rmse_list[4]])
rmse_table.add_row(['2nd version hot', second_hot_rmse_list[0],second_hot_rmse_list[1],second_hot_rmse_list[2],second_hot_rmse_list[3],second_hot_rmse_list[4]])
rmse_table.add_row(['2nd version cool', second_cool_rmse_list[0],second_cool_rmse_list[1],second_cool_rmse_list[2],second_cool_rmse_list[3],second_cool_rmse_list[4]])
print('Q2 A:')
print(rmse_table)

# plot
x = ['split 1','split 2','split 3','split 4','split 5']
x_axis = np.arange(len(x))
plt.bar(x_axis - 0.3, first_hot_rmse_list, width=0.2, label='hot rmse 1')
plt.bar(x_axis - 0.1, first_cool_rmse_list, width=0.2, label='cool rmse 1')
plt.bar(x_axis + 0.1, second_hot_rmse_list, width=0.2, label='hot rmse 2')
plt.bar(x_axis + 0.3, second_cool_rmse_list, width=0.2, label='cool rmse 2')
plt.xlabel("splits")
plt.ylabel("value")
plt.xticks(x_axis, x)
plt.title("rmse plot")
plt.legend()
plt.show()
plt.savefig("../Output/Q2A.png")

# k-means

# first create lists to store the max and min tags
top_tags = []
bottom_tags = []
final_table = PrettyTable(["tags", "first split", "second split", "third split", "fourth split", "fifth split"])
for i in range(len(movie_fac)):
  # build k-means model
  kmeans = KMeans().setK(10).setSeed(123)
  # train k-means model
  kmeans_model = kmeans.fit(movie_fac[i])
  # predict the class of each movie
  kmeans_predictions = kmeans_model.transform(movie_fac[i])
  # find out the two largest clusters
  top_group_count = kmeans_predictions.select('prediction').groupBy('prediction').count().sort('count', ascending=False).limit(2)

  # get the number of the groups
  large_group1 = top_group_count.collect()[0][0]
  large_group2 = top_group_count.collect()[1][0]
  
  # get two dataframes, each corresponds to one of the two largest clusters
  large_df1 = kmeans_predictions.filter(kmeans_predictions.prediction == large_group1)
  large_df2 = kmeans_predictions.filter(kmeans_predictions.prediction == large_group2)
  
  # generate the dataframe with tags
  large1_full = large_df1.join(tags,large_df1.id ==  tags.movieId,"inner")
  large2_full = large_df2.join(tags,large_df2.id ==  tags.movieId,"inner")

  # count the amount of each tag
  tag_amount1 = large1_full.groupBy('tag').count().sort('count', ascending=False)
  tag_amount2 = large2_full.groupBy('tag').count().sort('count', ascending=False)
  
  # append the tags to the list
  top_tags.append(tag_amount1.collect()[0][0])
  top_tags.append(tag_amount2.collect()[0][0])
  bottom_tags.append(tag_amount1.collect()[-1][0])
  bottom_tags.append(tag_amount2.collect()[-1][0])

# finish the table
final_table.add_row(["first group top tag", top_tags[0], top_tags[2], top_tags[4], top_tags[6], top_tags[8]])
final_table.add_row(["second group top tag", top_tags[1], top_tags[3], top_tags[5], top_tags[7], top_tags[9]])
final_table.add_row(["first group bottom tag", bottom_tags[0], bottom_tags[2], bottom_tags[4], bottom_tags[6], bottom_tags[8]])
final_table.add_row(["second group bottom tag", bottom_tags[1], bottom_tags[3], bottom_tags[5], bottom_tags[7], bottom_tags[9]])
print('Q2 B:')
print(final_table)