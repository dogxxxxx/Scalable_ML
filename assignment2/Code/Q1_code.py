from pyspark.sql import SparkSession
import pyspark
spark = SparkSession.builder \
        .master("local[10]") \
        .appName("Q1") \
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

logFile = spark.read.text("../Data/NASA_access_log_Jul95.gz").cache()



data = logFile.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
                .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
                .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()

data = data.withColumn('day', data['timestamp'].substr(0, 2))
data = data.withColumn('month', F.format_string('1995-07-'))
data = data.withColumn('date', F.concat(data.month,data.day))
data = data.drop('day', 'month')

# count the amount of requests of each day in July
date_count = data.select('date').groupBy('date').count().sort('count', ascending=False)
# create a column showing the day of week of each day in July
# Here, 1 means Sunday, 2 means Monday, 3 means Tuesday, 4 means Thursday and so on because the first day of the week is Sunday.
date_count = date_count.withColumn('day_of_week',F.dayofweek(date_count.date))
date_count = date_count.where(date_count.day_of_week.isNotNull())

# find the maximum amount of requests of each day of week
max_count = date_count.groupBy('day_of_week').agg(F.max('count')).sort('day_of_week')
# find the minimum amount of requests of each day of week
min_count = date_count.groupBy('day_of_week').agg(F.min('count')).sort('day_of_week')

# combine them together and show the result
merged = max_count.join(min_count,max_count['day_of_week'] == min_count['day_of_week']).drop(min_count.day_of_week).sort('day_of_week')
pdf = merged.toPandas()
Q1A = merged.select('day_of_week','max(count)','min(count)')
print('Q1A:')
Q1A.show()

# plot
plt.plot(pdf['day_of_week'], pdf['max(count)'], label='max counts')
plt.plot(pdf['day_of_week'], pdf['min(count)'], label='min counts')
plt.xlabel('day of week')
plt.ylabel('counts of requests')
plt.title('graph of Q1B')
plt.legend
plt.show()
plt.savefig("../Output/Q1B.png")

# get the request with .mpg
qc = data.filter(F.col('request').contains(".mpg")).cache()

# IMPORTANT: here I chose to get the name first, not just use the original request to group. This is because I care about which video is requested, not the path of the request
# get the names of the videos
requests = qc.withColumn('name', F.regexp_extract(F.col('request'), '.*/([^/$]*.mpg)', 1)).drop('request')

# get the 12 most and least frequent requests
request_count_max = requests.select('name').groupBy('name').count().sort('count', ascending=False).limit(12)
request_count_min = requests.select('name').groupBy('name').count().sort('count').limit(12)

# merge them and show
merged_request = request_count_max.unionByName(request_count_min)
print('Q1C:')
merged_request.show(24, False)

# create the height for the plot
max_height = request_count_max.withColumn("height", request_count_max["count"].cast('int'))
min_height = request_count_min.withColumn("height", request_count_min["count"].cast('int'))

# convert columns in dataframe to list
max_names=max_height.rdd.map(lambda x: x.name).collect()
max_values=max_height.rdd.map(lambda x: x.height).collect()
min_names=min_height.rdd.map(lambda x: x.name).collect()
min_values=min_height.rdd.map(lambda x: x.height).collect()

# plot two bar plots in one figure
plt.subplot(2,1,1)
plt.barh(max_names,max_values)
plt.subplot(2,1,2)
plt.barh(min_names,min_values)
plt.tight_layout()
plt.savefig("../Output/Q1D.png")