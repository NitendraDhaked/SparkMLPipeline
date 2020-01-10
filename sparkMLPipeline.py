
################################################################################
#
# @author: Nitendra
#
################################################################################


from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import *
 
 
spark = SparkSession.builder.getOrCreate()
spark._jsc.hadoopConfiguration().set("fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
spark._jsc.hadoopConfiguration().set("fs.s3a.access.key", "<ACCESS_KEY>")
spark._jsc.hadoopConfiguration().set("fs.s3a.secret.key", "<SECRET_KEY>")
sparkDF = spark.read.options(header='true', inferschema='true', delimiter=',').csv("s3a://xxx-datalake-glue/data/sagemaker/data/forecast-XXX0008215.csv")
#sparkDF =spark.read.csv("s3a://vrts-datalake-glue/data/sagemaker/data/forecast-XXXX0008215.csv", sep=',', header=True)
 
print(sparkDF.printSchema())
print(sparkDF.show())
filterDF =sparkDF.filter(col('type').isin(['Capacity']))
print(filterDF.show())
 
features = ["time"]
lin_data = filterDF.select(col("cap").alias("label"), *features)
print(lr_data.printSchema())
(training, test) = lin_data.randomSplit([.7, .3])
vectorAssembler = VectorAssembler(inputCols=features, outputCol="unscaled_features")
standardScaler = StandardScaler(inputCol="unscaled_features", outputCol="features")
lin_reg = LinearRegression(maxIter=10, regParam=.01)
 
 
stages = [vectorAssembler, standardScaler, lin_reg]
pipeline = Pipeline(stages=stages)
 
 
model = pipeline.fit(training)
prediction = model.transform(test)
 
print(prediction.show())
