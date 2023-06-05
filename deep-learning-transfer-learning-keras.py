# Databricks notebook source
# MAGIC %md ## Featurization using a pretrained model for transfer learning
# MAGIC
# MAGIC This notebook demonstrates how to take a pre-trained deep learning model and use it to compute features for downstream models.  This is sometimes called *transfer learning* since it allows transfering knowledge (i.e., the feature encoding) from the pre-trained model to a new model.
# MAGIC
# MAGIC In this notebook:
# MAGIC
# MAGIC * The flowers example dataset
# MAGIC * Distributed featurization using pandas UDFs
# MAGIC   * Load data using Apache Spark's binary files data source
# MAGIC   * Load and prepare a model for featurization
# MAGIC   * Compute features using a Scalar Iterator pandas UDF
# MAGIC
# MAGIC This notebook does not take the final step of using those features to train a new model.  For examples of training a simple model such as logistic regression, refer to the "Machine Learning" examples in the Databricks documentation.
# MAGIC
# MAGIC Requirements:
# MAGIC * Databricks Runtime for Machine Learning

# COMMAND ----------

import pandas as pd
from PIL import Image
import numpy as np
import io

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

from pyspark.sql.functions import col, pandas_udf, PandasUDFType

# COMMAND ----------

# MAGIC %md ### The flowers dataset
# MAGIC
# MAGIC We use the [flowers dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers) from the TensorFlow team as our example dataset, which contains flower photos stored under five sub-directories, one per class. It is hosted under Databricks Datasets for easy access.

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/flower_photos

# COMMAND ----------

# MAGIC %md ### Featurization using pandas UDFs
# MAGIC
# MAGIC This section shows the workflow of computing features using pandas UDFs.  This workflow is flexible, supporting image preprocessing and custom models.  It is also efficient since it takes advantage of pandas UDFs for performance.
# MAGIC
# MAGIC The major steps are:
# MAGIC 1. Load DataFrame
# MAGIC 1. Prepare model
# MAGIC 1. Define image loading and featurization methods
# MAGIC 1. Apply the model in a pandas UDF

# COMMAND ----------

# MAGIC %md #### Load data
# MAGIC
# MAGIC Load images using Spark's binary file data source.  You could alternatively use Spark's image data source, but the binary file data source provides more flexibility in how you preprocess images.

# COMMAND ----------

images = spark.read.format("binaryFile") \
  .option("pathGlobFilter", "*.jpg") \
  .option("recursiveFileLookup", "true") \
  .load("/databricks-datasets/flower_photos")

display(images.limit(5))

# COMMAND ----------

# MAGIC %md #### Prepare your model
# MAGIC
# MAGIC Download a model file for featurization, and truncate the last layer(s).  This notebook uses ResNet50.
# MAGIC
# MAGIC Spark workers need to access the model and its weights.
# MAGIC * For moderately sized models (< 1GB in size), a good practice is to download the model to the Spark driver and then broadcast the weights to the workers.  This notebook uses this approach.
# MAGIC * For large models (> 1GB), it is best to load the model weights from distributed storage to workers directly.

# COMMAND ----------

model = ResNet50(include_top=False)
model.summary()  # verify that the top layer is removed

# COMMAND ----------

bc_model_weights = sc.broadcast(model.get_weights())

def model_fn():
  """
  Returns a ResNet50 model with top layer removed and broadcasted pretrained weights.
  """
  model = ResNet50(weights=None, include_top=False)
  model.set_weights(bc_model_weights.value)
  return model

# COMMAND ----------

# MAGIC %md #### Define image loading and featurization logic in a Pandas UDF
# MAGIC
# MAGIC This notebook defines the logic in steps, building up to the Pandas UDF.  The call stack is:
# MAGIC * pandas UDF
# MAGIC   * featurize a pd.Series of images
# MAGIC     * preprocess one image
# MAGIC
# MAGIC This notebook uses the newer Scalar Iterator pandas UDF to amortize the cost of loading large models on workers.

# COMMAND ----------

def preprocess(content):
  """
  Preprocesses raw image bytes for prediction.
  """
  img = Image.open(io.BytesIO(content)).resize([224, 224])
  arr = img_to_array(img)
  return preprocess_input(arr)

def featurize_series(model, content_series):
  """
  Featurize a pd.Series of raw images using the input model.
  :return: a pd.Series of image features
  """
  input = np.stack(content_series.map(preprocess))
  preds = model.predict(input)
  # For some layers, output features will be multi-dimensional tensors.
  # We flatten the feature tensors to vectors for easier storage in Spark DataFrames.
  output = [p.flatten() for p in preds]
  return pd.Series(output)

# COMMAND ----------

@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
  '''
  This method is a Scalar Iterator pandas UDF wrapping our featurization function.
  The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).
  
  :param content_series_iter: This argument is an iterator over batches of data, where each batch
                              is a pandas Series of image data.
  '''
  # With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
  # for multiple data batches.  This amortizes the overhead of loading big models.
  model = model_fn()
  for content_series in content_series_iter:
    yield featurize_series(model, content_series)

# COMMAND ----------

# MAGIC %md #### Apply featurization to the DataFrame of images

# COMMAND ----------

# Pandas UDFs on large records (e.g., very large images) can run into Out Of Memory (OOM) errors.
# If you hit such errors in the cell below, try reducing the Arrow batch size via `maxRecordsPerBatch`.
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")

# COMMAND ----------

# We can now run featurization on our entire Spark DataFrame.
# NOTE: This can take a long time (about 10 minutes) since it applies a large model to the full dataset.
features_df = images.repartition(16).select(col("path"), col("content"), featurize_udf("content").alias("features"))
features_df.write.mode("overwrite").parquet("dbfs:/ml/tmp/flower_photos_features")

# COMMAND ----------

features_df.display()

# COMMAND ----------

# MAGIC %md ### Train a new model using pre-computed features
# MAGIC
# MAGIC The final step in transfer learning would be to use our pre-computed features to train a new model for a new task, such as classifying flowers.  We omit that step in this notebook.  To complete this final step, refer to Databricks documentation "Machine Learning" section for examples of training a simple model such as logistic regression.

# COMMAND ----------

from pyspark.ml.functions import array_to_vector

features_df = features_df.withColumn("features", array_to_vector("features"))


# COMMAND ----------

features_df.count()

# COMMAND ----------

from pyspark.ml.clustering import KMeans
import mlflow

with mlflow.start_run() as run:
    mlflow.pyspark.ml.autolog()
    
    kmeans = KMeans(k=5, seed=42, maxIter=1000)

    #  Call fit on the estimator and pass in iris_two_features_df
    model = kmeans.fit(features_df)
    mlflow.spark.log_model(model, "kmeans-model", registered_model_name='tf-flowers-kmeans')

    # Obtain the clusterCenters from the KMeansModel
    centers = model.clusterCenters()

    # Use the model to transform the DataFrame by adding cluster predictions
    transformed_sdf = model.transform(features_df).select("path", "content", "prediction")

#     print(centers)
    run_id = run.info.run_id
    print(run_id)

# COMMAND ----------

transformed_sdf = model.transform(features_df).select("path", "content", "prediction")
transformed_sdf.display()

# COMMAND ----------

transformed_sdf.filter(col("prediction")==0).display()

# COMMAND ----------

transformed_sdf.filter(col("prediction")==1).display()

# COMMAND ----------

transformed_sdf.filter(col("prediction")==2).display()

# COMMAND ----------

transformed_sdf.filter(col("prediction")==3).display()

# COMMAND ----------

transformed_sdf.filter(col("prediction")==4).display()

# COMMAND ----------


