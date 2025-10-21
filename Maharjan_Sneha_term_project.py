from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, PCA
from pyspark.ml.classification import LogisticRegression, LinearSVC
import sys
from pyspark.sql.functions import col, sum as Fsum
from pyspark.sql.functions import col, when, udf, concat_ws, avg, lower, regexp_replace
from pyspark.sql.types import IntegerType, StringType
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
import numpy as np
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

sc = SparkContext(appName="CustomerRecommendations")
spark = SparkSession(sc)

df = spark.read.csv(sys.argv[1], header=True, inferSchema=True)

#choosing only the required columns
#since review text might not be so clear to detect the sentiment, also using reviews ratings
selected_columns = ['name','categories','reviews_text','user_sentiment','reviews_rating','reviews_username']
df = df.select(selected_columns)
df.show(20)
print("current row count",df.count())

# number of nulls in each column
col_null_counts = df.select([Fsum(col(c).isNull().cast("int")).alias(c) for c in df.columns])
col_null_counts.show()
#checking rows for user sentiment value being null
df.filter(col("user_sentiment").isNull()).show(5)


#checking and handling null values
df = df.withColumn("categories", when(col("categories").isNull(), "Unknown").otherwise(col("categories"))) #for missing categories values, replace with word Unknown
df = df.dropna(how='any') #deleting rows with nulls
#null count for all columns again
col_null_counts = df.select([Fsum(col(c).isNull().cast("int")).alias(c) for c in df.columns])
col_null_counts.show()
print("current row count after removing null rows",df.count())

#since there are still some empty/missing values for user sentiment
derived_sentiment_val = udf(lambda r: 1 if r >= 4 else (0 if r <= 3 else None), IntegerType())

#adding labels for sentiments
labelled_df = df.withColumn(
    "label",
    when(col("user_sentiment") == "Positive", 1)
    .when(col("user_sentiment") == "Negative", 0)
    .otherwise(derived_sentiment_val(col("reviews_rating"))
    ))
print("Label distribution:")
labelled_df.groupBy("label").count().show()

# Cleaning the user review text for sentiment classification
clean_text_df = labelled_df.withColumn("cleaned_text", lower(regexp_replace(col("reviews_text"), "[^a-zA-Z ]", " ")))

tokenizer = Tokenizer(inputCol="cleaned_text", outputCol="token_words")
stop_words_remover = StopWordsRemover(inputCol="token_words", outputCol="filtered_words")
tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=5000)
idf = IDF(inputCol="raw_features", outputCol="features")

# Logistic Regression used for Sentiment Classification
log_reg_setup = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)
pipeline = Pipeline(stages=[tokenizer, stop_words_remover, tf, idf, log_reg_setup])

train_df, test_df = clean_text_df.randomSplit([0.8, 0.2], seed=80)
lr_model = pipeline.fit(train_df)

# Evaluating the model on test data
lr_predictions = lr_model.transform(test_df)

# Evaluation metrics
lr_predictions.groupBy("label", "prediction").count().show()

#Confusion matrix
log_predictions_and_labels = lr_predictions.select("prediction", "label") \
                              .rdd.map(lambda r: (float(r[0]), float(r[1])))
log_metrics = MulticlassMetrics(log_predictions_and_labels)
cm = log_metrics.confusionMatrix().toArray()
print("Logistic regression confusion matrix",cm)

# Correct metrics
print("Corrected metrics")
print("Accuracy:", log_metrics.accuracy)
print("Precision (label 1):", log_metrics.precision(1.0))
print("Recall (label 1):", log_metrics.recall(1.0))
print("F1 Score (label 1):", log_metrics.fMeasure(1.0))

#######
binary_evaluator = BinaryClassificationEvaluator(labelCol="label")
print("Test AUC:", binary_evaluator.evaluate(lr_predictions))


# Predict Sentiment for Entire Dataset using the logistic regression model we made above
all_pred = lr_model.transform(clean_text_df)
positive_df = all_pred.filter((col("prediction") == 1.0) & (col("reviews_rating") >= 4))
print("Positive review count:", positive_df.count())

# Looking at all the Positive Reviews per Product
product_aggregate = (
    positive_df.groupBy("name")
    .agg(
        concat_ws(" ", F.collect_list("cleaned_text")).alias("all_product_text"), #collecting all positive review texts for a product into a list
        avg("reviews_rating").alias("average_rating"), #calculate mean of all the ratings for that product
        F.first("categories").alias("categories") #take the first non-null categories value for the product
    )
)

print("No. of unique products:", product_aggregate.count())

# Compute TF-IDF for Products

tokenizer2 = Tokenizer(inputCol="all_product_text", outputCol="token_words2")
stop_words_remover2 = StopWordsRemover(inputCol="token_words2", outputCol="filtered_words2")
tf2 = HashingTF(inputCol="filtered_words2", outputCol="raw_features2", numFeatures=5000)
idf2 = IDF(inputCol="raw_features2", outputCol="features2")

tfidf_product_pipeline = Pipeline(stages=[tokenizer2, stop_words_remover2, tf2, idf2])

#creating and implementing a transformation model to vectorize text
tfidf_model = tfidf_product_pipeline.fit(product_aggregate)
product_tfidf = tfidf_model.transform(product_aggregate).select("name", "categories", "average_rating", "features2")

# Computing the Cosine Similarity between products

def product_recommender(product_name, top_n=5, contribution_factor=0.7):
    target = product_tfidf.filter(F.col("name") == product_name).collect()[0]
    if not target:
        print(f"Product '{product_name}' is invalid")
        return
    current_product_tfidf_vec = target['features2'] #the tf-idf vector for the product
    target_rating = target['average_rating'] #average numeric rating of the product
    current_categories = target["categories"]

    #split string into a list, removes whitespaces and empty strings for target product's categories only
    current_categories_list = [c.strip() for c in current_categories.split(",") if c.strip()]
    current_categories_array = F.array(*[F.lit(c) for c in current_categories_list])

    # Broadcasting the target vector
    target_vec_broadcast = sc.broadcast(current_product_tfidf_vec)

    def cosine_similarity(v2):
        v1= target_vec_broadcast.value 
        dot_product = float(v1.dot(v2))
        magnitude_product = np.linalg.norm(v1.toArray()) * np.linalg.norm(v2.toArray())
        return float(dot_product / magnitude_product) if magnitude_product != 0 else 0.0
    
    cosine_udf = F.udf(cosine_similarity, DoubleType()) #determining how simlar products are with a custom function

    #final cleaned and formatted category array for all products
    df_with_categories_array = product_tfidf.withColumn(
        "categories_array",
        F.split(F.regexp_replace(F.col("categories"), ",\\s*", ","), ",")
    )

    cosine_similarity_calcuation = (
        df_with_categories_array
        .withColumn("similarity", cosine_udf(F.col("features2"))) #calls the cosine function on each product's tf-idf vector and the current product's tf-idf vector
        .filter(F.size(F.array_intersect(F.col("categories_array"), current_categories_array)) > 2) #only get the products with atleast this many matching categories
        .filter(F.col("name") != product_name) #removing the product being asked recommendations for
    )
    cosine_similarity_calcuation = cosine_similarity_calcuation.withColumn(
        "final_score",
        contribution_factor * F.col("similarity") + (1 - contribution_factor) * (F.col("average_rating") / 5) 
    )
    top_n_products = cosine_similarity_calcuation.orderBy(F.col("final_score").desc()).limit(top_n)
    results_df = top_n_products.select("name", "categories", "average_rating", "similarity", "final_score")
    results_df.show(truncate=False)
    return results_df


def evaluate_recs(product_name: str, recommendations_df, product_tfidf):

    # Mean cosine similarity between reviews
    mean_cosine_similarity = recommendations_df.agg(F.mean("similarity").alias("mean_similarity")).collect()[0]["mean_similarity"]
    
    # Mean user rating
    mean_rating = recommendations_df.agg(F.mean("average_rating").alias("mean_rating")).collect()[0]["mean_rating"]

    # print metrics
    print("product", product_name)
    print("mean_similarity", mean_cosine_similarity)
    print("mean_rating", mean_rating)
    

sample_products = product_tfidf.orderBy(F.rand(seed=50)).limit(5).collect()

for product in sample_products:
    product_name = product["name"]
    results = product_recommender(product_name, top_n=5)
    evaluate_recs(product_name, results, product_tfidf)





