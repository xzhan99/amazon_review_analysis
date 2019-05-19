import argparse
import logging
import re
from operator import add

import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from pyspark.ml.feature import Word2Vec
from pyspark.sql import Row
from pyspark.sql import SparkSession
from scipy.spatial import distance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def review_sent_tokenize(review):
    sents = []
    for sent in sent_tokenize(review['review_body']):
        # remove punctuations, lowercasing
        sent = re.sub(r'[^\w\s]', '', sent).lower()
        sents.append(word_tokenize(sent))
    if len(sents) <= 1:
        return []
    return sents


def compute_cosine_distance(pair):
    dist = distance.cosine(pair[0]['vector'], pair[1]['vector'])
    if np.isnan(dist):
        return 0
    return dist


def compute_average_distance(reviews):
    # sentence tokenization for positive reviews
    sentences = reviews.flatMap(review_sent_tokenize)
    sentences = sentences.filter(lambda x: len(x) > 0)  # remove []
    num_of_sents = sentences.count()

    tokenized_sents_df = sentences.map(Row('text')).toDF()
    # learn a mapping from sentences to Vectors
    word2vec = Word2Vec(vectorSize=100, minCount=0, inputCol='text', outputCol='vector')
    word2vec_model = word2vec.fit(tokenized_sents_df)
    sent_vectors = word2vec_model.transform(tokenized_sents_df)

    sent_vectors_rdd = sent_vectors.rdd
    # calculate cosine distance between every two vectors
    distances = sent_vectors_rdd.cartesian(sent_vectors_rdd).map(compute_cosine_distance)
    # average distance = sum of distance / number of vectors
    avg_distance = distances.reduce(add) / (num_of_sents * (num_of_sents - 1))
    return avg_distance


def save_result(result):
    logging.warning('=' * 50)
    logging.warning(str(float(result)))
    logging.warning('=' * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='the input path')
    parser.add_argument('--output', help='the output path')
    args = parser.parse_args()
    input_path, output_path = args.input, args.output

    # start spark session
    spark = SparkSession \
        .builder \
        .appName("read") \
        .getOrCreate()

    # read file
    df = spark.read.option("sep", "\t").csv(input_path, header=False, inferSchema="true")
    df = df.withColumnRenamed('_c0', 'customer_id') \
        .withColumnRenamed('_c1', 'product_id') \
        .withColumnRenamed('_c2', 'star_rating') \
        .withColumnRenamed('_c3', 'review_body')
    # convert to rdd
    rdd = df.rdd

    # choose a product
    selected_rdd = rdd.filter(lambda x: 'B00006J6VG' == x['product_id']).sample(withReplacement=False, fraction=0.1)
    # differentiate positive and negative reviews, then cache
    positive_rdd = selected_rdd.filter(lambda x: x['star_rating'] >= 4).filter(
        lambda x: isinstance(x['review_body'], str)).cache()
    negative_rdd = selected_rdd.filter(lambda x: x['star_rating'] <= 2).filter(
        lambda x: isinstance(x['review_body'], str)).cache()

    # calculate average distance for both positive and negative class
    pos_avg_distance = compute_average_distance(positive_rdd)
    neg_avg_distance = compute_average_distance(negative_rdd)

    save_result(pos_avg_distance)
    save_result(neg_avg_distance)
    spark.stop()
