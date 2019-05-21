import argparse
import re
from operator import add
from pprint import pprint

import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from pyspark.ml.feature import Word2Vec
from pyspark.sql import Row
from pyspark.sql import SparkSession
from scipy.spatial import distance


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
    # sentence tokenization
    sentences = reviews.flatMap(review_sent_tokenize)
    sentences = sentences.filter(lambda x: len(x) > 0)  # remove empty sentence
    num_of_sents = sentences.count()

    tokenized_sents_df = sentences.map(Row('text')).toDF()
    # learn a mapping from sentences to Vectors
    word2vec = Word2Vec(vectorSize=10, minCount=0, inputCol='text', outputCol='vector')
    word2vec_model = word2vec.fit(tokenized_sents_df)
    sent_vectors = word2vec_model.transform(tokenized_sents_df)

    sent_vectors_rdd = sent_vectors.rdd
    # calculate cosine distance between every two vectors
    distances = sent_vectors_rdd.cartesian(sent_vectors_rdd).map(compute_cosine_distance)
    # average distance = sum of distance / number of vectors
    avg_distance = distances.reduce(add) / (num_of_sents * (num_of_sents - 1))
    return avg_distance


def print_result(name, result):
    print('=' * 50 + name + '=' * 50)
    pprint(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='the input path')
    parser.add_argument('--output', help='the output path')
    args = parser.parse_args()
    input_path, output_path = args.input, args.output

    # start spark session
    spark = SparkSession \
        .builder \
        .appName("stage 4") \
        .getOrCreate()

    # read file
    df = spark.read.option("sep", "\t").csv(input_path, header=False, inferSchema="true")
    # logging.warning(df.columns)
    df = df.select(['_c0', '_c2', '_c3', '_c4']) \
        .withColumnRenamed('_c0', 'customer_id') \
        .withColumnRenamed('_c2', 'product_id') \
        .withColumnRenamed('_c3', 'star_rating') \
        .withColumnRenamed('_c4', 'review_body')
    # convert to rdd
    rdd = df.rdd

    # choose a product
    selected_rdd = rdd.filter(lambda x: 'B00006J6VG' == x['product_id'])
    # differentiate positive and negative reviews, then cache
    positive_rdd = selected_rdd.filter(lambda x: x['star_rating'] >= 4).filter(
        lambda x: isinstance(x['review_body'], str)).cache()
    negative_rdd = selected_rdd.filter(lambda x: x['star_rating'] <= 2).filter(
        lambda x: isinstance(x['review_body'], str)).cache()

    # calculate average distance for both positive and negative class
    pos_avg_distance = compute_average_distance(positive_rdd)  # 0.6222155474625909
    neg_avg_distance = compute_average_distance(negative_rdd)  # 0.5326376488012364

    print_result('pos_avg_distance', pos_avg_distance)
    print_result('neg_avg_distance', neg_avg_distance)
    spark.stop()
