import argparse
from operator import add

import numpy as np
from nltk.tokenize import sent_tokenize
from pyspark.sql import SparkSession


def filter_data(review):
    body = review['review_body']
    if not isinstance(body, str):
        return False
    if len(sent_tokenize(body)) < 2:
        return False
    if review['customer_id'] in invalid_users or review['product_id'] in invalid_products:
        return False
    return True


def median_sent_by_customer(review):
    sentences = review['review_body']
    number_of_sentences = len(sent_tokenize(sentences))
    return review['customer_id'], number_of_sentences


def median_sent_by_product(review):
    sentences = review['review_body']
    number_of_sentences = len(sent_tokenize(sentences))
    return review['product_id'], number_of_sentences


def find_median(ids, len_sent):
    med = np.median(len_sent)
    return med


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='the input path')
    parser.add_argument('--output', help='the output path')
    args = parser.parse_args()
    input_path, output_path = args.input, args.output

    spark = SparkSession \
        .builder \
        .appName("Stage 1&2") \
        .getOrCreate()

    # read file
    df = spark.read.option("sep", "\t").csv(input_path, header=True, inferSchema='true').select(
        ['customer_id', 'review_id', 'product_id', 'star_rating', 'review_body'])
    # convert to rdd
    rdd = df.rdd

    # the total number of reviews
    df.count()
    # the number of unique users
    map_by_user = rdd.map(lambda x: (x['customer_id'], 1))
    map_by_user.reduceByKey(lambda x, y: 1).count()
    # the number of unique products
    map_by_product = rdd.map(lambda x: (x['product_id'], 1))
    map_by_product.reduceByKey(lambda x, y: 1).count()

    # user-review distribution
    # the largest number of reviews published by a single user
    top_user = map_by_user.reduceByKey(add).sortBy(lambda x: x[1], ascending=False)
    top_user.first()
    # the top 10 users ranked by the number of reviews they publish
    top_user.take(10)
    # the median number of reviews published by a user
    median_user = np.median(top_user.values().collect())

    # product-review distribution
    # the largest number of reviews written for a single product
    top_product = map_by_product.reduceByKey(add).sortBy(lambda x: x[1], ascending=False)
    top_product.first()
    # the top 10 products ranked by the number of reviews they have
    top_product.take(10)
    # the median number of reviews a product has
    median_product = np.median(top_product.values().collect())

    invalid_users = set(top_user.filter(lambda x: x[1] < median_user).keys().collect())
    invalid_products = set(top_product.filter(lambda x: x[1] < median_product).keys().collect())

    filtered_rdd = rdd.filter(filter_data)

    top_10_users = filtered_rdd.map(median_sent_by_customer).reduceByKey(find_median).sortBy(lambda x: x[1],
                                                                                             ascending=False).take(10)
    top_10_products = filtered_rdd.map(median_sent_by_product).reduceByKey(find_median).sortBy(lambda x: x[1],
                                                                                               ascending=False).take(10)

    spark.sparkContext.parallelize(top_10_users).saveAsTextFile(output_path)
    # spark.sparkContext.parallelize(top_10_products).saveAsTextFile(output_path)
