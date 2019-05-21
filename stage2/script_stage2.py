import argparse
from pprint import pprint

import numpy as np
from nltk.tokenize import sent_tokenize
from pyspark.sql import SparkSession

MEDIAN_USER = 1
MEDIAN_PRODUCT = 2


def filter_data(review):
    body = review['review_body']
    if not isinstance(body, str):
        return False
    if len(sent_tokenize(body)) < 2:
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


def save_filtered_data(rdd, file_path):
    def to_csv_line(data):
        return '\t'.join(str(d) for d in data)
    rdd.map(to_csv_line).repartition(1).saveAsTextFile(file_path)


def print_result(name, result):
    print('=' * 50 + name + '=' * 50)
    pprint(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='the input path')
    parser.add_argument('--output', help='the output path')
    args = parser.parse_args()
    input_path, output_path = args.input, args.output

    spark = SparkSession \
        .builder \
        .appName("Stage 2") \
        .getOrCreate()

    # read file
    df = spark.read.option("sep", "\t").csv(input_path, header=True, inferSchema='true').select(
        ['customer_id', 'review_id', 'product_id', 'star_rating', 'review_body'])
    # filter unwanted data
    df.createOrReplaceTempView("review")
    df.cache()
    customer = spark.sql('select customer_id, count(review_id) as c_count from review group by customer_id')
    customer.createOrReplaceTempView('customer')
    product = spark.sql('select product_id, count(review_id) as p_count from review group by product_id')
    product.createOrReplaceTempView('product')

    filtered = df.join(customer, df.customer_id == customer.customer_id, 'inner') \
        .where('c_count >= %s' % str(MEDIAN_USER)) \
        .join(product, df.product_id == product.product_id, 'inner') \
        .where('p_count >= %s' % str(MEDIAN_PRODUCT)) \
        .select(['review.customer_id', 'review_id', 'review.product_id', 'star_rating', 'review_body'])
    # remove data where review_body contains less than 2 sentences
    filtered_rdd = filtered.rdd.filter(filter_data).cache()

    # top 10 users ranked by median number of sentences in the reviews they have published
    top_10_users = filtered_rdd.map(median_sent_by_customer) \
        .reduceByKey(find_median).sortBy(lambda x: x[1], ascending=False).take(10)
    # top 10 products ranked by median number of sentences in the reviews they have received
    top_10_products = filtered_rdd.map(median_sent_by_product) \
        .reduceByKey(find_median).sortBy(lambda x: x[1], ascending=False).take(10)

    save_filtered_data(filtered_rdd, output_path)
    print_result('top_10_users_ranked_by_median_number_of_sentences', top_10_users)
    print_result('top_10_products_ranked_by_median_number_of_sentences', top_10_products)
    spark.stop()
