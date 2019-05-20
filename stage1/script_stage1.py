import argparse
from pprint import pprint

import numpy as np
from pyspark.sql import SparkSession


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
        .appName("Stage 1") \
        .getOrCreate()

    # read file
    df = spark.read.option("sep", "\t").csv(input_path, header=True, inferSchema='true').select(
        ['customer_id', 'review_id', 'product_id', 'star_rating', 'review_body'])
    df.createOrReplaceTempView("review")
    df.cache()

    # the total number of reviews
    num_reviews = df.count()
    # the number of unique users
    num_unique_users = spark.sql('select count(distinct customer_id) from review').collect()
    # the number of unique products
    num_unique_products = spark.sql('select count(distinct product_id) from review').collect()

    # user-review distribution
    # the top 10 users ranked by the number of reviews they publish
    top_ten_users = spark.sql(
        'select customer_id, count(review_id) as review_count from review '
        'group by customer_id ORDER BY review_count DESC limit 10').collect()
    # the largest number of reviews published by a single user
    top_user = top_ten_users[0]
    # the median number of reviews published by a user
    median_user = np.median(spark.sql(
        'select customer_id, count(review_id) as review_count from review group by customer_id').select(
        'review_count').collect())

    # product-review distribution
    # the top 10 products ranked by the number of reviews they have
    top_ten_products = spark.sql(
        'select product_id, count(review_id) as review_count from review '
        'group by product_id ORDER BY review_count DESC limit 10').collect()
    # the largest number of reviews written for a single product
    top_product = top_ten_products[0]
    # the median number of reviews a product has
    median_product = np.median(spark.sql(
        'select product_id, count(review_id) as review_count from review group by product_id').select(
        'review_count').collect())

    print_result('number_of_reviews', num_reviews)
    print_result('number_of_unique_users', num_unique_users)
    print_result('number_of_unique_products', num_unique_products)
    print_result('largest_number_of_reviews_published_by_a_user', top_user)
    print_result('top_10_users_ranked_by_reviews', top_ten_users)
    print_result('median_number_of_reviews_by_user', median_user)
    print_result('largest_number_of_reviews_published_for_a_product', top_product)
    print_result('top_10_products_ranked_by_reviews', top_ten_products)
    print_result('median_number_of_reviews_by_product', median_product)
    spark.stop()
