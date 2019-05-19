import argparse
import logging
from operator import add

import tensorflow as tf
import tensorflow_hub as hub
from nltk.tokenize import sent_tokenize
from pyspark.sql import SparkSession

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def review_embed(review_partition):
    # Create graph and finalize (optional but recommended).
    g = tf.Graph()
    with g.as_default():
        text_input = tf.placeholder(dtype=tf.string, shape=[None])
        embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
        my_result = embed(text_input)
        init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    g.finalize()

    # Create session and initialize.
    session = tf.Session(graph=g)
    session.run(init_op)

    # mapPartition would supply element inside a partition using generator stype
    # this does not fit tensorflow stype
    sents = []
    for review in review_partition:
        review_id = review['review_id']
        length_of_sent = len(sent_tokenize(review['review_body']))
        for sent in sent_tokenize(review['review_body']):
            if length_of_sent > 1:
                my_result_out = session.run(my_result, feed_dict={text_input: [sent]})
                sents.append([review_id + str(length_of_sent), my_result_out])
                length_of_sent = length_of_sent - 1
            else:
                my_result_out = session.run(my_result, feed_dict={text_input: [sent]})
                sents.append([review_id, my_result_out])

    return sents


def calculate_distance(pair):
    with tf.Session() as session:
        distance = session.run(tf.losses.cosine_distance(pair[0][1], pair[1][1], axis=1))
        return pair[0][0], distance


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
        .appName("Stage 3") \
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
    selected_rdd = rdd.filter(lambda x: 'B00006J6VG' == x['product_id'])

    logging.info('Computing average cosine distance for positive products')
    positive_rdd = selected_rdd.filter(lambda x: x['star_rating'] >= 4)
    positive_embedded = positive_rdd.mapPartitions(review_embed)
    positive_distance = positive_embedded.cartesian(positive_embedded).map(calculate_distance)
    added_positive = positive_distance.reduceByKey(add)
    added_positive_number = added_positive.count()
    average_positive = added_positive.values().sum() / (added_positive_number * (added_positive_number - 1))

    logging.info('Computing average cosine distance for negative products')
    negative_rdd = selected_rdd.filter(lambda x: x['star_rating'] <= 2)
    negative_embedding = negative_rdd.mapPartitions(review_embed)
    negative_distance = negative_embedding.cartesian(negative_embedding).map(calculate_distance)
    added_negative = negative_distance.reduceByKey(add)
    added_negative_number = added_negative.count()
    average_negative = added_negative.values().sum() / (added_negative_number * (added_negative_number - 1))

    save_result(average_positive)
    save_result(average_negative)
