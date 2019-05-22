import argparse
from operator import add
from pprint import pprint

import tensorflow as tf
import tensorflow_hub as hub
from nltk.tokenize import sent_tokenize
from pyspark.sql import SparkSession


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
            # mark sentences to know if this review contains multiple sentence
            if length_of_sent > 0:
                my_result_out = session.run(my_result, feed_dict={text_input: [sent]})
                sents.append([review_id + "," + str(length_of_sent), my_result_out])
                length_of_sent = length_of_sent - 1
            else:
                my_result_out = session.run(my_result, feed_dict={text_input: [sent]})
                sents.append([review_id, my_result_out])

    return sents


def calculate_distance(pair):
    with tf.Session() as session:
        distance = session.run(tf.losses.cosine_distance(pair[0][1], pair[1][1], axis=1))
        # return key as review ID and the review iD of another is placed into value
        return pair[0][0], (pair[1][0], distance)


def calculate_center(total, count):
    # since the total including a 0 which is distance to it self, so that the count should be
    # count -1 which exclude itself
    center = total[1]/(count-1)
    return center


def map_values(value):
    # return only one value of the k,v pair when there are two bounded values
    return value[1]


def add_neighbour(neighbour_list, review_id_list):
    for neighbour in review_id_list:
        # if there is multiple sentences in review_body and select the correct one
        neighbour_id = neighbour.split(",")[0]
        neighbour_pos = neighbour.split(",")[1]
        neighbour_in_rdd = rdd.filter(lambda x: neighbour_id == x['review_id'])
        # the 4th attribute is review id
        neighbour_review_body = neighbour_in_rdd.map(lambda x:x[4])
        neighbour_arr = sent_tokenize(neighbour_review_body.collect()[0])
        neighbour_sent = neighbour_arr[int(neighbour_pos)-1]
        neighbour_list.append(neighbour_sent)
    return neighbour_list


def print_result(name, result):
    print('=' * 50 + name + '=' * 50)
    pprint(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='the input path')
    parser.add_argument('--selected_product', help='the product id')
    args = parser.parse_args()
    input_path, selected_product = args.input, args.selected_product

    # start spark session
    spark = SparkSession \
        .builder \
        .appName('stage 3') \
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
    selected_rdd = rdd.filter(lambda x: selected_product == x['product_id'])

    # Computing average cosine distance for positive products
    positive_rdd = selected_rdd.filter(lambda x: x['star_rating'] >= 4).filter(
        lambda x: isinstance(x['review_body'], str)).cache()
    positive_embedded = positive_rdd.mapPartitions(review_embed)
    positive_distance = positive_embedded.cartesian(positive_embedded).map(calculate_distance)
    added_positive = positive_distance.mapValues(map_values).reduceByKey(add)
    added_positive_number = added_positive.count()
    average_positive = added_positive.values().sum() / (added_positive_number * (added_positive_number - 1))

    # Computing average cosine distance for negative products
    negative_rdd = selected_rdd.filter(lambda x: x['star_rating'] <= 2).filter(
        lambda x: isinstance(x['review_body'], str)).cache()
    negative_embedding = negative_rdd.mapPartitions(review_embed)
    negative_distance = negative_embedding.cartesian(negative_embedding).map(calculate_distance)
    added_negative = negative_distance.mapValues(map_values).reduceByKey(add)
    added_negative_number = added_negative.count()
    average_negative = added_negative.values().sum() / (added_negative_number * (added_negative_number - 1))

    # Computing class center for positive reviews
    positive_sorted_center = added_positive.mapValues(
        lambda value: calculate_center(value, added_positive_number)).sortBy(lambda a: a[1])
    positive_center_arr = positive_sorted_center.collect()
    positive_center = positive_center_arr[0][0]
    pos_review_id = positive_center_arr[0][0].split(",")[0]
    pos_pos = positive_center_arr[0][0].split(",")[1]
    pos_review_center_rdd = rdd.filter(lambda x: pos_review_id == x['review_id'])
    # the 4th attribute is review id
    pos_review_center_review_body = pos_review_center_rdd.map(lambda x: x[4])
    pos_class_center_arr = sent_tokenize(pos_review_center_review_body.collect()[0])
    # the class center
    pos_class_center = pos_class_center_arr[int(pos_pos) - 1]  # 'One of my favorite songs.'
    # start to find neighbour by sorting the distance between
    pos_find_neighbour = positive_distance.filter(lambda x: x[0] == positive_center).sortBy(lambda a: a[1][1])
    # the smallest must be compared to itself, so take 11 and remove the first one
    pos_find_neighbour_arr = pos_find_neighbour.take(11)
    pos_find_neighbour_arr = pos_find_neighbour_arr[1:]
    # take the review id out
    pos_neighbours = [el[1][0] for el in pos_find_neighbour_arr]
    # create positive neighbour list and start match the review id into real sentences
    positive_neighbours = []
    positive_neighbours = add_neighbour(positive_neighbours, pos_neighbours)

    # the negative class center and neighbours computation is basically the same as positive one
    negative_sorted_center = added_negative.mapValues(
        lambda value: calculate_center(value, added_negative_number)).sortBy(lambda a: a[1])
    negative_center_arr = negative_sorted_center.collect()
    negative_center = negative_center_arr[0][0]
    neg_review_id = negative_center_arr[0][0].split(",")[0]
    neg_pos = negative_center_arr[0][0].split(",")[1]
    neg_review_center_rdd = rdd.filter(lambda x: neg_review_id == x['review_id'])
    neg_review_center_review_body = neg_review_center_rdd.map(lambda x: x[4])
    neg_class_center_arr = sent_tokenize(neg_review_center_review_body.collect()[0])
    neg_class_center = neg_class_center_arr[int(neg_pos) - 1]
    nag_find_neighbour = negative_distance.filter(lambda x: x[0] == negative_center).sortBy(lambda a: a[1][1])
    nag_find_neighbour_arr = nag_find_neighbour.take(11)
    nag_find_neighbour_arr = nag_find_neighbour_arr[1:]
    nag_neighbours = [el[1][0] for el in nag_find_neighbour_arr]
    negative_neighbours = []
    negative_neighbours = add_neighbour(negative_neighbours, nag_neighbours)

    print_result('pos_avg_distance', average_positive)  # 0.6348533025686292
    print_result('nag_avg_distance', average_negative)  # 0.7027369293901655
    print_result('positive_neighbours', positive_neighbours)
    print_result('negative_neighbours', negative_neighbours)
    spark.stop()
