"""
Distribution of negative words among news article titles
"""

import pandas as pd
import numpy
import matplotlib.pyplot as plt
import numpy as np

# -----
# setting up the spark
import pyspark.pandas as ps
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
conf = SparkConf()
conf.set('spark.executor.memory', '20g')
conf.set('spark.driver.memory', '20g')
SparkContext(conf=conf)

# -----
# nltk library
from nltk.corpus import opinion_lexicon

# -----
#plotting the distribution
from wordcloud import WordCloud

# -----

filepath = ('.../abcnews-date-text.csv')

spark = SparkSession.builder.appName("abcnews").master("local[*]").getOrCreate()

df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(filepath)
# print(df.head())

# -----
neg_list=set(opinion_lexicon.negative())

word_frequency = {}

def calc_words(row):
    for word in row.split():
        if word in neg_list:
            word_frequency[word] = word_frequency.get(word, 0) + 1
    return word_frequency

pd_df=df.toPandas()
headlines = pd_df['headline_text'].apply(calc_words)
# print(headlines.head)

word_distribution = headlines.head(1).values
# print(word_distribution)

my_dic = word_distribution[0]
print(type(my_dic))

my_df = ps.DataFrame(my_dic, index=[0])
# print(my_df)

my_df_transpose = my_df.T
# print(my_df_transpose.head())

# -----

a = my_df_transpose.sort_values(by = 0, ascending=False)
# print(a.shape) # (3145, 1)
b = a.nlargest(1000, 0)
# print(b.head)

# -----

new_dic = b.to_dict() # converting to dictionary
# print(new_dic)

word_cloud = WordCloud(background_color = 'black').generate_from_frequencies(new_dic[0])

# -----
# visualize the image
 fig=plt.figure(figsize=(15, 8))
 plt.imshow(word_cloud, interpolation='bilinear')
 plt.axis("off")
 plt.show()
