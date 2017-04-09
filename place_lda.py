#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      simon scheider
#
# Created:     19/01/2017
# Copyright:   (c) simon 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from __future__ import division, print_function

import numpy as np
import lda
import lda.datasets

# document-term matrix
X = lda.datasets.load_reuters()
##print("type(X): {}".format(type(X)))
##print("shape: {}\n".format(X.shape))
for i in range(10):
    print(format(X[i]))


# the vocab
vocab = lda.datasets.load_reuters_vocab()
##print("type(vocab): {}".format(type(vocab)))
##print("len(vocab): {}\n".format(len(vocab)))

# titles for each story
titles = lda.datasets.load_reuters_titles()
##print("type(titles): {}".format(type(titles)))
##print("len(titles): {}\n".format(len(titles)))

doc_id = 0
word_id = 3117

print("doc id: {} word id: {}".format(doc_id, word_id))
print("-- count: {}".format(X[doc_id, word_id]))
print("-- word : {}".format(vocab[word_id]))
print("-- doc  : {}".format(titles[doc_id]))

#Fit the model
##model = lda.LDA(n_topics=20, n_iter=500, random_state=1)
##model.fit(X)

# topic-word probabilities
##topic_word = model.topic_word_
###print("type(topic_word): {}".format(type(topic_word)))
##print("shape: {}".format(topic_word.shape))

#check the first 5 normalized sums of probabilities
##for n in range(5):
##    sum_pr = sum(topic_word[n,:])
##    print("topic: {} sum: {}".format(n, sum_pr))

# get the top 5 words for each topic (by probablity)
n = 5
##for i, topic_dist in enumerate(topic_word):
##    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
##    print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))

#Document topic probabilities
##doc_topic = model.doc_topic_
##print("type(doc_topic): {}".format(type(doc_topic)))
##print("shape: {}".format(doc_topic.shape))


# Get the most probable topic for each document:
##for n in range(10):
##    topic_most_pr = doc_topic[n].argmax()
##    print("doc: {} topic: {}\n{}...".format(n,
##                                            topic_most_pr,
##                                            titles[n][:50]))

#Most important: apply a given model to a new test dataset

#X = lda.datasets.load_reuters()
#titles = lda.datasets.load_reuters_titles()
X_train = X[10:]
X_test = X[:10]
titles_test = titles[:10]
model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
model.fit(X_train)
topic_word = model.topic_word_
#print("type(topic_word): {}".format(type(topic_word)))
print("shape: {}".format(topic_word.shape))

# get the top 5 words for each topic (by probablity)
n = 5
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
    print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))

# apply topic model to new test data set
doc_topic_test = model.transform(X_test)
for title, topics in zip(titles_test, doc_topic_test):
    print("{} (top topic: {})".format(title, topics.argmax()))
