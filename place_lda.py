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
import overpy
import string
import lda
import lda.datasets
import sys
import os
#sys.path.append(os.getcwd())
import placewebscraper
from googleplaces import GooglePlaces, types, lang
#from geopy.geocoders import Nominatim



YOUR_API_KEY = 'AIzaSyA2O6G7eCxOFTbu1HjPuqpuLEnllSDQDB8'

def matchtoGoogleP(placename,lat, lng):
    lat_lng = {}
    lat_lng['lat']=lat
    lat_lng['lng']=lng
    google_places = GooglePlaces(YOUR_API_KEY)
    query_result = google_places.text_search(placename,lat_lng,radius=300)
    #if query_result.has_attributions:
    #    print query_result.html_attributions

    place = query_result.places[0]
    ##    print place.name
    ##    print place.geo_location
    ##    print place.place_id

    # The following method has to make a further API call.
    place.get_details()
    # Referencing any of the attributes below, prior to making a call to
    # get_details() will raise a googleplaces.GooglePlacesAttributeError.
    ##    print place.details # A dict matching the JSON response from Google.
##    print place.website
##    print place.types
##    print place.details['opening_hours']
##    #print place.details['reviews']
##    if 'reviews' in place.details.keys():
##        for r in place.details['reviews']:
##            print r['text']
##    print place.rating
    return place

def getCentroid(nodes):
    points = [(n.lat,n.lon) for n in nodes]
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    _len = len(points)
    centroid_x = sum(x_coords)/_len
    centroid_y = sum(y_coords)/_len
    return (centroid_x, centroid_y)

def getOSMInfo(osmid, elementtype='node'):
    keysofinterest = ['shop', 'amenity', 'leisure', 'tourism', 'historic', 'man_made']
    api = overpy.Overpass()
    #geolocator = Nominatim()

    # We can also see a node's metadata:
    osm = {}
    result = api.query((elementtype+"({}); out;").format(osmid))
    if elementtype == 'node':
        res = result.get_node(osmid)
        osm['lat'] = res.lat
        osm['lon'] = res.lon
    elif elementtype == 'way':
        res = result.get_way(osmid)
        c = getCentroid(res.get_nodes(resolve_missing=True))
        osm['lat']=c[0]
        osm['lon']=c[0]

    #print(res.attributes)
    osm['name'] = res.tags['name']
    osm['keys'] = []
    #location = geolocator.reverse(osmid)
    #print(location.address)
    for k in keysofinterest:
        if k in res.tags.keys():
            osm['keys'].append(k +':'+res.tags[k])
    if 'website' in res.tags.keys():
            osm['website'] = res.tags[k]
    if 'opening_hours' in res.tags.keys():
            osm['opening_hours'] = res.tags[k]
    #print(osm)
    return osm



def extractLDATopics():
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


def enrichOSM(osmid,elementtype):
    enriched = {}
    osm = getOSMInfo(osmid, elementtype)
    #print(osm)
    enriched['osmtype']='|'.join(sorted(osm['keys']))
    place = matchtoGoogleP(osm['name'],osm['lat'],osm['lon'])
    #print(place.details)
    enriched['website']=place.website
    enriched['Gtype'] ='|'.join(sorted(place.types))
    #print place.details['opening_hours']
    #print place.details['reviews']
    if 'reviews' in place.details.keys():
        enriched['reviews']= '. '.join([r['text']+' ' for r in place.details['reviews']])
    wt = placewebscraper.scrape(place.website)
    enriched['webtext']= wt['text']
    enriched['title'] = wt['title']
    print(enriched)



if __name__ == '__main__':
##    osmid= 60018172
##    osm = getOSMInfo(osmid)
    osmid = 266614887
    enrichOSM(osmid,'way')


