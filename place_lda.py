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
import csv
import os
#sys.path.append(os.getcwd())
import placewebscraper
from googleplaces import GooglePlaces, types, lang
#from geopy.geocoders import Nominatim
import json


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
#import pandas as pd
import nltk
from nltk.corpus import stopwords



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
    if 'name' in res.tags:
        osm['name'] = res.tags['name']
        osm['keys'] = []
        #location = geolocator.reverse(osmid)
        #print(location.address)
        for k in keysofinterest:
            if k in res.tags.keys():
                osm['keys'].append(k +':'+res.tags[k])
        if 'website' in res.tags.keys():
                osm['website'] = res.tags['website']
        if 'opening_hours' in res.tags.keys():
                osm['opening_hours'] = res.tags['opening_hours']
        #print(osm)
        return osm
    else:
        return None



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
    print("type(topic_word): {}".format(type(topic_word)))
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
    import re
    print('enrich: '+elementtype+' '+str(osmid))
    enriched = {}
    osm = getOSMInfo(osmid, elementtype)
    if osm == None:
        return
    #print(osm)
    enriched['name']= osm['name']
    print(osm['name'])
    enriched['osmtype']='|'.join(sorted(osm['keys']))
    place = matchtoGoogleP(osm['name'],osm['lat'],osm['lon'])
    enriched['GoogleId'] = place.place_id
    #print(place.details)
    if place.website != None:
        wt = placewebscraper.scrape(place.website)
        if wt !=None:
            enriched['webtext']= wt['text']
            enriched['webtitle'] = wt['title']
        enriched['website']=place.website
    enriched['googletype'] ='|'.join(sorted(place.types))
    #print place.details['opening_hours']
    #print place.details['reviews']
    if 'reviews' in place.details.keys():
        text= ' '.join([r['text']+' ' for r in place.details['reviews']])
        text = text.replace('\n', ' ').replace('\r', '')
        text = re.sub(r'[?|$|.|!]',r'',text)
        text = re.sub(r'[^a-zA-Z]',r' ',text)
        enriched['reviewtext'] = text
    return enriched

def constructTrainingData(filename):
    from pprint import pprint
    td = {}
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for line in reader:
            #line = line.rstrip('\r\n').split("\t")
            osmid = int(os.path.basename(line[0]))
            elementtype = (os.path.basename(os.path.dirname(line[0])))
            en = enrichOSM(osmid,elementtype)
            if en != None:
                en['class']=line[1]
                td[line[0]] = en

    csvfile.close
    #pprint(td)
    import json
    out = (os.path.splitext(os.path.basename(filename))[0][:9])+'_train.json'
    with open(out, 'w') as fp:
        json.dump(td, fp)


def trainLDA(jsonfile, textkey, language='dutch'):
    texts = []
    titles = []
    classes = []
    features = []

    with open(jsonfile) as json_data:
        d = json.load(json_data)
        #print(d)
        for k in d:
            if textkey in (d[k]).keys():
                texts.append(d[k][textkey])
                titles.append(d[k]['name'])
                classes.append(d[k]['class'])
                array = [d[k]['osmtype'],d[k]['googletype']]
                features.append(array)
    json_data.close
    #texts = [NLPclean(t) for t in texts]
    #print(zip(titles,texts))
    vectorizer = CountVectorizer(min_df = 1, stop_words = stopwords.words(language), analyzer = 'word', tokenizer=tokenize)
    X = vectorizer.fit_transform(texts)
    #print(X)
    vocab = vectorizer.get_feature_names()
    print(vocab)
    model = lda.LDA(n_topics=10, n_iter=1500, random_state=1)
    model.fit(X)
    topic_word = model.topic_word_
    #print("type(topic_word): {}".format(type(topic_word)))
    print("shape: {}".format(topic_word.shape))

    # get the top 5 words for each topic (by probablity)
    n = 5
    c = 0
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
        print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))
        c += 1
    #return model
    # apply topic model to new test data set
    doc_topic_test = model.transform(X)
    #print(doc_topic_test)
    i = 0
    for title, topics in zip(titles, doc_topic_test):
        print("{} (top topic: {})".format(title, topics.argmax()))
        features[i].extend(topics)
        i+=1
    print(features)
    print(classes)
    return (titles, doc_topic_test, classes,features)

def classify(topicmodel):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    h = .02  # step size in the mesh

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    X = topicmodel[3]
    y = topicmodel[2]
    print(X)
    print(y)
    #X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.2, random_state=42)

     # iterate over classifiers
    for name, clf in zip(names, classifiers):
        #ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print('classifier: '+name+' score: '+score)

def tokenize(text):
    #from nltk.stem.porter import PorterStemmer
    from nltk.stem.snowball import DutchStemmer
    # Create p_stemmer of class PorterStemmer
    p_stemmer = DutchStemmer()
    text = text.lower()
    from nltk.corpus import stopwords
    stop = set(stopwords.words('dutch'))
    tokens = nltk.word_tokenize(text)
    tokens = [i for i in tokens if i not in string.punctuation and len(i)>=3]
    tokens = [i for i in tokens if i not in stop]
    tokens = [i for i in tokens if i.isalpha()]
    tokens = [p_stemmer.stem(i) for i in tokens]
    #stems = stem_tokens(tokens, stemmer)
    return tokens

def getOSMfeatures(cat):
        placeCategories = {#this is a dictionary of place categories in OSM based on they key value pairs.
        #Possible categories can be retrieved at:
        "amenity": 	{"key" : "amenity","value" : "", "element" : "node"},
        "shop": 	{"key" : "shop","value" : "", "element" : "node"},
        "bar": {"key" : "amenity", "value" : "bar", "element" : "node"},
        "police": {"key" : "amenity", "value" : "police", "element" : "node"},
        "optician": {"key" : "shop", "value" : "optician", "element" : "node"},
        "station": {"key" : "railway", "value" : "station", "element" : "node"},
        "public transport station": {"key" : "public_transport", "value" : "platform", "element" : "node"},
        "office": {"key" : "office", "value" : "", "element" : "node"},
        "leisure": {"key" : "leisure", "value" : "", "element" : "node"},
        "historic": {"key" : "historic", "value" : "", "element" : "node"},
        "civic building":  {"key" : "building", "value" : "civic", "element" : "area"},
        "school building":  {"key" : "building", "value" : "school", "element" : "area"},
        "building":  {"key" : "building", "value" : "", "element" : "area"},
        }
        api = overpy.Overpass()
        #print cat
        #placeCategory = "amenity=police"

        pc = placeCategories[cat]
        if (pc["value"] == ""): #If querying only by key
            kv = pc["key"]
        else:
            kv = pc["key"]+"="+pc["value"]
        elem = pc["element"]

        if (elem =="area" or elem == "line"):
            OSMelem ="way"
        else:
            OSMelem ="node"

        bbox ="50.600, 7.100, 50.748, 7.157"
        #bbox = ", ".join(self.listtoString(self.getCurrentBBinWGS84()))#"50.600, 7.100, 50.748, 7.157"

        #Using Overpass API: http://wiki.openstreetmap.org/wiki/Overpass_API
        result = api.query(OSMelem+"""("""+bbox+""") ["""+kv+"""];out body;
            """)
        results = []
        if (elem == "node"):
            results = result.nodes
        elif (elem == "area" or elem == "line"):
            results = result.ways

        print("Number of results:" + str(len(results)))

        out = 'training.csv'
        with open(out, 'wb') as fp:
            writer = csv.writer(fp,delimiter=' ')
            for element in results:
                url = "https://www.openstreetmap.org/"+str(elem) +'/'+ str(element.id)
                tag = element.tags['leisure'].strip()
                writer.writerow([url, tag])
                #print(tag)
                #print(element.tags.get(tag, "n/a"))
        fp.close



if __name__ == '__main__':
##    osmid= 60018172
##    osm = getOSMInfo(osmid)
##    osmid = 266614887
##    enrichOSM(osmid,'way')
    #getOSMfeatures('leisure')
    constructTrainingData('training.csv')
    #topicmodel = trainLDA('training_train.json', 'webtext')
    #classify(topicmodel)





