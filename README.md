# PlaceLDA
A python module for extracting activities from place webpages

## Code 
The Python module [place_lda.py](https://github.com/simonscheider/PlaceLDA/blob/master/place_lda.py) is the main method. It contains functions for extracting texts from webpages as well as obtaining data (tags as well as reviewtexts) from Open Street Map as well as Google Places. Furthermore it can be used to construct an LDA topic model from these texts and to train and test ML classifiers on the data.

In detail:
  - *constructTrainingData()*: extract, for a list of places (identified by OSM ids) given in a csv file,the webtexts from corresponding websites (given as input) and social media posts (Google places, automatically linked)
  - *trainLDA()*: Build a topic model (with Latent Dirichlet Allocation) from these webtexts and put topics together with place tags (OSM, Google Places) into feature vectors for data mining
  - *classify()*: Run and test different classifiers on these features to predict a given class label that stands for explicit predefined topics (e.g. place types or activities at places).
  - *exportSHP()*: Exports place topics as a shape file (however obly those that could be scraped from OSM
  
## Data:
 - [*training.csv*](https://github.com/simonscheider/PlaceLDA/blob/master/training.csv): This is a 'raw' csv table containing manual ontological classifications of activities for 189 OSM places in Zwolle. Note: Many places have more than one activity (>300 records in total). Activities are captured in terms of the ulo: ontology, with ulo:Activity and ulo:Referent. Also, places have URLs denoting websites from which the information was manually obatined. 
 - [*training_train_u.json*](https://github.com/simonscheider/PlaceLDA/blob/master/training_train_u.json): This is a json file containing the 189 web enriched OSM places (identified by OSM address osm:123 for nodes and osmw:123 for ways). Enrichment was done in several iterations and then results were joined. Still, only 153 places have obtained webtexts, and many less have obtained reviewtexts. Enriched with the following keys:
    - 'class' : Activity class manually added in terms of ulo ontology. Format *ulo:Activity|ulo:Referent*
    - 'uloplace' : Place type manually added in terms of ulo ontology
    - 'website' : URL of the website used to scrape place descriptions
    - 'webtitle': Title of the website used to scrape place descriptions 
    - 'webtext': Text of the website used to scrape place descriptions (cleaned with Beautifulsoup, see placewebscraper.py))
    - 'name': Name of the place (manually added)
    - 'reviewtext':  Text of Google Place reviews (if available). Google place information was added based on spatial distance and name similarity
    - 'googletype': Place tags from Google Places (if available). (in alphabetical order)
    - 'GoogleId': Google Place Id (if available).
    - 'lat': WGS 84 Y Coordinate (taken from OSM, converted to centroid for ways) (if available)
    - 'lon':  WGS 84 X Coordinate (taken from OSM, converted to centroid for ways) (if available)
    - 'shop', 'amenity', 'leisure', 'tourism', 'historic', 'man_made', 'tower', 'cuisine', 'clothes', 'tower', 'beer', 'highway', 'surface', 'place', 'building': Open Street Map key tags containing their respective values, or 'No' if not present at OSM
 
 - [*log_webscraping.txt*](https://github.com/simonscheider/PlaceLDA/blob/master/log_webscraping.txt): This contains a log of the scraping result contained in *training_train_u.json*. What went wrong a.s.o
 
 ## Results:
 - *models/modelX.txt*: This folder contains different LDA + classifier model runs toegther with evaluation results. For example, [models/model1.txt](https://github.com/simonscheider/PlaceLDA/blob/master/models/model1.txt) is a model run on 'training_train_u.json', using 'webtext' for generating 18 topics with LDA, language='dutch', using tags from OSM and Google Places as features in addition to topic probabilities (usetypes=True), constraining the class labels to only ulo:activity classes (no ulo:referent classes) (actlevel=True), and constraining the size of classes to contain at least 5 instances (minclasssize=5). The following classifiers were taken from scikit learn and tested by 10-fold cross validation:  
   - LogisticRegression(C=1e5),
   - KNeighborsClassifier(5),
   - SVC(kernel="linear", C=0.025),
   - SVC(kernel='rbf',gamma=2, C=1),
   - GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
   - DecisionTreeClassifier(max_depth=5),
   - RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
   - MLPClassifier(alpha=1),
   - AdaBoostClassifier(),
   - GaussianNB()
  
  - *placetopics.shp*: A shp file with a subset of places with lat lon together with LDA topics, taken from the model *models/model1allclass.txt*
  
 - *models/treeX.dot* ...: Contains a print out of the decision tree for each model run. Can be opened with [Graphviz editor](http://www.graphviz.org)


  
  
