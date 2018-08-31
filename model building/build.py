import re
import json
import sys
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics import euclidean_distances
from sklearn import manifold
from sklearn.decomposition import PCA
import math
import time
import os
import glob
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob

def printraw(s):
    sys.stdout.buffer.write(s.encode('utf8'))

def get_lemma(word):

    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

def grab_businesses():
    biz_dict = {}
    # open the file
    with open('data\yelp_academic_dataset_business.json', encoding='utf-8') as biz_file:
        for line in biz_file:
            j = json.loads(line)
            cats = "{}".format(j["categories"])
            #Wawa is listed as one of these
            if("Food" in cats and "Convenience Stores" in cats):
                biz_dict[j["business_id"]] = j["state"]

    return biz_dict


def grab_reviews(biz_dict):

    locations = {}
    training_sets = defaultdict(dict)

    # open the file
    with open('data\yelp_academic_dataset_review.json', encoding='utf-8') as review_file:
        for line in review_file:
            j = json.loads(line)

            #if this is a target business
            if j["business_id"] in biz_dict:

                #find physical location
                l_temp = biz_dict[j["business_id"]]
                if l_temp in locations:
                    locations[l_temp] = locations[l_temp]+1
                else:
                    locations[l_temp] = 0

                #store the items in multiple locations
                training_sets[l_temp][j["review_id"]] = j

    
    #write to disk many files
    millis = int(round(time.time() * 1000))
    for l in training_sets.keys():
        with open('data/full/{}.json'.format(l), 'w', encoding='utf-8') as training_file:
            for item in training_sets[l]:
                training_file.write(json.dumps(training_sets[l][item]))
                training_file.write("\n")
 
    return locations            


def create_datasets():
    biz_dict = grab_businesses()
    locations = grab_reviews(biz_dict)

    for x in locations:
        printraw("{}:{}\t".format(x, locations[x]))

    print("\n")

    print(len(biz_dict))


def topk(feature_names, clf,k):

    k = math.ceil(k/2)
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    coefs_with_fns.sort(key=lambda x: x[0])

    topkandbottomk = []
    topkandbottomk.extend(coefs_with_fns[0:k])
    topkandbottomk.extend(coefs_with_fns[-k:])
    return topkandbottomk

def namedfeaturesonly(feature_names, clf, names):
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))

    topkandbottomk = [token for token in coefs_with_fns if token[1] in names]
    return topkandbottomk

def multikeydict_to_2darray(d):
    #find columns (|predictions|)
    uniq_models = list(d.keys())
    col = len(d)+1

    #find rows (uniq(values))
    uniq_features = []
    for item in d:
        uniq_features.extend(d[item].keys())

    uniq_features = list(set(uniq_features))
    row = len(uniq_features)

    a = [[0] * (col) for i in range(row)]

    #add labels
    for j in range(row-1):
        ngram = uniq_features[j]
        a[j+1][0] = ngram

    for i in range(col-1):
        ngram = uniq_models[i]
        a[0][i+1] = ngram

    #add data
    for i in range(col-1):
        ngram = uniq_features[i]
        for j in range(row-1):
            #print("{} {}".format(i,j))
            #print("{} \t {}".format(uniq_features[j],uniq_models[i]))
            #print(d[uniq_models[i]])
            if(uniq_features[j] in d[uniq_models[i]]):
                a[j+1][i+1] = d[uniq_models[i]][uniq_features[j]]
            else:
                a[j+1][i+1] = 0

    return a

def mgs_preprocessor(text):
    text = re.sub('(\d)+', '', text)

    #lemmatize the words
    tokens = nltk.word_tokenize(text)
    tokens = [get_lemma(token.lower()) for token in tokens]

    #retext it 
    text = " ".join(tokens)

    #extract noun phrases
    blob = TextBlob(text)
    #print(blob.noun_phrases)
    tokens.extend([re.sub('[^0-9a-zA-Z]+', '_', item) for item in blob.noun_phrases ])

    return " ".join(tokens)


def find_common_features(model_store, feature_store):

    k = 30
    topfeatures = []
    #first find the topk features for each model
    for model_name in model_store.keys():
        model_topk = topk(feature_store[model_name], model_store[model_name], k)
        topfeatures.extend([item[1] for item in model_topk])
        
    #find the uniq set of the top-k features from every model
    topfeatures = list(set(topfeatures))

    #pull our the weights for the top features over all models
    features_for_all = {}
    for model_name in model_store.keys():
        model_scores = namedfeaturesonly(feature_store[model_name], model_store[model_name], topfeatures)
        features_for_all[model_name] = {}
        for line in model_scores:
            features_for_all[model_name][line[1]] = line[0]        

    return multikeydict_to_2darray(features_for_all)


def write_2darray(a, filename):
    with open(filename, 'w') as model_file:
        for row in a:
            first = True
            model_file.write("[")
            for cell in row:
                if first:
                    model_file.write("'{}', ".format(cell))
                else:
                    model_file.write("{},".format(cell))
                first = False
            model_file.write("],\n")


def save_topics(model, feature_names, no_top_words):
    similarities = euclidean_distances(model.components_)
    #rint (similarities)

    #if we later need this we can add a little noise to smooth the data
    n_samples = len(model.components_)
    noise = np.random.rand(n_samples, n_samples)
    noise = (noise + noise.T)/100
    noise[np.arange(noise.shape[0]), np.arange(noise.shape[0])] = 0
    #similarities += noise
    #print (noise)

    #fit MDS and rescale the positions
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=np.random.RandomState(seed=3),
                       dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(similarities).embedding_
    pos *= np.sqrt((similarities ** 2).sum()) / np.sqrt((pos ** 2).sum())
    #print(pos)

    topic_model = []
    for topic_idx, topic in enumerate(model.components_):
        topic_output = []
        topic_output.append("Topic {}".format(topic_idx))
        topic_output.append("{}".format(pos[topic_idx][0]))
        topic_output.append("{}".format(pos[topic_idx][1]))
        topic_output.append(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        topic_model.append(topic_output)

    return topic_model

def write_topic_model(topic_model, filename):
    with open(filename, 'w') as model_file:
        model_file.write("state,topic,x,y,example\n")
        for k in topic_model.keys():
            topic = topic_model[k]
            for line in topic:
                model_file.write(k.replace(".json","") + ",")
                model_file.write(",".join(line))
                model_file.write("\n")

def train(direc):

    no_features = 10000
    model_store = {}
    feature_store = {}
    topic_store = {}

    #remove old models
    for f in glob.glob(direc + "*.model"):
        os.remove(f)    
    for f in glob.glob(direc + "*.*~"):
        os.remove(f)    

    features_for_all = {}
    for file in os.listdir(direc):
        print("{}\n".format(file))
        data = []
        labels = []
        with open(direc + file, 'r') as training_set:
            tf_vectorizer = CountVectorizer(preprocessor=mgs_preprocessor,stop_words='english')
            tfidf_vectorizer = TfidfVectorizer(preprocessor=mgs_preprocessor, max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
            for line in training_set:
                j = json.loads(line)
                data.append(j['text'])
                label_10 = 1 if j['stars'] > 3 else 0
                labels.append(label_10)

            #consolidate into proper training sets
            tf = tf_vectorizer.fit_transform(data)
            tf_feature_names = tf_vectorizer.get_feature_names()

            tfidf = tfidf_vectorizer.fit_transform(data)
            tfidf_feature_names = tfidf_vectorizer.get_feature_names()

            #run the classifier
            clf = MultinomialNB().fit(tf, labels)
            model_store[file] = clf
            feature_store[file] = tf_feature_names

            #get the topics for df and tfidf by running NMF and LDA
            try:
                no_topics = 15
                no_top_words = 4

                nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
                topic_store[file] = save_topics(nmf, tfidf_feature_names, no_top_words)

                #lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
                #topic_store[file] = save_topics(lda, tfidf_feature_names, no_top_words)
            except:
                print("No NMF topic model for {}".format(file))


        #write out the top k features (also save to memory in a matrix)
        write_2darray(find_common_features(model_store, feature_store), direc + "overall.model")

        #write out the topic model
        write_topic_model(topic_store, direc + "topic.model")

#create_datasets()

train('data/full_food_and_convenience/')
#train('data/full_small/')

