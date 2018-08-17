import re
import json
import sys
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import time
import os
import glob

common_words = [u'_eleven',u'_eleven location',u'_eleven location scary',u'ab',u'ab equivalency',u'ab equivalency walgreens',u'ab product',u'ab product employee',u'ab said',u'ab said replied',u'aback argue',u'aback argue quoted',u'aback didn',u'aback didn tell',u'aback gage',u'aback gage needle',u'abaissé',u'abaissé ma',u'abaissé ma note',u'abandon',u'abandon filled',u'abandon filled cars',u'abandoned',u'abandoned lid',u'abandoned lid cup',u'abc',u'abc beer',u'abc beer bigger',u'abc store huge',u'abdominal',u'abdominal surgery',u'abdominal surgery epidural',u'abe',u'abe lincoln',u'abe lincoln guy',u'aber ganz',u'aber ganz launig',u'aberration',u'abfüllmaschine',u'abfüllmaschine aber',u'abfüllmaschine aber ganz',u'abide',u'abide ada',u'ability cash',u'ability cash purchase',u'ability empathize',u'ability empathize customers',u'ability pick',u'ability pick beer',u'ability simple',u'ability simple house',u'ability understand',u'ability understand customer',u'ability watch',u'ability watch chefs',u'able answer',u'able answer simple',u'able assist',u'able assist cashiers',u'able assist ended',u'able change',u'able change clothes',u'able come',u'able come experiences',u'able complete',u'able complete days',u'able coupons',u'able coupons scan',u'able deodorant',u'able deodorant example',u'able desire',u'able desire fulfill',u'able different',u'able different location',u'able download',u'able download yelp',u'able eat',u'able enjoy',u'able enjoy looking',u'able enjoy place',u'able experience',u'able experience real',u'able figure',u'able figure exactly',u'able flavours',u'able flavours wanted',u'able gas hurry',u'able glass',u'able glass glasses',u'able going',u'able going far',u'able going let',u'able hear',u'able hear thing',u'able hear various',u'able help',u'able help employees',u'able help question',u'able injection',u'able injection waited',u'able inside',u'able inside clerk',u'able locate',u'able locate items',u'able looking',u'able make',u'able make better',u'able make home',u'able make quick',u'able manually',u'able manually did',u'able needed quickly',u'able order',u'able order cake',u'able order food',u'able park',u'able pay',u'able pay wining',u'able pick',u'able pick prescription',u'able plate',u'able plate number',u'able request',u'able request short',u'able reverse',u'able reverse pump',u'able succeed',u'able succeed future',u'able tell',u'able tell specifications',u'able uphold',u'able uphold good',u'able video',u'able video car',u'abnormal',u'abnormal time',u'abnormal time eyeing',u'abound',u'abound montreal',u'abound montreal skip',u'abrasive',u'abrasive don',u'abrasive don expect',u'abruptly',u'abruptly hangs',u'abruptly hangs phone',u'absence',u'absence convenient',u'absence convenient parking',u'absolument pas',u'absolument pas pour',u'absolute',u'absolute laziest',u'absolute laziest bunch',u'absolute necessity',u'absolute necessity choose',u'absolute premium',u'absolute premium say',u'absolute tank',u'absolute tank hold',u'absolute time',u'absolute time having',u'absolute worst',u'absolute worst customer',u'absolutely',u'absolutely chock',u'absolutely chock impulse',u'absolutely clueless',u'absolutely clueless word',u'absolutely disgusting',u'absolutely disgusting inedible',u'absolutely disrespectful',u'absolutely disrespectful came',u'absolutely floored',u'absolutely floored embarrassed',u'absolutely horrendous',u'absolutely horrendous job',u'absolutely horrible',u'absolutely horrible attitudes',u'absolutely horrible guessed',u'absolutely horrible honestly',u'absolutely horrible stop',u'absolutely hurry',u'absolutely hurry help',u'absolutely issues',u'absolutely issues returned',u'absolutely law',u'absolutely law nature',u'absolutely like',u'absolutely like sweet',u'absolutely mention',u'absolutely mention ordering',u'absolutely recommend',u'absolutely recommend registry',u'absolutely worst',u'absolutely worst pharmacy',u'abusais',u'abusais privilège',u'abusais privilège indu',u'abusive',u'abusive unethical',u'abusive unethical ago',u'accelerated',u'accelerated track',u'accelerated track seconds',u'accept',u'accept code',u'accept code numbers',u'accept contrary',u'accept contrary website',u'accept credit',u'accept credit card',u'accept return',u'accept return akhavan',u'acceptance',u'acceptance office',u'acceptance office asked',u'accepted pump',u'access',u'access pizza',u'access pizza counter',u'access problems',u'access problems gas',u'accessible',u'accessible directions',u'accessible directions traffic',u'accessories',u'accessories definitely',u'accessories definitely extensive',u'accommodating',u'accommodating patient',u'accommodating patient requests',u'according',u'according sign',u'according sign hodge',u'accordingly',u'accordingly don',u'accordingly don accept',u'account',u'account pay',u'account pay gas',u'accusing',u'accusing people',u'accusing people stealing',u'accusing people wrongly',u'achat',u'achat une',u'achat une boîte',u'achète',u'achète maintenant',u'achète maintenant mes',u'acheter du',u'acheter du takeout',u'acheter et',u'acheter et souvent',u'acknowledge',u'acknowledge mother',u'acquiring',u'acquiring tell',u'acquiring tell proud',u'act',u'act like',u'act like fault',u'action',u'action accusing',u'action accusing people',u'actually better',u'actually better service',u'actually figured',u'actually figured hours',u'actually half',u'actually half way',u'actually lot',u'airport',u'allerdings',u'allerdings dass',u'allerdings dass man',u'alongside',u'alongside convenient',u'alongside convenient store',u'alten land haben',u'amato',u'amazing',u'amazing reason',u'amazing reason guys',u'amazingly',u'amazingly friendly',u'amazingly friendly helpful',u'angemacht',u'angemacht wurde',u'angemacht wurde dass',u'anschließende',u'anschließende essen',u'anschließende essen hat',u'area',u'ask mustard',u'ask mustard mayo',u'asked',u'asked head',u'asked head inside',u'atmosphere',u'atmosphere good',u'atmosphere good did',u'auf',u'auf sich',u'auf sich warten',u'aus dem',u'aus dem alten',u'auto',u'auto glass',u'auto glass abusive',u'available',u'bad',u'bad right',u'bad right walk',u'bad tank',u'bad tank gas',u'bar',u'bauern',u'bauern aus',u'bauern aus dem',u'bbq',u'bbq like',u'bbq like flame',u'bedenkt',u'bedenkt dass',u'bedenkt dass der',u'beef',u'beer',u'bereits',u'best',u'bestimmt reichlich',u'bestimmt reichlich getrunken',u'beware',u'beware just',u'beware just filled',u'bières',u'big',u'big tasted',u'big tasted just',u'birds',u'birds stone',u'birds stone ok',u'biscuits',u'biscuits gravy',u'bit dry',u'bit dry didn',u'bit ketchup',u'bit ketchup ask',u'boot',u'boot sale',u'boot sale just',u'border good',u'border good waited',u'boßel',u'boßel abfüllmaschine',u'boßel abfüllmaschine aber',u'boßeltour',u'breakfast morning',u'breakfast morning looked',u'brisket',u'bro',u'building',u'building property',u'building property boot',u'burgers',u'butcher',u'buy',u'buy crack',u'buy crack marijuana',u'cafe',u'cafe door',u'cafe door breakfast',u'calgary',u'called',u'called dive',u'called dive crack',u'car',u'car wash',u'charlotte',u'chicken',u'claim',u'claim good',u'claim good trashy',u'clean',u'clean necessities',u'clean necessities staff',u'close',u'coffee',u'come',u'come talk',u'come talk auto',u'convenience',u'convenience store',u'cream',u'cut',u'cut fries',u'da',u'dat',u'day',u'day good',u'day good location',u'delicious',u'den',u'denn',u'depending',u'depending day',u'depending day good',u'der',u'des',u'die',u'dog',u'don',u'donair',u'du',u'easy',u'eat',u'eine',u'en',u'est',u'et',u'food',u'fresh',u'friendly',u'fries',u'gas',u'gas station',u'gas stations',u'gas stations market',u'glass',u'glass abusive',u'glass abusive unethical',u'good',u'good location',u'got',u'gravy',u'great',u'great food',u'grocery',u'guys',u'guys come',u'gyros',u'haben',u'hier',u'hotel',u'huge',u'ice',u'ice cream',u'ick',u'il',u'items',u'je',u'jimmy',u'just',u'kwik',u'la',u'land',u'large',u'le',u'les',u'like',u'little',u'lobster',u'local',u'location',u'love',u'make',u'man',u'manager',u'meat',u'mit',u'ne',u'need',u'needed',u'new',u'new manager',u'nice',u'noch',u'obst',u'obsthof',u'order',u'över',u'pas',u'peach',u'peach stand',u'people',u'pharmacy',u'pizza',u'place',u'plus',u'pork',u'pour',u'prices',u'produce',u'pulled',u'qt',u'que',u'quick',u'rate',u'really',u'regular',u'roll',u'rolls',u'room',u'sandwich',u'sandwiches',u'sauce',u'sehr',u'selection',u'serves',u'service',u'servicemen',u'servicemen cash',u'servicemen cash registers',u'shawarma',u'sheetz',u'shop',u'shops',u'simple',u'simple gas',u'simple gas pumps',u'small',u'snacks',u'sont',u'staff',u'staff friendly',u'stand',u'stars',u'station',u'stop',u'store',u'store associate',u'store associate ve',u'store bare',u'store bare necessities',u'summer',u'terry',u'things',u'time',u'today',u'today try',u'today try burgers',u'tour',u'treat',u'trip',u'truck',u'try',u'try breakfast',u'try burgers',u'try burgers fries',u'tv',u'tv brisket',u'tv brisket hamburger',u'und',u'une',u'variety',u've',u've encountered',u've encountered young',u'vegas',u'vibe',u'vibe great',u'vibe great service',u'visited',u'visited min',u'visited min ago',u'von',u'waiting',u'waiting hot',u'waiting hot fingers',u'walgreens',u'wash',u'way',u'winning',u'winning parking',u'winning parking ample',u'wir',u'wo',u'woman',u'work',u'working',u'working register',u'working register hiring',u'yogurt',u'young',u'young male',u'young male working',u'yummy',u'yummy hope',u'yummy hope kickin',u'zeit auf fragen',u'zum',u'zwech']

def printraw(s):
    sys.stdout.buffer.write(s.encode('utf8'))

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


def topk(vectorizer, clf, class_labels,k):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    coefs_with_fns.sort(key=lambda x: x[0])

    topkandbottomk = []
    topkandbottomk.extend(coefs_with_fns[0:k])
    topkandbottomk.extend(coefs_with_fns[-k:])
    return topkandbottomk

def namedfeaturesonly(vectorizer, clf, class_labels,names):
    feature_names = vectorizer.get_feature_names()
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

def no_number_preprocessor(tokens):
    r = re.sub('(\d)+', '', tokens.lower())
    # This alternative just removes numbers:
    # r = re.sub('(\d)+', '', tokens.lower())
    return r


def train(direc):
    #remove old models
    for f in glob.glob(direc + "*.model"):
        os.remove(f)    

    features_for_all = {}
    for file in os.listdir(direc):
        print("{}\n".format(file))
        data = []
        labels = []
        with open(direc + file, 'r') as training_set:
            count_vect = CountVectorizer(preprocessor=no_number_preprocessor,ngram_range=(1,3),stop_words='english')
            for line in training_set:
                j = json.loads(line)
                data.append(j['text'])
                label_10 = 1 if j['stars'] > 3 else 0
                labels.append(label_10)

            #consolidate into proper training sets
            sparse_data = count_vect.fit_transform(data)
            clf = MultinomialNB().fit(sparse_data, labels)
            #print(clf.score(sparse_data, labels))

            #write out the top k features (also save to memory in a matrix)
            
            #topfeatures = topk(count_vect, clf, labels,30)
            topfeatures = namedfeaturesonly(count_vect, clf, labels,common_words)
            features_for_all[file] = {}
            with open(direc + file + ".model", 'w') as model_file:
                model_file.write("#{}\n".format(file))
                for line in topfeatures:
                    model_file.write("{}\t{}\n".format(line[0], line[1]))
                    features_for_all[file][line[1]] = line[0]


    with open(direc + "overall.model", 'w') as model_file:
        a = multikeydict_to_2darray(features_for_all)
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

train('data/full_food_and_convenience/')
#train('data/full_small/')
#create_datasets()

#l = biz_dict.values()
#s = set(l)
#print(len(s))
#for city in s:
#    printraw("{}\n".format(city))
#city_counts = dict((x,l.count(x)) for x in set(l))
#print(city_counts)
