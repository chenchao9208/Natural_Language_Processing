import codecs
import nltk
import unicodedata
import sys
import string
import math
import numpy as np
from xml.dom import minidom
from sklearn import svm
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn
from scipy.stats import chisquare
from sklearn.naive_bayes import MultinomialNB
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.linear_model import SGDClassifier
#from sklearn.linear_model import RidgeClassifier
#from sklearn.linear_model import Perceptron
#from sklearn.linear_model import PassiveAggressiveClassifier
#from sklearn.naive_bayes import BernoulliNB
#from sklearn.naive_bayes import GaussianNB

# Results:
# English: 64.7%
# Spanish: 83.2%
# Catalan: 85.4%

# python main.py data/English-train.xml data/English-dev.xml English.answer English
# python main.py data/Spanish-train.xml data/Spanish-dev.xml Spanish.answer Spanish
# python main.py data/Catalan-train.xml data/Catalan-dev.xml Catalan.answer Catalan

# ./scorer2 English.answer data/English-dev.key data/English.sensemap
# ./scorer2 Spanish.answer data/Spanish-dev.key
# ./scorer2 Catalan.answer data/Catalan-dev.key


def config(language):
    """
    Change the setting according to different languages

    :param language: language for this case
    """
    global k  # Set The window size
    global ignore_U_activated  # Set True to ignore the training data with Unknown senseid
    global vector_0_1  # Set True to convert vectors to "binary form" e.g.: [0,3,0,4]->[0,1,0,1]
    global remove_punctuations_activated  # Set True to ignore all the punctuation tokens
    global lowercase_activated  # Set True to convert all the tokens to lowercase
    global stemming_activated  # Set True to do stemming on all the tokens
    global remove_stop_words_activated  # Set True to ignore stop words
    global expand_synset_activated  # Set True to involve synsets, hypernyms, hypornyms in the model
    global extract_4c_feature_activated  # Set True to involve feature introduced in 4c in the requirement
    global extract_chi_square_activated  # Set True to involve chi-square feature in 4d in the requirement
    global extract_pmi_activated  # Set True to involve PMI in 4d in the requirement

    if language.__eq__("English"):
        k = 13
        ignore_U_activated = True
        vector_0_1 = False
        remove_punctuations_activated = True
        lowercase_activated = True
        stemming_activated = True
        remove_stop_words_activated = True
        expand_synset_activated = False
        extract_4c_feature_activated = True
        extract_chi_square_activated = False
        extract_pmi_activated = False

    elif language.__eq__("Spanish"):
        k = 13
        ignore_U_activated = True
        vector_0_1 = False
        remove_punctuations_activated = True
        lowercase_activated = True
        stemming_activated = True
        remove_stop_words_activated = True
        expand_synset_activated = False  # not applicable to Spanish, set to False
        extract_4c_feature_activated = True
        extract_chi_square_activated = False
        extract_pmi_activated = False

    elif language.__eq__("Catalan"):
        k = 13
        ignore_U_activated = True
        vector_0_1 = True
        remove_punctuations_activated = True
        lowercase_activated = True
        stemming_activated = False  # not applicable to Catalan, set to False
        remove_stop_words_activated = False  # not applicable to Catalan, set to False
        expand_synset_activated = False  # not applicable to Catalan, set to False
        extract_4c_feature_activated = True
        extract_chi_square_activated = False
        extract_pmi_activated = True


def replace_accented(input_str):
    """
    replace the accented char to unicode

    :param input_str: the string need to check
    :return: string with accented char replaced
    """
    nkfd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])


def remove_punctuations(tokens):
    """
    Method to remove the punctuations.
    used in get_left_right_lists()

    :param tokens: a list of tokens
    :return: a list of tokens without punctuations
    """
    if not remove_punctuations_activated:
        return tokens
    output = []
    for token in tokens:
        if token not in string.punctuation:
            output.append(token)
    return output


def remove_stop_words(tokens, language):
    """
    Method to remove the stop words
    used in get_left_right_lists()
    This method does not work for Catalan language.

    :param tokens: a list of tokens
    :param language: the language of the model
    :return: a list of tokens without stop words
    """
    if not remove_stop_words_activated or language.__eq__("Catalan"):
        return tokens
    output = []
    stop = stopwords.words(language.lower())
    for token in tokens:
        if token not in stop:
            output.append(token)
    return output


def stemming(tokens, language):
    """
    Method to perform stemming
    used in ge_left_right_lists()
    This method does not work for Catalan language

    :param tokens: a list of tokens
    :param language: the language of the model
    :return: a list of stemmed tokens
    """
    if not stemming_activated or language.__eq__("Catalan"):
        return tokens
    output = []
    stemmer = SnowballStemmer(language.lower())
    for token in tokens:
        output.append(stemmer.stem(token))
    return output


def lowercase(tokens):
    """
    Method to convert words to lowercase
    used in ge_left_right_lists()

    :param tokens: a list of tokens
    :return: a list of lowercase tokens
    """
    if not lowercase_activated:
        return tokens
    output = [token.lower() for token in tokens]
    return output


def expand_synset(neighbor_word_list, language):
    """
    Method to expand the neighbor_word_list with synsets, hyponyms and hypernyms.
    used in get_neighbor_words_set()

    :param neighbor_word_list: a union of list of neighbors within k distance of the target
    :param language: the language of the model
    :return: the input list adding the synsets, hyponyms and hypernyms of every item in the list
    """
    if not expand_synset_activated or not language.__eq__("English"):
        return neighbor_word_list

    new_neighbor_word_set = set()
    for word in neighbor_word_list:
        new_neighbor_word_set.add(word)

        synsets = wn.synsets(word)
        synonyms_list = [item.name().split('.')[0] for item in synsets]  # extract: "dog.n.01"->"dog"
        for new_word in synonyms_list:
            new_neighbor_word_set.add(new_word)

        for i in xrange(synonyms_list.__len__()):
            if synonyms_list[i].__eq__(word):
                hyponyms = synsets[i].hyponyms()
                hyponyms_list = [item.name().split('.')[0] for item in hyponyms]  # extract: "dog.n.01"->"dog"
                for new_word1 in hyponyms_list:
                    new_neighbor_word_set.add(new_word1)

                hypernyms = synsets[i].hypernyms()
                hypernyms_list = [item.name().split('.')[0] for item in hypernyms]  # extract: "dog.n.01"->"dog"
                for new_word1 in hypernyms_list:
                    new_neighbor_word_set.add(new_word1)
    return list(new_neighbor_word_set)


def get_left_right_lists(sentence, language):
    """
    Method to get the tokens on the left and right of the target word.
    The result list of tokens will be filtered according to different settings.

    :param sentence: the sentence identity(context or target) in the .xml file
    :param language: the language of the model
    :return: two lists representing words on the left and right
    """
    left_list = nltk.word_tokenize(replace_accented(sentence.childNodes[0].nodeValue.replace('\n', '')))
    right_list = nltk.word_tokenize(replace_accented(sentence.childNodes[2].nodeValue.replace('\n', '')))

    left_list = remove_stop_words(left_list, language)
    right_list = remove_stop_words(right_list, language)

    left_list = remove_punctuations(left_list)
    right_list = remove_punctuations(right_list)

    left_list = lowercase(left_list)
    right_list = lowercase(right_list)

    left_list = stemming(left_list, language)
    right_list = stemming(right_list, language)

    return left_list, right_list


def get_neighbor_words_list(sentence, language):
    """
    Method to get a list of words that are within k distance of the target word in the sentece.
    The result might add the synsets, hyponyms, hypernyms according to the setting.

    :param sentence: the sentence identity(context) in .xml file
    :param language: the language of the model
    :return: a list of neighbor words, sometimes add with synsets, hyponyms and hypernyms.
    """
    if language.__eq__("Spanish") or language.__eq__("Catalan"):
        sentence = sentence.getElementsByTagName('target')[0]

    left_list, right_list = get_left_right_lists(sentence, language)
    neighbor_word_list = []

    for item in left_list[-k:]:
        neighbor_word_list.append(item)
    for item in right_list[:k]:
        neighbor_word_list.append(item)

    neighbor_word_list = expand_synset(neighbor_word_list, language)  # add synsets, hypernyms, hyponyms here

    return neighbor_word_list


def extract_vector(inst, neighbor_word_list, _4c_4d_feature, language):
    """
    Map the target word instance into the vector space

    :param inst: a instance of the lexelt item
    :param neighbor_word_list: the feature-neighbors within k distance of the target
    :param _4c_4d_feature: the feature extracted by 4c and 4d in Homework requirement
    :param language: the language of the model
    :return: a vector to represent the word instance in the vector space
    """
    if language.__eq__("English"):
        sentence = inst.getElementsByTagName('context')[0]
    else:
        sentence = inst.getElementsByTagName('context')[0].getElementsByTagName('target')[0]

    x = []
    neighbors = {}
    left_list, right_list = get_left_right_lists(sentence, language)

    for word in left_list[-k:]:
        count = neighbors.get(word, 0)
        neighbors[word] = count + 1
    for word in right_list[:k]:
        count = neighbors.get(word, 0)
        neighbors[word] = count + 1

    for i in xrange(neighbor_word_list.__len__()):
        n = neighbors.get(neighbor_word_list[i], 0)
        if vector_0_1 and n > 0:
            n = 1
        x.append(n)

    for i in xrange(_4c_4d_feature.__len__()):
        n = neighbors.get(_4c_4d_feature[i], 0)
        if vector_0_1 and n > 0:
            n = 1
        x.append(n)
    return x


def extract_4c_4d_feature(neighbor_word_list, senseid_list, inst_list, language):
    """
    Method to extract the features defined in 4c and 4d of the requirement

    :param neighbor_word_list: the feature-neighbors within k distance of the target
    :param senseid_list: a list of all potential senseids for the target word
    :param inst_list: a list of instances of the lexelt
    :param language: language of the model
    :return: a list of feature words
    """
    if not extract_4c_feature_activated and not extract_chi_square_activated and not extract_pmi_activated:
        return []

    output = []
    # count_map usage:  count_map[neighbor_index][sense_index]
    # each cell represent how many times has a word appears in the training set under the specific senseid
    # this will be used for 4c and 4d feature extraction
    count_map = [[0 for i in xrange(senseid_list.__len__())] for j in xrange(neighbor_word_list.__len__())]
    for inst in inst_list:
        sentence = inst.getElementsByTagName('context')[0]
        senseid = inst.getElementsByTagName('answer')[0].getAttribute('senseid')
        y = senseid_list.index(senseid)
        word_list = get_neighbor_words_list(sentence, language)
        for word in word_list:
            x = neighbor_word_list.index(word)
            count_map[x][y] += 1
    count_map = np.array(count_map)

    if extract_4c_feature_activated:
        probability_map = [[0 for i in xrange(senseid_list.__len__())] for j in xrange(neighbor_word_list.__len__())]
        for i in xrange(neighbor_word_list.__len__()):
            for j in xrange(senseid_list.__len__()):
                a = (0.0 + count_map[i, j]) / np.sum(count_map[i, :])  # N(s,c)
                b = 1 - a  # N(!s,c)
                a += 0.0001  # avoid math domain error
                b += 0.0001  # avoid math domain error
                probability_map[i][j] = math.log(
                    a / b)  # expression in the requirement can be simplified: a/s / b/s = a/b
        probability_map = np.array(probability_map)
        for j in xrange(senseid_list.__len__()):
            i = probability_map[:, j].argmax()  # pick out the word with highest value to represent the sense
            output.append(neighbor_word_list[i])

    if extract_chi_square_activated:
        chi = chisquare(count_map)
        chi = np.array(chi)
        # the process is similar to 4c
        for j in xrange(chi.__len__()):
            i = chi[:, j].argmax()
            output.append(neighbor_word_list[i])

    if extract_pmi_activated:
        pmi_map = [[0 for i in xrange(senseid_list.__len__())] for j in xrange(neighbor_word_list.__len__())]
        # the process is similar to 4c
        for i in xrange(neighbor_word_list.__len__()):
            for j in xrange(senseid_list.__len__()):
                a = (0.0 + count_map[i, j]) / np.sum(count_map[i, :])
                b = (np.sum(count_map[:, j]) + 0.0) / np.sum(count_map)
                a += 0.0001
                b += 0.0001
                pmi_map[i][j] = math.log(a / b)
        pmi_map = np.array(pmi_map)
        for j in xrange(senseid_list.__len__()):
            i = pmi_map[:, j].argmax()
            output.append(neighbor_word_list[i])

    return output


def parse_train_data(training_set, language):
    """
    Method to read the training set, parse the data, and train the classifier models.

    :param training_set: path to the training set
    :param language: language of the model
    :return: information extracted from the training set, including the classifiers, the neighbor_word_lists and extra 4c_4d_features
    """
    print "Reading training set: " + training_set
    xmldoc = minidom.parse(training_set)
    lex_list = xmldoc.getElementsByTagName('lexelt')
    training_output = {}

    print "Processing training set and training models..."
    for node in lex_list:
        lexelt = node.getAttribute('item')
        training_output[lexelt] = {}
        inst_list = node.getElementsByTagName("instance")
        # setup the neighbor_word_list within k distance of the word
        neighbor_word_list = []
        senseid_set = set()
        for inst in inst_list:
            sentence = inst.getElementsByTagName('context')[0]
            senseid_set.add(inst.getElementsByTagName('answer')[0].getAttribute('senseid'))
            neighbor_word_list = list(set(neighbor_word_list + get_neighbor_words_list(sentence, language)))
        senseid_list = list(senseid_set)
        training_output[lexelt]["neighbor_word_list"] = neighbor_word_list
        _4c_4d_feature = extract_4c_4d_feature(neighbor_word_list, senseid_list, inst_list, language)
        training_output[lexelt]["4c_4d_feature"] = _4c_4d_feature
        x_list = []
        y_list = []
        for inst in inst_list:
            y = inst.getElementsByTagName('answer')[0].getAttribute('senseid')
            if ignore_U_activated and y.__eq__('U'):
                continue
            y_list.append(str(replace_accented(y)))
            x = extract_vector(inst, neighbor_word_list, _4c_4d_feature, language)
            x_list.append(x)
        # for each node, build a classifier
        if language.__eq__("English"):
            #clf = RandomForestClassifier(n_estimators=10) 58.9
            #clf = SGDClassifier() 61.1
            #clf = MultinomialNB() 62.9
            #clf = BernoulliNB() 55.8
            #clf = Perceptron() 60.4
            #clf = PassiveAggressiveClassifier() 62.1
            #clf = RidgeClassifier() 62.7
            #clf = svm.LinearSVC() 62.5
            #clf = KNeighborsClassifier()
            #clf = GaussianNB()
            clf = MultinomialNB(alpha=0.95)  #+ alpha=0.95 + k=13 + left_right_order + vector_0_1 off = 64.7
        elif language.__eq__("Spanish"):
            #clf = svm.LinearSVC() 82.0
            #clf = MultinomialNB() 82.2
            #clf = RidgeClassifier() 81.5
            #clf = PassiveAggressiveClassifier() 81.9
            #clf = BernoulliNB() 72.4
            clf = MultinomialNB(alpha=0.50)  #0.25:82.6 0.4:83.1 0.45:83.2  0.5: 83.2 0.55:83.2 0.6:82.8 0.75:82.7
        elif language.__eq__("Catalan"):
            #clf = svm.LinearSVC() # 82.8
            #clf = MultinomialNB() # 80.8
            #clf = RidgeClassifier() 82.6
            #clf = svm.LinearSVC(C=1.5) 82.9
            clf = MultinomialNB(alpha=0.25)  # 0.5:84.3 0.35:84.6 0.3:84.8 0.25:85.4 0.2:85.3
        else:
            clf = svm.LinearSVC()
        clf.fit(x_list, y_list)
        training_output[lexelt]["Classifier"] = clf
    print "Models trained."
    return training_output


def parse_test_data(test_set, training_output, language):
    """
    Method to read the test set, parse the test data, and map each instance into a vector space

    :param test_set: path to the test set
    :param training_output: the result of parse_train_data()
    :param language: language of the model
    :return: parsed vectors
    """
    print "Reading test set: " + test_set
    xmldoc = minidom.parse(test_set)
    data = {}
    lex_list = xmldoc.getElementsByTagName('lexelt')
    for node in lex_list:
        lexelt = node.getAttribute('item')  # item "active.v"
        data[lexelt] = []
        inst_list = node.getElementsByTagName('instance')
        for inst in inst_list:
            instance_id = inst.getAttribute('id')  # id "activate.v.bnc.00024693"
            neighbor_word_list = training_output[lexelt]["neighbor_word_list"]
            _4c_4d_feature = training_output[lexelt]["4c_4d_feature"]
            x = extract_vector(inst, neighbor_word_list, _4c_4d_feature, language)
            data[lexelt].append((instance_id, x))

    return data


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print 'Usage: python main.py <training_file> <test_file> <output_file> <language>'
        print 'Example: python main.py data/English-train.xml data/English-dev.xml English.answer English'
        sys.exit(0)
    config(sys.argv[4])
    training_output = parse_train_data(sys.argv[1], sys.argv[4])
    data = parse_test_data(sys.argv[2], training_output, sys.argv[4])
    outfile = codecs.open(sys.argv[3], encoding='utf-8', mode='w')
    print "Predicting..."
    for lexelt, instances in sorted(data.iteritems(), key=lambda d: replace_accented(d[0].split('.')[0])):
        for instance_id, x in sorted(instances, key=lambda d: int(d[0].split('.')[-1])):
            sid = training_output[lexelt]["Classifier"].predict(x)[0]
            outfile.write(replace_accented(lexelt + ' ' + instance_id + ' ' + sid + '\n'))
    outfile.close()
    print "Complete."
