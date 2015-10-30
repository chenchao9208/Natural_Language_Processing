# author: Chao Chen
# UNI: cc3736

import nltk
import math
import time
import numpy as np
import logging
from nltk.corpus import brown as bwn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# this function takes the words from the training data and returns a python list of all of the words that occur more than 5 times
#wbrown is a python list where every element is a python list of the words of a particular sentence
def calc_known(wbrown):
    knownwords = []
    word_count = {}  # a dictionary to record how many times each word has appears.
    for line in wbrown:
        for word in line:
            if not word in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1
    for (word, count) in word_count.items():
        if count > 5:
            knownwords.append(word)
    return knownwords


#this function takes a set of sentences and a set of words that should not be marked '_RARE_'
#brown is a python list where every element is a python list of the words of a particular sentence
#and outputs a version of the set of sentences with rare words marked '_RARE_'
def replace_rare(brown, knownwords):
    return [[word if word in knownwords else "_RARE_" for word in sentence] for sentence in brown]


#this function takes the ouput from replace_rare and outputs it
def q3_output(rare):
    outfile = open("B3.txt", 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


#this function takes tags from the training data and calculates trigram probabilities
#tbrown (the list of tags) should be a python list where every element is a python list of the tags of a particular sentence
#it returns a python dictionary where the keys are tuples that represent the trigram, and the values are the log probability of that trigram
def calc_trigrams(tbrown):
    bigram_n = {}
    trigram_n = {}
    for line in tbrown:
        bigram_tuples = tuple(nltk.bigrams(line))
        trigram_tuples = tuple(nltk.trigrams(line))
        for one_tuple in bigram_tuples:
            if not one_tuple in bigram_n:
                bigram_n[one_tuple] = 1
            else:
                bigram_n[one_tuple] += 1
        for one_tuple in trigram_tuples:
            if not one_tuple in trigram_n:
                trigram_n[one_tuple] = 1
            else:
                trigram_n[one_tuple] += 1
    return {trigram: math.log(float(number) / bigram_n[(trigram[0], trigram[1])], 2) for (trigram, number) in
            trigram_n.items()}


#this function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(qvalues):
    #output
    outfile = open("B2.txt", "w")
    for trigram in qvalues:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(qvalues[trigram])])
        outfile.write(output + '\n')
    outfile.close()


#this function calculates emission probabilities and creates a list of possible tags
#the first return value is a python dictionary where each key is a tuple in which the first element is a word
#and the second is a tag and the value is the log probability of that word/tag pair
#and the second return value is a list of possible tags for this data set
#wbrown is a python list where each element is a python list of the words of a particular sentence
#tbrown is a python list where each element is a python list of the tags of a particular sentence
def calc_emission(wbrown, tbrown):
    evalues = {}
    tagdict = {}
    for w_line, t_line in zip(wbrown, tbrown):
        for word, tag in zip(w_line, t_line):
            if not tag in tagdict:
                tagdict[tag] = {word: 1}
            elif not word in tagdict[tag]:
                tagdict[tag][word] = 1
            else:
                tagdict[tag][word] += 1
    taglist = tagdict.keys()
    for tag in taglist:
        denominator = np.sum(tagdict[tag].values())
        for word in tagdict[tag].keys():
            evalues[(word, tag)] = math.log(float(tagdict[tag][word]) / denominator, 2)
    return evalues, taglist


#this function takes the output from calc_emissions() and outputs it
def q4_output(evalues):
    #output
    outfile = open("B4.txt", "w")
    for item in evalues:
        output = " ".join([item[0], item[1], str(evalues[item])])
        outfile.write(output + '\n')
    outfile.close()


#this function takes data to tag (brown), possible tags (taglist), a list of known words (knownwords),
#trigram probabilities (qvalues) and emission probabilities (evalues) and outputs a list where every element is a string of a
#sentence tagged in the WORD/TAG format
#brown is a list where every element is a list of words
#taglist is from the return of calc_emissions()
#knownwords is from the the return of calc_knownwords()
#qvalues is from the return of calc_trigrams, could be regarded as transtions.
#evalues is from the return of calc_emissions()
#tagged is a list of tagged sentences in the format "WORD/TAG". Each sentence is a string with a terminal newline, not a list of tokens.
def viterbi(brown, taglist, knownwords, qvalues, evalues):
    taglist.remove("*")
    taglist.remove("STOP")
    return [viterbi_entity(sentence, taglist, knownwords, qvalues, evalues) for sentence in brown]


def viterbi_entity(sentence, taglist, knownwords, qvalues, evalues):
    tl = taglist.__len__()
    sl = sentence.__len__()
    viterbi_map = [[[-1000 for i in xrange(sl)] for j in xrange(tl)] for k in xrange(tl)]
    backpointer_map = [[[[-1, -1] for i in xrange(sl)] for j in xrange(tl)] for k in xrange(tl)]
    # Initialization Step. Processing the first word
    word = sentence[0] if sentence[0] in knownwords else "_RARE_"
    for s in xrange(tl):
        q = qvalues.get(("*", "*", taglist[s]), -1000)
        e = evalues.get((word, taglist[s]), -1000)
        viterbi_map[0][s][0] = q + e
        backpointer_map[0][s][0] = -1
    # Recuision Step. Processing the following words
    for t in xrange(1, sl):
        word = sentence[t] if sentence[t] in knownwords else "_RARE_"
        for s in xrange(tl):
            for s_2 in xrange(tl):
                # looking for max
                max_val = float("-inf")
                max_pointer = None
                for s_1 in xrange(tl):
                    first_tag = taglist[s_1] if t != 1 else "*"
                    second_tag = taglist[s_2]
                    current_tag = taglist[s]
                    q = qvalues.get((first_tag, second_tag, current_tag), -1000)
                    e = evalues.get((word, current_tag), -1000)
                    val = viterbi_map[s_1][s_2][t - 1] + q + e
                    if val > max_val:
                        max_val = val
                        max_pointer = [s_1, s_2]
                viterbi_map[s_2][s][t] = max_val
                backpointer_map[s_2][s][t] = max_pointer
    # Termination Step.
    last_s = [-1, -1]
    last_val_max = float("-inf")
    for s_1 in xrange(tl):
        for s_2 in xrange(tl):
            first_tag = taglist[s_1]
            second_tag = taglist[s_2]
            q = qvalues.get((first_tag, second_tag, "STOP"), -1000)
            last_val = viterbi_map[s_1][s_2][sl - 1] + q
            if last_val > last_val_max:
                last_val_max = last_val
                last_s = [s_1, s_2]
    # Recover the WORD/TAG sequence as output.
    output = sentence[-1] + "/" + taglist[last_s[1]] + "\r\n"
    for t in xrange(sl - 1, 0, -1):
        last_s = backpointer_map[last_s[0]][last_s[1]][t]
        output = sentence[t - 1] + "/" + taglist[last_s[1]] + " " + output
    return output


#this function takes the output of viterbi() and outputs it
def q5_output(tagged):
    outfile = open('B5.txt', 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()


#this function uses nltk to create the taggers described in question 6
#brown is the data to be tagged
#tagged is a list of lists of tokens in the WORD/TAG format.
def nltk_tagger(brown):
    tagged = []
    default_tagger = nltk.DefaultTagger("NOUN")
    training = bwn.tagged_sents(tagset="universal")
    bigram_tagger = nltk.BigramTagger(training, backoff=default_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)
    for sentence in brown:
        tagged_sentence = trigram_tagger.tag(sentence)
        tmp = [pair[0] + "/" + pair[1] for pair in tagged_sentence]
        tagged.append(tmp)
    return tagged


def q6_output(tagged):
    outfile = open('B6.txt', 'w')
    for sentence in tagged:
        output = ' '.join(sentence) + '\n'
        outfile.write(output)
    outfile.close()


#a function that returns two lists, one of the brown data (words only) and another of the brown data (tags only)
def split_wordtags(brown_train):
    wbrown = []
    tbrown = []
    for sentence in brown_train:
        w_temp = []
        t_temp = []
        sentence = sentence.strip("\r\n")
        sentence = "*/* */* " + sentence + "STOP/STOP"
        for pair in sentence.split(" "):
            pair = pair.rsplit("/", 1)
            w_temp.append(pair[0])
            t_temp.append(pair[1])
        wbrown.append(w_temp)
        tbrown.append(t_temp)
    return wbrown, tbrown


def main():
    ts0 = time.time()
    logging.info("Processing B1: separate tags and words.")

    #open Brown training data
    infile = open("Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    #split words and tags, and add start and stop symbols (question 1)
    wbrown, tbrown = split_wordtags(brown_train)

    ts1 = time.time()
    logging.info("Time cost for B1: " + str(ts1 - ts0))
    logging.info("Processing B2: calculate trigram probabilities.")

    #calculate trigram probabilities (question 2)
    qvalues = calc_trigrams(tbrown)

    #question 2 output
    q2_output(qvalues)

    ts2 = time.time()
    logging.info("Time cost for B2: " + str(ts2 - ts1))
    logging.info("Processing B3: calculate knownwords and replace rare words.")

    #calculate list of words with count > 5 (question 3)
    knownwords = calc_known(wbrown)

    #get a version of wbrown with rare words replace with '_RARE_' (question 3)
    wbrown_rare = replace_rare(wbrown, knownwords)

    #question 3 output
    q3_output(wbrown_rare)

    ts3 = time.time()
    logging.info("Time cost for B3: " + str(ts3 - ts2))
    logging.info("Processing B4: calculate emission probabilities.")


    #calculate emission probabilities (question 4)
    evalues, taglist = calc_emission(wbrown_rare, tbrown)

    #question 4 output
    q4_output(evalues)

    ts4 = time.time()
    logging.info("Time cost for B4: " + str(ts4 - ts3))
    logging.info("Processing B5: Viterbi for Brown_dev.")

    #delete unneceessary data
    del brown_train
    del wbrown
    del tbrown
    del wbrown_rare

    #open Brown development data (question 5)
    infile = open("Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()
    #format Brown development data here
    brown_dev = [nltk.word_tokenize(sentence) for sentence in brown_dev]

    #do viterbi on brown_dev (question 5)
    viterbi_tagged = viterbi(brown_dev, taglist, knownwords, qvalues, evalues)

    #question 5 output
    q5_output(viterbi_tagged)

    ts5 = time.time()
    logging.info("Time cost for B5: " + str(ts5 - ts4))
    logging.info("Processing B6: NLTK's taggers for Brown_dev.")

    #do nltk tagging here
    nltk_tagged = nltk_tagger(brown_dev)

    #question 6 output
    q6_output(nltk_tagged)

    ts6 = time.time()
    logging.info("Time cost for B6: " + str(ts6 - ts5))
    logging.info("Total time cost for Part B: " + str(ts6 - ts0))


if __name__ == "__main__": main()
