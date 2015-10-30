# author: Chao Chen
# UNI: cc3736

import nltk
import time
import math
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# a function that calculates unigram, bigram, and trigram probabilities
#brown is a python list of the sentences
#this function outputs three python dictionaries, where the key is a tuple expressing the ngram and the value is the log probability of that ngram
#make sure to return three separate lists: one for each ngram
def calc_probabilities(brown):
    # Three dictionaries used to record the numbers of ngrams in the paragraphs.
    unigram_n = {}
    bigram_n = {}
    trigram_n = {}

    uni_denominator = 0

    for sentence in brown:
        sentence += "STOP "
        tokens = nltk.word_tokenize(sentence)  # Type = List
        uni_denominator += tokens.__len__()  # Denominator used by unigram
        for token in tokens:
            if (token,) in unigram_n:
                unigram_n[(token,)] += 1
            else:
                unigram_n[(token,)] = 1
        tokens.insert(0, "*")
        bigram_tuples = tuple(nltk.bigrams(tokens))
        for one_tuple in bigram_tuples:
            if one_tuple in bigram_n:
                bigram_n[one_tuple] += 1
            else:
                bigram_n[one_tuple] = 1
        tokens.insert(0, "*")
        trigram_tuples = tuple(nltk.trigrams(tokens))
        for one_tuple in trigram_tuples:
            if one_tuple in trigram_n:
                trigram_n[one_tuple] += 1
            else:
                trigram_n[one_tuple] = 1

    unigram_p = {item[0]: math.log(float(item[1]) / uni_denominator, 2) for item in unigram_n.items()}
    unigram_n[("*",)] = brown.__len__()
    bigram_p = {item[0]: math.log(float(item[1]) / unigram_n[(item[0][0],)], 2) for item in bigram_n.items()}
    bigram_n[("*", "*")] = brown.__len__()
    trigram_p = {item[0]: math.log(float(item[1]) / bigram_n[(item[0][0], item[0][1])], 2) for item in
                 trigram_n.items()}
    return unigram_p, bigram_p, trigram_p


#each ngram is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams):
    #output probabilities
    outfile = open('A1.txt', 'w')
    for unigram in unigrams:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')
    for bigram in bigrams:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1] + ' ' + str(bigrams[bigram]) + '\n')
    for trigram in trigrams:
        outfile.write(
            'TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')
    outfile.close()


#a function that calculates scores for every sentence
#ngram_p is the python dictionary of probabilities
#n is the size of the ngram
#data is the set of sentences to score
#this function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, data):
    scores = []
    if n == 1:
        for sentence in data:
            line_score = 0
            sentence += "STOP "
            unigram_tokens = nltk.word_tokenize(sentence)
            for token in unigram_tokens:
                line_score += ngram_p[(token,)]
            scores.append(line_score)
    elif n == 2:
        for sentence in data:
            line_score = 0
            sentence = "* " + sentence + "STOP "
            bigram_tuples = tuple(nltk.bigrams(nltk.word_tokenize(sentence)))
            for bigram in bigram_tuples:
                line_score += ngram_p[bigram]
            scores.append(line_score)
    elif n == 3:
        for sentence in data:
            line_score = 0
            sentence = "* * " + sentence + "STOP "
            trigra_tuples = tuple(nltk.trigrams(nltk.word_tokenize(sentence)))
            for trigram in trigra_tuples:
                line_score += ngram_p[trigram]
            scores.append(line_score)
    return scores


#this function outputs the score output of score()
#scores is a python list of scores, and filename is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()


#this function scores brown data with a linearly interpolated model
#each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
#like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, brown):
    scores = []
    for sentence in brown:
        line_score = 0
        sentence += "STOP "
        unigram_tokens = nltk.word_tokenize(sentence)
        bigram_tokens = nltk.word_tokenize(sentence)
        bigram_tokens.insert(0, "*")
        bigram_tuples = tuple(nltk.bigrams(bigram_tokens))
        bigram_tokens.insert(0, "*")
        trigram_tuples = tuple(nltk.trigrams(bigram_tokens))
        for i in xrange(unigram_tokens.__len__()):
            uni_score = 2 ** unigrams[(unigram_tokens[i],)] if (unigram_tokens[i],) in unigrams else 0
            bi_score = 2 ** bigrams[bigram_tuples[i]] if bigram_tuples[i] in bigrams else 0
            tri_score = 2 ** trigrams[trigram_tuples[i]] if trigram_tuples[i] in trigrams else 0
            if uni_score == 0 and bi_score == 0 and tri_score == 0:
                line_score = -1000
                break
            else:
                line_score += math.log(1.0 / 3, 2) + math.log(uni_score + bi_score + tri_score, 2)
        scores.append(line_score)
    return scores


def main():
    ts0 = time.time()
    logging.info("Processing A1: calculate probabilities.")

    #open data
    infile = open('Brown_train.txt', 'r')
    brown = infile.readlines()
    infile.close()

    #calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(brown)

    #question 1 output
    q1_output(unigrams, bigrams, trigrams)

    ts1 = time.time()
    logging.info("Time cost for A1: " + str(ts1 - ts0))
    logging.info("Processing A2: score brown training data with uni-/bi-/trigrams.")

    #score sentences (question 2)
    uniscores = score(unigrams, 1, brown)
    biscores = score(bigrams, 2, brown)
    triscores = score(trigrams, 3, brown)

    #question 2 output
    score_output(uniscores, 'A2.uni.txt')
    score_output(biscores, 'A2.bi.txt')
    score_output(triscores, 'A2.tri.txt')

    ts2 = time.time()
    logging.info("Time cost for A2: " + str(ts2 - ts1))
    logging.info("Processing A3: Linear interpolation for brown training data.")

    #linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, brown)

    #question 3 output
    score_output(linearscores, 'A3.txt')

    ts3 = time.time()
    logging.info("Time cost for A3: " + str(ts3 - ts2))
    logging.info("Processing A5: Linear interpolation for Sample1 and Sample2.")

    #open Sample1 and Sample2 (question 5)
    infile = open('Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open('Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close()

    #score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    #question 5 output
    score_output(sample1scores, 'Sample1_scored.txt')
    score_output(sample2scores, 'Sample2_scored.txt')

    ts4 = time.time()
    logging.info("Time cost for A5: " + str(ts4 - ts3))
    logging.info("Total time cost for Part A: " + str(ts4 - ts0))


if __name__ == "__main__": main()
