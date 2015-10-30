#######################################################################################

UNI: cc3736
Name: Chao Chen

#######################################################################################
Running time(second) on cc3736@paris clic.cs.columbia.edu

Total: 709.839099884

        Part A: 92.8003339767
                A1: 18.3074958324
                A2: 39.2762839794
                A3: 28.3945481777
                A4: NONE
                A5: 6.82200598717

        Part B: 617.038765907
                B1: 1.09125804901
                B2: 1.27624988556
                B3: 74.6376950741
                B4: 0.361382007599
                B5: 502.020458937
                B6: 37.6517219543

#######################################################################################
Answer of A2, A3:

In A2 and A3, we used four models to train the data and the results are shown below:

MODEL                           |       PERPLEXITY
unigram                         |       1104.83292814
bigram                          |       57.2215464238
trigram                         |       5.89521267642
linear interpolation            |       13.0759217039

#######################################################################################
Answer of A4:

    As defined in the textbook, the perplexity of a language model on a test set is the
inverse probability of the test set, normalized by the number of words. The higher the
probability of the word sequence, the lower the perplexity. Thus minimizing perplexity
equivalent to maximizing the test probability according to the language model.
    In this experiment, the perplexity:
            { trigram < linear interpolation < bigram < unigram }
    It means that trigram  gives the lowest perplexity, in another word, assigns the
highest probabilities to sentences in Brown_train. Thus it gives more information about
the sentences and performs the best.
    On the contrary, unigram model gives the highest perplexity, namely the lowest
probabilities and the least information about the sentences, thus it performs the worst.
    Linear interpolation uses the weighted average of the three ngrams to evaluate the
probability, thus its perplexity is neither the highest nor the lowest.

#######################################################################################
Answer of A5:

DATA SET                        |       PERPLEXITY
Sample1.txt                     |       11.6492786046
Sample2.txt                     |       1611241155.03

    Perplexity on Sample2.txt is much higher. It means the model trained on Brown_train
gives much less information about Sample2 than Sample1. Thus Sample2 is not from
the Brown dataset.

#######################################################################################
Answer of B5:

Viterbi tagger:
Percent correct tags: 93.7008827776

#######################################################################################
Answer of B6

NLTK tagger:
Percent correct tags: 96.9354729304

    The pos.py calculate the correct tagging rate according to the standard tagged file.
    The NLTK tagger performs better than the Viterbi tagger because NLTK tagger gives a
higher correct percentage.
