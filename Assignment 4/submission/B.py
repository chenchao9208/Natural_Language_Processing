from __future__ import division
from collections import defaultdict
from nltk.align.ibm1 import IBMModel1
from nltk.align import AlignedSent
import A


class BerkeleyAligner():
    def __init__(self, align_sents, num_iter):
        # self.t, self.q = self.train(align_sents, num_iter)
        self.probabilities, self.alignments = self.train(align_sents, num_iter)

    # Computes the alignments for align_sent, using this model's parameters. Return
    # an AlignedSent object, with the sentence pair and the alignments computed.
    def align(self, align_sent):
        """
        Returns the alignment result for one sentence pair.
        """
        if self.probabilities is None or self.alignments is None:
            raise ValueError("The model does not train.")

        alignment = []

        l_e = align_sent.words.__len__()
        l_f = align_sent.mots.__len__()

        for j, en_word in enumerate(align_sent.words):
            # Initialize the maximum probability with Null token
            max_align_prob = (self.probabilities[en_word][None] * self.alignments[0][j + 1][l_e][l_f], None)
            for i, fr_word in enumerate(align_sent.mots):
                # Find out the maximum probability
                max_align_prob = max(max_align_prob, (
                self.probabilities[en_word][fr_word] * self.alignments[i + 1][j + 1][l_e][l_f], i))

            # If the maximum probability is not Null token,
            # then append it to the alignment.
            if max_align_prob[1] is not None:
                alignment.append((j, max_align_prob[1]))

        return AlignedSent(align_sent.words, align_sent.mots, alignment)


    # Implement the EM algorithm. num_iters is the number of iterations. Returns the
    # translation and distortion parameters as a tuple.
    def train(self, aligned_sents, num_iters):

        # Get initial translation probability distribution
        # from a few iterations of Model 1 training.
        t_ef = IBMModel1(aligned_sents, 4).probabilities

        aligned_sents_fe = []
        for aligned_sent in aligned_sents:
            aligned_sents_fe.append(aligned_sent.invert())

        t_fe = IBMModel1(aligned_sents_fe, 8).probabilities

        # Vocabulary of each language
        fr_vocab = set()
        en_vocab = set()
        for aligned_sent in aligned_sents:
            en_vocab.update(aligned_sent.words)
            fr_vocab.update(aligned_sent.mots)
        fr_vocab.add(None)
        en_vocab.add(None)
        # t_ef = defaultdict(lambda: defaultdict(lambda: 1 / (len(en_vocab))))
        #t_fe = defaultdict(lambda: defaultdict(lambda: 1 / (len(fr_vocab))))

        align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))
        r_align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))

        # Initialize the distribution of alignment probability,
        # a(i|j,l_e, l_f) = 1/(l_f + 1)
        for aligned_sent in aligned_sents:
            en_set = [None] + aligned_sent.words
            fr_set = [None] + aligned_sent.mots
            l_f = len(fr_set) - 1
            l_e = len(en_set) - 1
            initial_value = 1 / (l_f + 1)
            for i in range(0, l_f + 1):
                for j in range(1, l_e + 1):
                    align[i][j][l_e][l_f] = initial_value
            r_initial_value = 1 / (l_e + 1)
            for i in range(0, l_e + 1):
                for j in range(1, l_f + 1):
                    r_align[i][j][l_f][l_e] = r_initial_value

        # EM iterations
        for k in xrange(num_iters):
            count_ef = defaultdict(lambda: defaultdict(lambda: 0.0))
            count_fe = defaultdict(lambda: defaultdict(lambda: 0.0))
            total_f = defaultdict(lambda: 0.0)
            count_align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))
            total_align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
            r_count_align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))
            r_total_align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
            total_e = defaultdict(lambda: 0.0)
            denominator = defaultdict(lambda: 0.0)
            r_denominator = defaultdict(lambda: 0.0)

            for aligned_sent in aligned_sents:
                en_set = [None] + aligned_sent.words
                fr_set = [None] + aligned_sent.mots
                l_f = len(fr_set) - 1
                l_e = len(en_set) - 1

                # compute denominators
                for j in xrange(1, l_e + 1):
                    en_word = en_set[j]
                    denominator[en_word] = 0
                    for i in xrange(0, l_f + 1):
                        denominator[en_word] += t_ef[en_word][fr_set[i]] * align[i][j][l_e][l_f]

                for j in xrange(1, l_f + 1):
                    fr_word = fr_set[j]
                    r_denominator[fr_word] = 0
                    for i in range(0, l_e + 1):
                        r_denominator[fr_word] += t_fe[fr_word][en_set[i]] * r_align[i][j][l_f][l_e]

                # add up counts
                for j in xrange(1, l_e + 1):
                    en_word = en_set[j]
                    for i in xrange(l_f + 1):
                        fr_word = fr_set[i]
                        delta = t_ef[en_word][fr_word] * align[i][j][l_e][l_f] / denominator[en_word]
                        count_ef[en_word][fr_word] += delta
                        total_f[fr_word] += delta
                        count_align[i][j][l_e][l_f] += delta
                        total_align[j][l_e][l_f] += delta

                for j in xrange(1, l_f + 1):
                    fr_word = fr_set[j]
                    for i in xrange(l_e + 1):
                        en_word = en_set[i]
                        delta = t_fe[fr_word][en_word] * r_align[i][j][l_f][l_e] / r_denominator[fr_word]
                        count_fe[fr_word][en_word] += delta
                        total_e[en_word] += delta
                        r_count_align[i][j][l_f][l_e] += delta
                        r_total_align[j][l_f][l_e] += delta

            # Get average count
            for en in count_ef:
                for fr in count_ef[en]:
                    count_ef[en][fr] = (count_ef[en][fr] + count_fe[fr][en]) / 2
                    count_fe[fr][en] = count_ef[en][fr]

            for aligned_sent in aligned_sents:
                src_set = [None] + aligned_sent.words
                tar_set = [None] + aligned_sent.mots
                src_l = len(src_set) - 1
                tar_l = len(tar_set) - 1
                for j in xrange(1, src_l + 1):
                    for i in xrange(tar_l + 1):
                        count_align[i][j][src_l][tar_l] = (count_align[i][j][src_l][tar_l] + r_count_align[j][i][tar_l][
                            src_l]) / 2
                        r_count_align[j][i][tar_l][src_l] = count_align[i][j][src_l][tar_l]

            # estimate probabilities
            t_ef = defaultdict(lambda: defaultdict(lambda: 0.0))
            align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))
            t_fe = defaultdict(lambda: defaultdict(lambda: 0.0))
            r_align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))

            # Estimate the new lexical translation probabilities
            for f in fr_vocab:
                for e in en_vocab:
                    t_ef[e][f] = count_ef[e][f] / total_f[f]

            for e in en_vocab:
                for f in fr_vocab:
                    t_fe[f][e] = count_fe[f][e] / total_e[e]

            # Estimate the new alignment probabilities
            for aligned_sent in aligned_sents:
                en_set = [None] + aligned_sent.words
                fr_set = [None] + aligned_sent.mots
                l_f = len(fr_set) - 1
                l_e = len(en_set) - 1
                for i in xrange(0, l_f + 1):
                    for j in xrange(1, l_e + 1):
                        align[i][j][l_e][l_f] = count_align[i][j][l_e][l_f] / total_align[j][l_e][l_f]
                for i in xrange(0, l_e + 1):
                    for j in xrange(1, l_f + 1):
                        r_align[i][j][l_f][l_e] = r_count_align[i][j][l_f][l_e] / r_total_align[j][l_f][l_e]

        return t_ef, align


def main(aligned_sents):
    ba = BerkeleyAligner(aligned_sents, 10)
    A.save_model_output(aligned_sents, ba, "ba.txt")
    avg_aer = A.compute_avg_aer(aligned_sents, ba, 50)

    print ('Berkeley Aligner')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
