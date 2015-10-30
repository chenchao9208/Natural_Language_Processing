from nltk.compat import python_2_unicode_compatible

# Since the feature extractor is now modified, it's set to True.
printed = True

@python_2_unicode_compatible
class FeatureExtractor(object):
    @staticmethod
    def _check_informative(feat, underscore_is_informative=False):
        """
        Check whether a feature is informative
        """

        if feat is None:
            return False

        if feat == '':
            return False

        if not underscore_is_informative and feat == '_':
            return False

        return True

    @staticmethod
    def find_left_right_dependencies(idx, arcs):
        left_most = 1000000
        right_most = -1
        dep_left_most = ''
        dep_right_most = ''
        for (wi, r, wj) in arcs:
            if wi == idx:
                if (wj > wi) and (wj > right_most):
                    right_most = wj
                    dep_right_most = r
                if (wj < wi) and (wj < left_most):
                    left_most = wj
                    dep_left_most = r
        return dep_left_most, dep_right_most

    @staticmethod
    def count_left_right_dependencies(idx, arcs):
        count_left = 0
	count_right = 0
        # for every arc in the list, check whether the head is the input.
        for (wi, r, wj) in arcs:
            if wi == idx:
                # dependency one the right
                if wj > wi:
                    count_right += 1
                # dependency on the left
                if wj < wi:
                    count_left += 1
                   
        return count_left, count_right

    @staticmethod
    def extract_features(tokens, buffer, stack, arcs):
	"""
        This function returns a list of string features for the classifier

        :param tokens: nodes in the dependency graph
        :param stack: partially processed words
        :param buffer: remaining input words
        :param arcs: partially built dependency tree

        :return: list(str)
        """

        """
        Think of some of your own features here! Some standard features are
        described in Table 3.2 on page 31 of Dependency Parsing by Kubler,
        McDonald, and Nivre

        [http://books.google.com/books/about/Dependency_Parsing.html?id=k3iiup7HB9UC]
        """

        result = []


        global printed
        if not printed:
            print("This is not a very good feature extractor!")
            printed = True

        # features related to the stack is processed as follows
        if stack:
	    # dealing with STK[0]
            stack_idx0 = stack[-1]
            token = tokens[stack_idx0]
            # FORM (word) of STK[0]
            if FeatureExtractor._check_informative(token['word'], True):
                result.append('STK_0_FORM_' + token['word'])
            # feats of STK[0]
            if 'feats' in token and FeatureExtractor._check_informative(token['feats']):
                feats = token['feats'].split("|")
                for feat in feats:
                    result.append('STK_0_FEATS_' + feat)
            if 'lemma' in token and FeatureExtractor._check_informative(token['lemma']):
		lemma = token['lemma']
		result.append('STK_0_LEMMA_' + lemma)
	    if 'tag' in token and FeatureExtractor._check_informative(token['tag']):
		tag = token['tag']
		result.append('STK_0_POSTAG_' + tag)
            if 'ctag' in token and FeatureExtractor._check_informative(token['ctag']):
                ctag = token['ctag']
                result.append('STK_0_CTAG_' + ctag)
           
            # dealing with STK[1]
	    if stack.__len__()>1:
		stack_idx1 = stack[-2]
		token = tokens[stack_idx1]
		if 'tag' in token and FeatureExtractor._check_informative(token['tag']):
		    tag = token['tag']
		    result.append('STK_1_POSTAG_' + tag)
                #if 'ctag' in token and FeatureExtractor._check_informative(token['ctag']):
                #    tag = token['ctag']
                #    result.append('STK_1_CTAG_' + tag)
                #result.append("STK0_STK1_DIST_"+str(abs(tokens[stack_idx0]["address"]-tokens[stack_idx1]['address'])))
                #Count Left, Right dependencies
                #count_left, count_right = FeatureExtractor.count_left_right_dependencies(stack_idx1, arcs)
                #result.append('STK_1_Count_Left_Dep_'+str(count_left))
                #result.append('STK_1_Count_Right_Dep_'+str(count_right))
                #dep_left_most, dep_right_most = FeatureExtractor.find_left_right_dependencies(stack_idx1, arcs)

                #if FeatureExtractor._check_informative(dep_left_most):
                #    result.append('STK_1_LDEP_' + dep_left_most)
                #if FeatureExtractor._check_informative(dep_right_most):
                #    result.append('STK_1_RDEP_' + dep_right_most)
                #if 'ctag' in token and FeatureExtractor._check_informative(token['ctag']):
                #    ctag = token['ctag']
                #    result.append('STK_1_CTAG_' + ctag)	   

	    #Count Left, Right dependencies
	    count_left, count_right = FeatureExtractor.count_left_right_dependencies(stack_idx0, arcs)
	    result.append('STK_0_Count_Left_Dep_'+str(count_left))
	    result.append('STK_0_Count_Right_Dep_'+str(count_right))
            #count_left, count_right = FeatureExtractor.count_left_right_parents(stack_idx0, arcs)
            #result.append('STK_0_Count_Left_Parents_'+str(count_left))
            #result.append('STK_0_Count_Right_Paretns_'+str(count_right))
	    #result.append('STK_0_Count_Dep_'+str(count_left+count_right))
     
            #Left most, right most dependency of stack[0]
            dep_left_most, dep_right_most = FeatureExtractor.find_left_right_dependencies(stack_idx0, arcs)

            if FeatureExtractor._check_informative(dep_left_most):
                result.append('STK_0_LDEP_' + dep_left_most)
            if FeatureExtractor._check_informative(dep_right_most):
                result.append('STK_0_RDEP_' + dep_right_most)
        # dealing with words in the buffer
        if buffer:
            buffer_idx0 = buffer[0]
            token = tokens[buffer_idx0]
            if FeatureExtractor._check_informative(token['word'], True):
                result.append('BUF_0_FORM_' + token['word'])
            if 'feats' in token and FeatureExtractor._check_informative(token['feats']):
                feats = token['feats'].split("|")
                for feat in feats:
                    result.append('BUF_0_FEATS_' + feat)
	    if 'lemma' in token and FeatureExtractor._check_informative(token['lemma']):
                lemma = token['lemma']
                result.append('BUF_0_LEMMA_' + lemma)
            if 'tag' in token and FeatureExtractor._check_informative(token['tag']):
                tag = token['tag']
                result.append('BUF_0_POSTAG_' + tag)
	    if 'ctag' in token and FeatureExtractor._check_informative(token['ctag']):
                ctag = token['ctag']
                result.append('BUF_0_CTAG_' + ctag) 
            # define a new feature to be the combination of word and it's tag.
            if FeatureExtractor._check_informative(token['word'], True) and 'tag' in token and FeatureExtractor._check_informative(token['tag']):
                combine = token['word']+'_'+token['tag']
                result.append('BUF_0_COMBINE_'+combine)
            # dealing with BUF[1]
            if buffer.__len__()>1:   
		buffer_idx1 = buffer[1]
                token = tokens[buffer_idx1]
 		if FeatureExtractor._check_informative(token['word'], True):
                    result.append('BUF_1_FORM_' + token['word'])
                if 'tag' in token and FeatureExtractor._check_informative(token['tag']):
                    tag = token['tag']
                    result.append('BUF_1_POSTAG_' + tag)
            # dealing with BUF[2]
            #if buffer.__len__()>2:
            #    buffer_idx2 = buffer[2]
            #    token = tokens[buffer_idx2]
            #    if 'tag' in token and FeatureExtractor._check_informative(token['tag']):
            #        tag = token['tag']
            #        result.append('BUF_2_POSTAG_' + tag)
            # dealing with BUF[3]
            if buffer.__len__()>3:
                buffer_idx3 = buffer[3]
                token = tokens[buffer_idx3]
                #if FeatureExtractor._check_informative(token['word'], True):
                #    result.append('BUF_3_FORM_' + token['word'])
                if 'tag' in token and FeatureExtractor._check_informative(token['tag']):
                    tag = token['tag']
                    result.append('BUF_3_POSTAG_' + tag)
	  
            dep_left_most, dep_right_most = FeatureExtractor.find_left_right_dependencies(buffer_idx0, arcs)

            if FeatureExtractor._check_informative(dep_left_most):
                result.append('BUF_0_LDEP_' + dep_left_most)
            if FeatureExtractor._check_informative(dep_right_most):
                result.append('BUF_0_RDEP_' + dep_right_most)
	if stack and buffer:
	    stack_idx0 = stack[-1]
	    buffer_idx0 = buffer[0]
	    token = tokens[stack_idx0]
            # if the first word on the top of stack is tagged to be a verb, add the distance to the features.
            if token['tag'] in ['VBN', 'VBZ', 'VBP', 'VB', 'VBD']:
                result.append("STK0_BUF0_DIST_"+str(abs(tokens[stack_idx0]["address"]-tokens[buffer_idx0]['address'])))

        return result
