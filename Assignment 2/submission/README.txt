#####################################################################
               COMS 4705 - Natural Language Processing
                          Assignment 2
                   Multilingual Dependency Parsing
#####################################################################
UNI: cc3736
Name: Chao Chen
#####################################################################

1.b How to determine if a dependency graph is projective?
    The approach shown in transitionparser.py could be described as:
    Given the list of arcs of the graph. 
    For each arc in the list, figure out an "undirected" pair(x, y), 
so that one of (x, y) is the head, and the other is the dependent, while
always x<y.
    Then we have a interval (x,y). For each p in (x, y), and for each
q outside of interval [x,y], check out if there is any arc (p, q) or 
(q, p) in the arc list. 
    If yes, then there exists some arc with one end in the interval 
and the other end outside of the interval, thus there must be some 
"crossing edge" between these two arcs, the graph is not projective.
    If no, then there is no "crossing edge" among the arcs, the graph 
is projective. 



1.c Write an example of a sentence which has a projective dependency 
graph and another which does not has one.

Sentence with projective dependency graph:
    TA included an example at the end of the assignment.

Sentence which doesn't have projective dependency graph:
    Amy pulls the luggage slowly which is too heavy for a little girl.



2.b The result of "python test.py" is:
            UAS: 0.229038040231 
            LAS: 0.125473013344
    The evaluation metric LAS here is Labeled Attachment Score, which 
indicates the proportion of "scoring" tokens that are assigned both 
the correct head and the correct dependency relation label. Here LAS 
is very low, and only about 12.5% of the tokens are correctly assigned.
Thus the overall performance is very poor. The badfeatures.model does
not provide a good performance.



3.a
More than 20 kinds of features are extracted from the configuration. 
Here I list three features of them as follows:

i. STK_0_LDEP (Samme to STK_0_RDEP)
    Note the last node on the stack is STK_0. This feature means the 
number of dependents(children) of STK_0 which are on the left of STK_0
(with address smaller than STK_0).

Implementation:
    1. Get the last element STK_0 on the stack, which is an index(address) 
of a token. 
    2. For each arc (wi, r, wj) in the arc list, check if the head of the 
arc wi is STK_0, and the dependent wj is on the left of the head 
(wj < wi). If these two conditions are satisfied, we add 1 to the counter.
    3. Return the counter and add it to the feature result.

Complexity:
    Assume the length of arc list is L, then
    For the four steps above in Implementation:
    1. T = O(1)
    2. T = O(L)
    3. T = O(1)
    Thus in total T = O(L), where L is the length of the arc list.

Performance:
    Here take the English data set for example. With the model for English 
dataset, when including this feature, it got LAS =  0.716049382716; when 
excluding this feature, it got LAS = 0.698765432099. Thus in this case it 
introduces improvement of about 2% in LAS.

ii. STK_0_POSTAG
    It means the fine-grained part-of-speech tag of the last node on 
the stack.
Implementation:
    1. Get the last element on the stack, which is an index.
    2. Get the token STK_0 from the token list with the index.
    3. If STK_0 has a 'tag', and the tag is informative and is not '_', then
add the tag in the feature result.

Complexity:
    1. T = O(1)
    2. T = O(1)
    3. T = O(1)
    Thus the total time complexity T = O(1), at a constant level.

Performance:
    Here take the English data set for example. With the model for English 
dataset, when including this feature, it got LAS =  0.716049382716; when 
excluding this feature, it got LAS = 0.698765432099. Thus in this case it 
introduces improvement of about 2% in LAS.

iii. STK_0_BUF_0_DIST
    This feature is added only when the following precondition is satisfied:
    The last node on the stack indicates a token with a tag of "verb", such 
as "VBN", "VBZ", "VBP", "VB", "VBD".
    The feature means the difference between the addresses(index) of the 
last node on the stack and the first node in the buffer.

Implementation:
    1. Get the last element on the stack and the first element in the buffer.
    2. Get the corresponding tokens STK_0 and BUF_0 from the token list.
    3. If STK_0 is tagged as a "verb" ("VBN", "VBZ", ...), add the absolute
difference between the address of STK_0 and BUF_0 to the feature result.

Complexity:
    1. T = O(1)
    2. T = O(1)
    3. T = O(1)
    Thus the total T = O(1). Time complexity at a constant level.

Performance:
   Here take the English data set for example. With the model for English 
dataset, when including this feature, it got LAS =  0.716049382716; when 
excluding this feature, it got LAS = 0.696296296296. Thus in this case it 
introduces improvement of about 2% in LAS.


3.c
The scores might be a bit different among several running times.
English:
    UAS: 0.743209876543 
    LAS: 0.711111111111

Danish:
    UAS: 0.799401197605 
    LAS: 0.717764471058

Swedish:
    UAS: 0.795060744872 
    LAS: 0.682135032862

Korean:
    UAS: 0.749710312862 
    LAS: 0.612205484743

3.d 
The time cost by the arc-eager shift-reduce parser depends on whether the dependency graph of the sentence is projective or not. For projective 
sentence dependency graphes, it takes linear time complexity T = O(N), where
N is the length of the sentence. While for non projective sentence dependency
graphs, it will take less time. Bucause the arc-eager shift-reduce parser only parses out
projective dependency graphes.



4. To use the english.model generated by my code, you have to comment out the
'ctag' features in feature_extractor, and train a new model. Otherwise it 
will have problem with the first sentence in englishfile


