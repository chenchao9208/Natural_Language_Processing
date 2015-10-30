####################################################################################
                       COMS 4705 - Natural Language Processing
                                   Assignment 4
                               Machine Translation
####################################################################################
Name: Chao Chen
UNI : cc3736
####################################################################################
Part A:
(3)
Result of IBMModel1 and IBMModel2
------------------------------------------------------------------------------------
IBM Model 1
---------------------------
Average AER: 0.665

IBM Model 2
---------------------------
Average AER: 0.650

Discussion of a sentence pair
------------------------------------------------------------------------------------
( Das Parlament erhebt sich zu einer Schweigeminute . )
( The House rose and observed a minute ' s silence )

Solution Provided:
0-0 1-1 2-2 3-3 3-4 3-5 4-3 5-5 6-6 7-7 7-8 7-9 7-10 8-10 9-11

IBMModel1: AER = 0.75
0-11 1-5 2-5 3-5 4-10 5-10 6-10 7-10 9-11

IBMModel2: AER = 0.666666666667
0-0 1-5 2-3 3-5 4-10 5-9 6-10 7-7 9-11

From the example above, we have that IBM model 2, in this case, outperforms the IBM
model 1 with a lower error rate. The difference between the two models is that: IBM
Model 2 introduces alignment parameters, while IBM Model 1 set all these alignment
parameters to be 1/(l+1)^m. The way that IBM model 2 calculates alignment probabilities
leads to higher accuracy in this sentence case.

(4)
The relations between iter_nums and AERs are listed as follows:
iter_nums |  3  |  4  |  5  |  8  | 10  | 15  | 20  | 25  | 30  | 40  | 50  | 60  | 70  | 80  |
-----------------------------------------------------------------------------------------------
IBM Model1|0.641|0.630|0.627|0.631|0.665|0.665|0.661|0.660|0.660|0.657|0.658|0.660|0.658|0.662|
IBM Model2|0.644|0.642|0.644|0.649|0.650|0.650|0.648|0.649|0.649|0.650|0.654|0.657|0.657|0.657|


We can observe that:
For IBM Model1, the lower bound of AER is 0.627, reached when iter_nums=5.
For IBM Model2, the lower bound of AER is 0.642, reached when iter_nums=4.

The relation between iter_nums and the AER is:
When iter_nums is small, the AER changes obviously with the change of iter_nums. The lower
bound is reached when iter_nums=5 and 4 for model1 and model2 respectively. This lower
bound is reached by chance because of (1) the parameters haven't convergent; (2) The AER is
calculated over a small subset of the sentences.
As the iter_nums increases, the AER of the models tend to be convergent. As shown above, 
when iter_nums>=20, the AER of model1 generally reaches the convergence at around 0.660; 
when iter_nums>=60, the AER of model2 generally reaches the convergence at around 0.657.

####################################################################################
Part B:
(4)
Berkeley Aligner
---------------------------
Average AER: 0.542

Total runing time of (Part A + Part B) is about 389.472847587 seconds.

With the same iteration number 10, the average AERs of the tree models are:
IBMModel 1 			: 0.665
IBMModel 2 			: 0.650
Berkeley Aligner 	: 0.542
Thus Berkeley performs the best among the three models and IBMModel2 performes the worst.

(5)
Comparison on a sentence pair
------------------------------------------------------------------------------------
Ich bitte Sie , sich zu einer Schweigeminute zu erheben . 
Please rise , then , for this minute ' s silence . 

IBMModel 1: AER = 0.75
0-1 1-1 2-1 3-4 4-10 5-10 6-10 7-10 8-10 9-1

IBMModel 2: AER = 0.666666666667
0-0 1-1 2-0 3-2 4-10 5-10 6-10 7-7 8-10 9-0

Berkeley Aligner: AER = 0.6
0-0 1-1 2-0 3-2 4-6 5-10 6-10 7-7 8-10 9-0 10-11

From the example above, we can observe that  Berkeley Aligner performs the best among the 
three models with the lowest error rate. Berkeley Aligner considers the intersection of
predictions of two directioanl models(from English to French & from French to English) to 
reach the agreement. The Berkeley Aligner not only makes the predictions of the models agree
at test time, it also encourages agreement during training. Thus the joint training of the two
models at opposite direction gives better performance.