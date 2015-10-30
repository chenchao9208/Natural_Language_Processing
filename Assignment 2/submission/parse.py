import sys
from providedcode.transitionparser import TransitionParser
from providedcode.dependencygraph import DependencyGraph

if __name__ == '__main__':
    # the raw sentences read from englishfile
    lines = sys.stdin;
 
    # if no sentences read or not enough parameters retrieved, exit the program.
    if not lines or not sys.argv.__len__() == 2:
        exit()
    # put the raw sentences in to dependency graphs and form a list of these graphs.
    sentences = [DependencyGraph.from_sentence(line) for line in lines];  

    model_name = sys.argv[1];

    # load the trained model
    tp = TransitionParser.load(model_name);
    
    # parse the sentences with the model
    parsed = tp.parse(sentences);

    # write the parsed sentences into the output file conll supported format.
    for parsed_line in parsed:
        print parsed_line.to_conll(10).encode('utf-8')

    #sentence = DependencyGraph.from_sentence('Hi, this is a test')
    #tp = TransitionParser.load('english.model')
    #parsed = tp.parse([sentence])
    #print parsed[0].to_conll(10).encode('utf-8')

