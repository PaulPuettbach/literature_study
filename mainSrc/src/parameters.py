# these are all the mutable parameters for the methodology and a brief explanation

"""for the NLP preprocessing"""
#the number of minimal cooccurance of two words in order for them to be considered a bigram
min_count=10


"""for the LDA model"""
#the number of topics of the model this is the number of papers that will be read later
num_topics = 8

# the number of passes per iteration
passes = 100

# the number of iterations
iterations = 800

# the number of models that will be added together in the ATM. Note that the memory overhead grows exponationally with increasing the number of models
n_models = 7

"""for the paper selection process"""

#topic adherence and citation normalized should summ to one

# topic adherence is how prevelant the topic is for the current document
topic_adherence = 0.9

# is the number of citations of the current paper normalized with the sum of all citations for the entire dataset
citation_normalized = 0.1