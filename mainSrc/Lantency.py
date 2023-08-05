import psycopg2
import re
import gensim
import math
import nltk
import numpy as np
import pandas.io.sql as psql
from wordcloud import WordCloud
import gensim.corpora as corpora
from gensim.models import Phrases
from gensim.test.utils import datapath
import matplotlib.pyplot as plt

#inspired by: https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#sphx-glr-auto-examples-tutorials-run-lda-py

nltk.download('stopwords')
from nltk.corpus import stopwords

db = psycopg2.connect(user="postgres",
                      password="postgres",
                      host="127.0.0.1",
                      port="4444",
                      database="aip")

query = """
    SELECT
        id, title, abstract, doi, n_citations
    FROM 
        publications 
    WHERE 
        (lower(title) LIKE ANY (array ['%serverless%', '%cloud%', '%cluster%', '%data center%']) OR lower(abstract) LIKE ANY (array ['%serverless%', '%cloud%', '%cluster%', '%data center%'])) AND
        (lower(title) LIKE ANY (array ['%latency%', '%runtime%', '%time complexity%', '%time-complexity%', '%TTC%', '%time to completion%']) OR lower(abstract) LIKE ANY (array ['%latency%', '%runtime%', '%time complexity%', '%time-complexity%', '%TTC%', '%time to completion%'])) AND
        (lower(title) LIKE '%graph processing%' OR lower(abstract) LIKE '%graph processing%') AND
         (lower(title) NOT LIKE '%journal%' AND lower(title) NOT LIKE '%proceedings%' AND lower(title) NOT LIKE '%keynote%')

    """
class LDAModel:
    def __init__(self, model, phi, theta):
        self.model = model
        self.phi = phi
        self.theta = theta

result = psql.read_sql(query, db)

n_cite = result['n_citations'].sum()
max_cite = result['n_citations'].max()
mean_cite = result['n_citations'].mean()
median_cite = result['n_citations'].median()

result['citations_normalized'] = result['n_citations'] / n_cite 


result['title'] = result['title'].map(lambda x: x.split())
result['abstract'] = result['abstract'].map(lambda x: x.split())
for idx in range(len(result['title'])):
    result['abstract'][idx].extend(result['title'][idx])

print(" ------------------ Latency metrics ------------------ ")
print("number of papers ", result.shape[0])
print("sum of citations ", n_cite)
print("max of citations ", max_cite)
print("median of citations ", median_cite)
print("mean of citations ", mean_cite)

exit()



#remove punctuation
result['abstract'] = result['abstract'].map(lambda line: list(map(lambda x: re.sub('[,\.:;!?()-]', '', x), line)))


# #lowecase everything
result['abstract'] = result['abstract'].map(lambda line: list(map(lambda x: x.lower(),line)))


#remove numbers and numbers that are part of 
result['abstract'] = result['abstract'].map(lambda line: list(filter(lambda token: token.isalpha(), line)))


# # #dont know if to stem or lemmatize i think neither

# # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
prior = []
for line in result.abstract.values:
    prior.extend(line)

bigram = Phrases(prior, min_count=10)
for line in result['abstract']:
    for idx in range(len(line)):
        for token in bigram[line[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                print("add a bigram")
                line[idx].append(token)



stop_words = stopwords.words('english')
#for scientific words
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

#to filter out words from the query
stop_words.extend(['serverless', 'cloud', 'cluster', 'data-center', 'latency', 'runtime', 'time complexity', 'time-complexity', 'TTC', 'time to completion', 'graph processing', 'graph'])

# remove stop words
result['abstract'] = result['abstract'].map(lambda line: list(filter(lambda token: token not in stop_words, line)))

# long_string = ','.join(prior)
# # Create a WordCloud object
# wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# # Generate a word cloud
# cloud = wordcloud.generate(long_string)
# # Visualize the word cloud
# cloud.to_file("mainSrc/img/cloud_vis.png")

# Create Dictionary
id2word = corpora.Dictionary(result['abstract'])

#remove extremes (words that occur less than 1% docs or more than in 50%)
one_percent_of_docs =  len(result['abstract'])/100 if len(result['abstract'])/100 >= 1 else  1
id2word.filter_extremes(no_below= one_percent_of_docs, no_above=0.8)
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in result['abstract']]



num_topics = 8
passes = 100
iterations = 800
eval_every = 1  # Don't evaluate model perplexity, takes too much time.

"""
the main idea is this:
generate 2 models all with the same parameters compare them to each other and
1. find the topics that are most like each other to do that
    take topic of model a find the one it is most like if that topic recipocates we decide it is the same topic if it doesnt
    it odd one out skip it for now this reduces the space necesarrily since there cannot be deadlock
    now we run the next round with reduced space
    diff(other, distance='kullback_leibler', num_words=100, n_ann_terms=10, diagonal=False, annotation=True, normed=True)
    take all permutations of topic matching and take the summed and normalized individual values
    this would mean ordered sampeling with replacement with 4 models that is going to be 8 topics to the power of 4 models which is 4096
    2 topics a 3 models
    01 02 03 12 13 23
    21 31 32 41 42 43 51 52 53 54
    """
n_models = 7
modelarray = []

for model_i in range(n_models):
    model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=id2word,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )
    #generate phi array for the model also has to be normalized along the y axis since it is topics along the rows and tokens along the collumns
    #dimensions are n_topics, n_tokens
    lambda_values = model.state.get_lambda()
    phi_array = np.apply_along_axis(lambda x: x/x.sum(),0,lambda_values)


    #generate the theta array from the model
    theta_array = np.reshape(np.array(model.get_document_topics(id2word.doc2bow(result['abstract'][0]),minimum_probability=0, per_word_topics=False), dtype="i,f"), (1,8))
    for count in range(len(result['abstract'])):
        if count == 0:
            continue
        theta_array = np.append(theta_array, np.reshape(np.array(model.get_document_topics(id2word.doc2bow(result['abstract'][count]),minimum_probability=0, per_word_topics=False), dtype="i,f"), (1,8)), axis=0)
    modelarray.append(LDAModel(model, phi_array, theta_array))
    # #-------------------------test----------------------------------------------------
    # title = "doc"
    # doi = "00.0000/journal.year"
    # top_doc_for_topic = [[title,0.0,doi]for i in range(num_topics)]
    # importance_metric = 0.0
    # for i in range(len(result['abstract'])):
    #     topic_distribution_for_doc = model.get_document_topics(id2word.doc2bow(result['abstract'][i]),minimum_probability=0, per_word_topics=False)
    #     for idx in range(num_topics):
    #         #we weigh the topic adherence at 90 percent and the citation normalized at 10 percent)
    #         importance_metric = ((topic_distribution_for_doc[idx][1] *0.9) + (result['citations_normalized'][i] *0.1))
    #         if top_doc_for_topic[idx][1] < importance_metric:
    #             title = " ".join(result['title'][i])
    #             top_doc_for_topic[idx][0] = title
    #             top_doc_for_topic[idx][1] = importance_metric
    #             top_doc_for_topic[idx][2] = result['doi'][i]
    # print("-------------this is for model {}--------------".format(model_i))
    # for doc in top_doc_for_topic:
    #     print(doc[0])
    #     print("---------------------------------------------------------")
    # #-----------------------test--------------------------------------------------------------

    for model_j in range(model_i):
        #the ndim is the same as the i
        # 10 20 21 30 31 32 40 41 42 43
        # its always gonna touch the new dimension its then also gonna touch dimension j counted from the right

        #shape i to j 
        to_be_added = modelarray[model_i].model.diff(modelarray[model_j].model, annotation=False)[0]
        if model_i == 1:
            new = to_be_added
            continue
        elif new.ndim == model_i:
            new = np.array([new,new,new,new,new,new,new,new])
        string = "new["
        for dim in range(new.ndim):
            if dim == new.ndim - model_i -1: # we need the ith position counted from the right
                string += "row"
            elif dim == new.ndim - model_j -1:
                string += "collumn"
            else:
                string += ":"
            if dim == new.ndim -1:
                string += "]"
                break
            string += ","
        to_exec = string + "=" + string + "+ to_be_added[row,collumn]"
        for row in range(len(to_be_added)):
            for collumn in range(len(to_be_added[row])):
                exec(to_exec)

chosen_permutations = []
for topic_id in range(num_topics):        
    indices = np.where(new == np.amax(new))
    ziplist = []
    for index in indices:
        ziplist.append(index)
    # zip the 2 arrays to get the exact coordinates
    listOfCordinates = list(zip(*ziplist))
    # travese over the list of cordinates
    chosen_permutations.append(listOfCordinates[0])

    for indx in range(new.ndim):
        string = "new["
        for dim in range(new.ndim):
            if dim == indx:
                string += str(listOfCordinates[0][dim])
            else:
                string += ":"
            if dim == new.ndim -1:
                string += "]"
                break
            string += ","
        string = string + " = 0"
        exec(string)
print("final all")
print(chosen_permutations) # always atleast one

#new models get added as the first dimension so we have to read them from the back for the 
#permutations
for topic in range(len(chosen_permutations)):
    for model in range(len(chosen_permutations[topic])):
        if model == 0:
            continue
        for tuple_id in range(len(modelarray[0].theta[:,chosen_permutations[0][-1]])):
            modelarray[0].theta[:,chosen_permutations[topic][-1]][tuple_id][1] = modelarray[0].theta[:,chosen_permutations[topic][-1]][tuple_id][1] + modelarray[model].theta[:,chosen_permutations[topic][-(model+1)]][tuple_id][1]  




# get_document_topics(bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False)
# list of (int, float) – Topic distribution for the whole document. Each element in the list is a pair of a topic’s id, and the probability that was assigned to it.
title = "doc"
doi = "00.0000/journal.year"
top_doc_for_topic = [[title,0.0,doi]for i in range(num_topics)]
importance_metric = 0.0

for idx in range(num_topics):
        #we weigh the topic adherence at 90 percent and the citation normalized at 10 percent)
    for i in range(len(result['abstract'])):
        topic_distribution_for_doc = modelarray[0].theta[i]
        importance_metric = ((topic_distribution_for_doc[idx][1] *0.9) + (result['citations_normalized'][i] *0.1))
        if top_doc_for_topic[idx][1] < importance_metric:
            title = " ".join(result['title'][i])
            #cut out duplicates
            if title in list(zip(*top_doc_for_topic))[0]:
                continue
            top_doc_for_topic[idx][0] = title
            top_doc_for_topic[idx][1] = importance_metric
            top_doc_for_topic[idx][2] = result['doi'][i]


print(" ------------------ scalability metrics ------------------ ")
print("number of papers ", result.shape[0])
print("sum of citations ", n_cite)
print("max of citations ", max_cite)

rejected_papers = []

while True:

    print(" ------------------ final top document for each topic ------------------ ")
    for doc in top_doc_for_topic:
        print(doc[0])
        print("---------------------------------------------------------")

    print('\n \n Do you Accept ? type \'no\' or hit enter')
    accept = input()

    if(accept == "no"):

        while True:
            print(' \n \n \nchange which paper? enter the number of the paper from the top (starting at 1)')
            temp = input()
            paper_to_exchange = int(temp)

            print("change this paper: {} ? if yes hit enter".format(top_doc_for_topic[paper_to_exchange - 1][0]))
            x = input()
            if x == "":
                break

        rejected_papers.append(top_doc_for_topic[paper_to_exchange - 1][0])
        print("this is the rejected papers {}".format(rejected_papers))

        top_doc_for_topic[paper_to_exchange-1][1] = -1
        top_doc_for_topic[paper_to_exchange-1][0] = "no other docs"
        for i in range(len(result['abstract'])):
            topic_distribution_for_doc = modelarray[0].theta[i]
            importance_metric = ((topic_distribution_for_doc[paper_to_exchange-1][1] *0.9) + (result['citations_normalized'][i] *0.1))
            if top_doc_for_topic[paper_to_exchange-1][1] < importance_metric:
                title = " ".join(result['title'][i])
                #cut out duplicates
                if title in rejected_papers:
                    continue
                if title in list(zip(*top_doc_for_topic))[0]:
                    continue
                top_doc_for_topic[paper_to_exchange-1][0] = title
                top_doc_for_topic[paper_to_exchange-1][1] = importance_metric
                top_doc_for_topic[paper_to_exchange-1][2] = result['doi'][i]
    else:
        break