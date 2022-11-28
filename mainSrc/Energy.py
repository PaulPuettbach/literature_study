import psycopg2
import re
import gensim
import nltk
import pandas.io.sql as psql
from wordcloud import WordCloud
import gensim.corpora as corpora
from gensim.models import Phrases

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
        (lower(title) LIKE ANY (array ['%energy%', '%power%', 'green']) OR lower(abstract) LIKE ANY (array ['%energy%', '%power%', 'green'])) AND
        (lower(title) LIKE '%graph processing%' OR lower(abstract) LIKE '%graph processing%')
    LIMIT 100

    """

result = psql.read_sql(query, db)
# result = result.drop(columns=['id', 'doi'], axis=1)


result['title'] = result['title'].map(lambda x: x.split())
result['abstract'] = result['abstract'].map(lambda x: x.split())
for idx in range(len(result['title'])):
    result['abstract'][idx].extend(result['title'][idx])


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
for line in result:
    for idx in range(len(line)):
        for token in bigram[line[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                print(prior[idx])
                prior[idx].append(token)



stop_words = stopwords.words('english')
#for scientific words
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

#to filter out words from the query
stop_words.extend(['serverless', 'cloud', 'cluster', 'data-center', 'energy', 'power', 'green', 'graph processing', 'graph'])

# remove stop words
result['abstract'] = result['abstract'].map(lambda line: list(filter(lambda token: token not in stop_words, line)))
prior = []
for line in result.title.values:
    prior.extend(line)
for line in result.abstract.values:
    prior.extend(line)


long_string = ','.join(prior)
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
cloud = wordcloud.generate(long_string)
# Visualize the word cloud
cloud.to_file("mainSrc/img/cloud_vis.png")

# Create Dictionary
id2word = corpora.Dictionary(result['abstract'])

#remove extremes (words that occur less than 1% docs or more than in 50%)
one_percent_of_docs =  len(result['abstract'])/100 if len(result['abstract'])/100 >= 1 else  1
id2word.filter_extremes(no_below= one_percent_of_docs, no_above=0.8)
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in result['abstract']]



# Set training parameters.
num_topics = 5
passes = 50
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make an index to word dictionary.

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


# get_document_topics(bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False)
# list of (int, float) – Topic distribution for the whole document. Each element in the list is a pair of a topic’s id, and the probability that was assigned to it.
title = "doc"
top_doc_for_topic = [[title,0.0]for i in range(num_topics)]

for i in range(len(result['abstract'])):
    topic_distribution_for_doc = model.get_document_topics(id2word.doc2bow(result['abstract'][i]),minimum_probability=0, per_word_topics=False)
    for idx in range(num_topics):
        if top_doc_for_topic[idx][1] < topic_distribution_for_doc[idx][1]:
            title = " ".join(result['title'][i])
            top_doc_for_topic[idx][0] = title
            top_doc_for_topic[idx][1] = topic_distribution_for_doc[idx][1]

print(" ------------------ number of citations ------------------ ")

print(" ------------------ final top document for each topic ------------------ ")
print(top_doc_for_topic)




