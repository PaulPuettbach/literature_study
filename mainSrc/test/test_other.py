import parameters

with open('keywords.txt', 'r') as file:
    added_stopwords = ['graph']
    query = """
        SELECT
            id, title, abstract, doi, n_citations
        FROM 
            publications 
        WHERE\n"""
    data = file.read().splitlines()
    for line in data[1:]:
        line = line.split(', ')
        added_stopwords.extend(line)
        query += "\t(lower(title) LIKE ANY (array ["
        for word in line:
            query += "'%" + str(word) + "%', "
        query = query[:-2]
        query += "]) OR lower(abstract) LIKE ANY (array ["
        for word in line:
            query += "'%" + str(word) + "%', "
        query = query[:-2]
        query += "])) AND\n"
    query += "\t(lower(title) NOT LIKE '%journal%' AND lower(title) NOT LIKE '%proceedings%' AND lower(title) NOT LIKE '%keynote%')"
print(added_stopwords)
