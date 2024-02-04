import nltk
import math
import json
from nltk.corpus import stopwords
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#nltk.download('stopwords')

def count_tf(logs,term):
   return logs.count(term)

def IDF (d_total, df):
    result = 0.0
    if (df > 0 ):
         result = math.log10(d_total/df)
    return result

def weight(tf,idf):
     return tf*idf

def cosine_sim(query_vectors,document_vectors,idoc):
     sim = 0.0
     nominator =0.0   
     norm_query_vectors = 0.0
     norm_document_vectors = 0.0
     denominator =0.0
     for token in query_vectors.keys():          
           nominator += query_vectors[token] * document_vectors[token][idoc]           
           norm_query_vectors += math.pow(query_vectors[token] , 2)
           norm_document_vectors += math.pow(document_vectors[token][idoc] , 2)

     denominator = math.sqrt(norm_query_vectors) * math.sqrt(norm_document_vectors)
    
     if (denominator!=0.0): sim = nominator / denominator
     return sim


logs =  [
            "Pembaruan izin akses untuk petugas baru",
            "Tim keamanan diaktifkan,percobaan investigasi sedang berlangsung",
            "Percobaan akses yang tidak sah"
        ]

dTotal = len(logs) 

print("Ada {} Dokumen".format(dTotal))

logs =  [log.lower() for log in logs ]
list_stopwords =  set(stopwords.words('indonesian'))
 
 
factory = StemmerFactory()
stemmer = factory.create_stemmer()

temp_token_list  = list()
new_logs = list()

for idx,log in enumerate(logs):
    tokens = nltk.tokenize.word_tokenize(log)     
    cleaned_tokens = [stemmer.stem(token) for token in tokens if token not in list_stopwords ]
    new_logs.append(' '.join(cleaned_tokens))    
    temp_token_list.extend(cleaned_tokens)

temp_token_list.remove("") #empty string
#token_list = list(filter(None, list(set(temp_token_list)))) #remove duplicates and empty string
token_list =  list(set(temp_token_list)) #remove duplicates 

#print("cleaned log ", new_logs)
print("Token List  " , token_list)

term_frequencies = [0] * dTotal
df_init_values = [0] * len(token_list)

document_frequencies = dict(zip(token_list, df_init_values))

weights = dict()
idf = dict()

for token in token_list:
    for idx,log in enumerate(new_logs):
            tokens = nltk.tokenize.word_tokenize(log)                   
            term_frequencies[idx] = count_tf(tokens,token)
           # print("TF:",term_frequencies)
            document_frequencies[token] += 1 if term_frequencies[idx] >=1 else 0 ;
    
    idf[token] = IDF(dTotal,document_frequencies[token] ) 
    weights[token] = dict()
    for idx,log in enumerate(new_logs):       
        weights[token][idx] = weight( term_frequencies[idx],idf[token])

print(json.dumps(weights, sort_keys=True, indent=2))

query = "Percobaan akses yang tidak sah"
print("Query Pencarian :  " , query)
query = query.lower()
tokens = nltk.tokenize.word_tokenize(query)   
cleaned_tokens = [stemmer.stem(token) for token in tokens if token not in list_stopwords ]
new_query = ' '.join(cleaned_tokens)
print(new_query)

query_weights = dict()

for token in cleaned_tokens:
    tokens = nltk.tokenize.word_tokenize(new_query)                   
    tf  = count_tf(tokens,token)
    query_weights[token] = weight(tf,idf[token])

print("Bobot Query :  " , query_weights) 

distance = dict()

for idx,log in enumerate(new_logs):
     distance[idx] = cosine_sim(query_weights,weights,idx)


sorted_distance = sorted(distance.items(), key=lambda x:x[1], reverse=True)
distance = dict(sorted_distance)
#print(distance)

print("Dokumen yang memiliki kemiripan:")
for id_doc in distance.keys():    
     print("{} ({} %)".format(logs[id_doc],distance[id_doc]*100))
          


 






#print(len(token_list) )

# tf_list =[]
# for tokens in token_list:     
#     tf_list.append(nltk.FreqDist(tokens))

# for tfreqs in tf_list:
#     print(tfreqs.most_common())
     


 