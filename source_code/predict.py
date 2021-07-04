import pandas as pd
import os
import nltk
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
import csv

spam_data = pd.read_csv('model/spam.csv')
ham_data = pd.read_csv('model/ham.csv')
spam_dict = spam_data.to_dict(orient ='list')
ham_dict = ham_data.to_dict(orient ='list')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

spam_t =176455
ham_t = 319924

vocab_len = 43200

def prediction_preprocess(filename):
  test_list =list()

  with open(filename, "r",encoding="utf-8",errors='ignore') as f:
    for line in f: 
      word_tokens = word_tokenize(line) 
      filtered_sentence = [w.lower() for w in word_tokens if not w in stop_words]  
      filtered_sentence = [w for w in filtered_sentence if len(w)>1]
      filtered_sentence = [w for w in filtered_sentence if w[0]!='{' or w[0]!='/' or w[0]!='<']
      filtered_sentence = [lemmatizer.lemmatize(w) for w in filtered_sentence if w.isdigit()==0]
      test_list.extend(filtered_sentence)
  return test_list

def predict(test_list):
  probs=1 
  probh = 1 
  for i in test_list:
    if i in spam_dict.keys():
      probs = probs * ((spam_dict[i][0]+1)/(spam_t+vocab_len))
    else :
      probs = probs * ((0+1)/(spam_t+vocab_len))

    if i in ham_dict.keys():
      probh = probh * ((ham_dict[i][0]+1)/(ham_t+vocab_len))
    else:
      probh = probh * ((0+1)/(ham_t+vocab_len))


  if(probs > probh ):
    return 1
  else:
    return 0

output = list()
f=open("output.txt", "w")
for i in os.listdir("test"):
  
  filename = "test/"+i

  if(i.endswith('txt')):
    xtest= prediction_preprocess(filename)
    f.write(i+" : "+str(predict(xtest))+"\n")
f.close()
