import tarfile
import os
import nltk
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
import csv
   
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def extract_words(folder):
  words_list =list()
  for i in os.listdir(folder):
    if(i.endswith('txt')):
      with open(folder+"/"+i, "r",encoding="utf-8",errors='ignore') as f:
        for line in f: 
          word_tokens = word_tokenize(line) 
          filtered_sentence = [w.lower() for w in word_tokens if not w in stop_words]  
          filtered_sentence = [w for w in filtered_sentence if len(w)>1]
          filtered_sentence = [w for w in filtered_sentence if w[0]!='{' or w[0]!='/' or w[0]!='<']
          filtered_sentence = [lemmatizer.lemmatize(w) for w in filtered_sentence if w.isdigit()==0 ]
          
          words_list.extend(filtered_sentence)
  return words_list   
def CountFrequency(my_list): 
    
    freq = {} 
    for items in my_list: 
        freq[items] = my_list.count(items) 

    return freq

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

  


def create_csv(csv_file,words_list):
  keys = list(words_list.keys())
  words_list = [words_list]
  try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        for data in words_list:
            writer.writerow(data)
  except IOError:
      print("I/O error")
