import tarfile
import preprocess as p
import os
import pandas as pd
from collections import Counter

tar = tarfile.open("enron1.tar.gz")
tar.extractall('./train/')
tar.close()

n_spam = len(os.listdir("train/enron1/spam"))
n_ham = len(os.listdir("train/enron1/ham"))

spam = p.extract_words("train/enron1/spam")
ham = p.extract_words("train/enron1/ham")

spam_dict=p.CountFrequency(spam)
ham_dict=p.CountFrequency(ham)
try:
  os.makedirs('model')
except:
  pass
p.create_csv('model/spam.csv',spam_dict)
p.create_csv('model/ham.csv',ham_dict)

#spam_data = pd.read_csv('model/spam.csv')
#ham_data = pd.read_csv('model/ham.csv')
#spam_dict = spam_data.to_dict(orient ='list')
#ham_dict = ham_data.to_dict(orient ='list')

spam_dict = dict(Counter(spam_dict).most_common(5000))
ham_dict = dict(Counter(ham_dict).most_common(5000))

vocab = Counter(spam_dict) + Counter(ham_dict)
vocab_len = len(vocab)
print(vocab_len)
spam_t =0
ham_t =0
for k,v in spam_dict.items():
  spam_t += v

for k,v in ham_dict.items():
  ham_t += v
print(spam_t,ham_t)

def predict(test_list):
  probs=1 #n_spam/(n_spam+ n_ham)
  probh = 1 # n_ham/(n_spam+ n_ham)
  for i in test_list:
    if i in spam_dict.keys():
      probs = probs * ((spam_dict[i]+1)/(spam_t+vocab_len))
    else :
      probs = probs * ((0+1)/(spam_t+vocab_len))

    if i in ham_dict.keys():
      probh = probh * ((ham_dict[i]+1)/(ham_t+vocab_len))
    else:
      probh = probh * ((0+1)/(ham_t+vocab_len))


  if(probs > probh ):
    return 1
  else:
    return 0


correct=0
wrong =0

for i in os.listdir('train/enron1/spam')[:80]:
  filename = "train/enron1/spam/"+i
  if(i.endswith('txt')):
    xtest= p.prediction_preprocess(filename)
    if 1==predict(xtest):
      correct +=1
    else:
      wrong +=1
accuracy = correct/(correct+wrong)
print("accuracy: " ,accuracy)
correct=0
wrong =0
for i in os.listdir('train/enron1/ham')[:80]:
  filename = "train/enron1/ham/"+i
  if(i.endswith('txt')):
    xtest= p.prediction_preprocess(filename)
    if 0==predict(xtest):
      correct +=1
    else:
      wrong +=1

accuracy = correct/(correct+wrong)
print("accuracy: " ,accuracy)

output = list()
f=open("output.txt", "w")
for i in os.listdir("test"):
  
  filename = "test/"+i

  if(i.endswith('txt')):
    xtest= p.prediction_preprocess(filename)
    f.write(i+" : "+str(predict(xtest))+"\n")
f.close()
