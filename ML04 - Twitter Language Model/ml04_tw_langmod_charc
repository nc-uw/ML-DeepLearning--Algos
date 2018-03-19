import os
import csv
import math
import unicodecsv
import re
#import tensorflow as tf
import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix

######DICTIONARY########
#creating dictionary of characters
kv={}
ylang={}
tweets=[]
data=defaultdict(list)
lang=[]
#kv_word={}
filename=os.getcwd()+'/lang_id_data/train.tsv'
with open(filename) as tsvfile:
    reader = unicodecsv.reader(tsvfile, delimiter='\n', encoding='utf-8') #encoding utf8
    #reader = csv.reader(tsvfile, delimiter='\n')
    for row in reader:        
        for tweet in row:
            tweet = tweet.strip('\n').split("\t",1)
            toke='<SS>'+tweet[1]+'</S>'
            toke=tweet[1]
            tweets.append(toke)
            lang.append(tweet[0])
            data[tweet[0]].append(tweet[1])
            for c in tweet[1]:
                kv[c]=kv.setdefault(c, 0)+1

for l in lang:                
    ylang[l]=ylang.setdefault(l,0)+1

#print(kv)
#test=''.join(kv.keys())
#test=''.join(sorted(test))
#print in sorted order all keys in dict
#print test

#filter characters with frequency of atleast 10
kv1={k: v for k, v in kv.iteritems() if v >9}
kv2={k: v for k, v in kv.iteritems() if v <10}

#token for characters with low freq
#start stop tokens
kv1['<Other>']=sum(kv2.values())
kv1['<SS>']=len(tweets)
kv1['</S>']=len(tweets)

def alphavec(dix):  
    word2int={}
    int2word={}
    for i,k in enumerate(dix.keys()):
        word2int[k] = i
        int2word[i] = k
    return word2int, int2word

kv1_word2int, kv1_int2word = alphavec(kv1)
ylang_word2int, ylang_int2word = alphavec(ylang)

print ("size of vocab atleast ten times", len(kv1))
print ("size of out of vocab %", len(kv2))

######Training Perplexity########
#initial perplexity
P=sum(kv1.values())
N=len(tweets)
Xentr=defaultdict(list)
perplex=defaultdict(list)
#excluding sart token for perplex calculation
perplex_dict={}
perplex_dict=kv1
perplex_dict.pop('<SS>', 0)
px={i:-(math.log(float(j)/float(P),2))/N for i,j in perplex_dict.items()}
Xentr=-sum(px.values())/len(kv1)
perplex=pow(2,Xentr)
print ("initial cross-entropy on training data", Xentr)
print ("initial perplexity on training data", perplex)

kv1['<SS>']=len(tweets)

###### MARKOV MODEL ########
###for every language compute aij
###numerator and denominator are calculated separately for efficient computing
Nall=defaultdict(list)
Dall=defaultdict(list)
count_all=0
#x = list(data[ylang.keys()[0]][0])
print ("total tweets", len(tweets))
for l in ylang.keys():
    #l=ylang.keys()[1]   
    N=np.zeros(shape=(len(kv1),len(kv1)))
    D=np.zeros(shape=(len(kv1)))
    count=0
    for tw in data[l]:
        #tw=data[l][18989]
        x = [j if j in kv1.keys() else '<Other>' for j in tw]
        #print ("tweet characters", x)
        N[kv1_word2int['<SS>']][kv1_word2int[x[0]]]+=1
        D[kv1_word2int['<SS>']]+=1
        if len(x)-1 > 0:
            for k in range(len(x)-1):
                N[kv1_word2int[x[k]]][kv1_word2int[x[k+1]]]+=1
                D[kv1_word2int[x[k]]]+=1
            N[kv1_word2int[x[k+1]]][kv1_word2int['</S>']]+=1
            D[kv1_word2int[x[k+1]]]+=1
        else:
            N[kv1_word2int[x[0]]][kv1_word2int['</S>']]+=1
            D[kv1_word2int[x[0]]]+=1
        print("language", l, "tweet in lang:", count, "tweet all", count_all)
        count+=1
        count_all+=1        
    Nall[l]=N
    Dall[l]=D
#NALL=Nall
#DALL=Dall

### Smoothing
### laplacian smoothing for characters with 0 freq
Aall_smooth=defaultdict(list)
#l=u'en'
for l in ylang.keys(): Aall_smooth[l] = (Nall[l]+1)/(Dall[l]+len(kv1))

#for every language compute pie 
Piall=defaultdict(list)
count_all=0
#x = list(data[ylang.keys()[0]][0])
print ("total tweets", len(tweets))
for l in ylang.keys():
    #l=ylang.keys()[1]   
    Pi=np.zeros(shape=(len(kv1)))
    count=0
    for tw in data[l]:
        #tw=data[l][18989]
        j=tw[0]
        if j in kv1.keys(): x = j
        else: x = '<Other>'
        #print ("tweet characters", x)
        Pi[kv1_word2int[x]]+=1
        print("language", l, "tweet in lang:", count, "tweet all", count_all)
        count+=1
        count_all+=1        
    Piall[l]=Pi/len(data[l])
    

######VALIDATION########
val_tweets=[]
val_lang=[]
#read validation data
filename=os.getcwd()+'/lang_id_data/val.tsv'
with open(filename) as tsvfile:
    reader = unicodecsv.reader(tsvfile, delimiter='\n', encoding='utf-8') #encoding utf8
    #reader = csv.reader(tsvfile, delimiter='\n')
    for row in reader:        
        for tweet in row:
            tweet = tweet.strip('\n').split("\t",1)
            toke='<SS>'+tweet[1]+'</S>'
            toke=tweet[1]
            val_tweets.append(toke)
            val_lang.append(tweet[0])

#predict language
pred_lang = []
count=0
correct=0
for tw,val_l in zip(val_tweets,val_lang):
    x = [j if j in kv1.keys() else '<Other>' for j in tw]
    Pi_val=np.array([Piall[l][kv1_word2int[x[0]]] for l in ylang])
    A_val=Pi_val*np.array([Aall_smooth[l][kv1_word2int['<SS>']][kv1_word2int[x[0]]] for l in ylang])
    if len(x)-1 > 0:
        for k in range(len(x)-1):
            A_val = A_val*np.array([Aall_smooth[l][kv1_word2int[x[k]]][kv1_word2int[x[k+1]]] for l in ylang])
        A_val = A_val*np.array([Aall_smooth[l][kv1_word2int[x[k+1]]][kv1_word2int['</S>']] for l in ylang])
    else:
        A_val = A_val*np.array([Aall_smooth[l][kv1_word2int[x[0]]][kv1_word2int['</S>']] for l in ylang])
    pred_lang.append(ylang.keys()[list(A_val).index(max(A_val))]) 
    if pred_lang[count] == val_l: correct+=1
    print("tweet #:", count, "correct?", pred_lang[count] == val_l, "accuracy", 100*(float(correct)/float(count+1)))
    #print("tweet #:", count, "pred lang", pred_lang[count], "actual lang", val_l, "correct?", pred_lang[count] == val_l)
    count+=1    

###### PRECISION AND RECALL (CONFUSION MATRIX) ############
label=[i for i in ylang.keys()]
cm=confusion_matrix(val_lang, pred_lang, labels = label)
cm = cm.astype(np.float32)
sum_4prec=cm.sum(axis=0)
sum_4recall=cm.sum(axis=1)
print (label)
print (cm)
precision = np.diagonal(cm) / sum_4prec
recall = np.diagonal(cm) / sum_4recall
print ("precision", precision)
print ("recall", recall)
