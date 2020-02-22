#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 13:14:01 2019

@author: linwei xu
"""
from nltk.tag import hmm
from nltk import probability as pb
import sys
import re
import csv

laplace_improved = False
text_process = False

if(len(sys.argv) == 2):
    path = sys.argv[1]
    
if(len(sys.argv) == 3):
    path = sys.argv[2]
    if(sys.argv[1] == '-laplace'):
        laplace_improved = True  
    if(sys.argv[1] == '-lm'):
        text_process = True      
if(len(sys.argv) == 4):
    path = sys.argv[3]
    laplace_improved = True
    text_process = True

##generate path
#test_x_path = str(path)+"/test_cipher.txt"
#test_y_path = str(path)+"/test_plain.txt"
#train_x_path = str(path)+"/train_cipher.txt"
#train_y_path = str(path)+"/train_plain.txt"        
#

#cipher1
test_x_path = "cipher1/test_cipher.txt"
test_y_path = "cipher1/test_plain.txt"
train_x_path = "cipher1/train_cipher.txt"
train_y_path = "cipher1/train_plain.txt"
##

#cipher2
#test_x_path = "cipher2/test_cipher.txt"
#test_y_path = "cipher2/test_plain.txt"
#train_x_path = "cipher2/train_cipher.txt"
#train_y_path = "cipher2/train_plain.txt"
##

#cipher3
#test_x_path = "cipher3/test_cipher.txt"
#test_y_path = "cipher3/test_plain.txt"
#train_x_path = "cipher3/train_cipher.txt"
#train_y_path = "cipher3/train_plain.txt"
##


train_x_sentence = []
train_y_sentence = []
train_x_characters = []
train_y_characters = []
x_list = []
y_list = []
train_data = []
test_x_characters = []
test_y_characters = []
test_x_sentence = []
test_y_sentence = []
test_data = []

with open(train_x_path,"r",encoding="utf-8",errors="ignore") as r:
    for line in r:
        train_x_sentence.append(line)
        characters = list(line)
        for i in characters:
            train_x_characters.append(i)
            if i not in x_list:
                x_list.append(i)
                
with open(train_y_path,"r",encoding="utf-8",errors="ignore") as r:
    for line in r:
        train_y_sentence.append(line)
        characters = list(line)
        for i in characters:
            train_y_characters.append(i)
            if i not in y_list:
                y_list.append(i)
                
with open(test_x_path,"r",encoding="utf-8",errors="ignore") as r:
    for line in r:
        test_x_sentence.append(line)
        characters = list(line)
        for i in characters:
            test_x_characters.append(i)
            
with open(test_y_path,"r",encoding="utf-8",errors="ignore") as r:
    for line in r:
        test_y_sentence.append(line)
        characters = list(line)
        for i in characters:
            test_y_characters.append(i)
            
##trying on sentence segmentation of the 3rd cipher
#test_x_characters = []
#for i in range(0,len(train_x_sentence)):
#    characters = []
#    x_characters = list(train_x_sentence[i])
#    y_characters = list(train_y_sentence[i])
#    for j in range(0,len(x_characters)):
#        characters.append((x_characters[j],y_characters[j]))
#    train_data.append(characters)     
#    
#for i in range(0,len(test_x_sentence)):
#    characters = []
#    test_characters = []
#    x_characters = list(test_x_sentence[i])
#    y_characters = list(test_y_sentence[i])
#    for j in range(0,len(x_characters)):
#        test_characters.append(x_characters[j])
#        characters.append((x_characters[j],y_characters[j]))
#    test_data.append(characters)
#    test_x_characters.append(test_characters)
##

##process the 3rd cipher mode
#train_x_characters = []
#for line in train_x_sentence:
#    characters = list(line)
#    characters_other = characters[:-3]
#    characters_3 = characters[-3:]
#    characters_new = characters_3
#    for i in characters_other:
#        characters_new.append(i)
#    sentence_lenth = len(characters)
#    for j in range(0,sentence_lenth):
#        characters_bi = [characters[j],characters_new[j]]
#        characters_join = ('').join(characters_bi)
#        train_x_characters.append(characters_join)
#            
#test_x_characters = []
#for line in test_x_sentence:
#    characters = list(line)
#    characters_other = characters[:-3]
#    characters_3 = characters[-3:]
#    characters_new = characters_3
#    for i in characters_other:
#        characters_new.append(i)
#    sentence_lenth = len(characters)
#    for j in range(0,sentence_lenth):
#        characters_bi = [characters[j],characters_new[j]]
#        characters_join = ('').join(characters_bi)
#        test_x_characters.append(characters_join)
#        
#hmm_trainer = hmm.HiddenMarkovModelTrainer()
##
            
##Test on mode
#laplace_improved = True
#text_process = True
##
                
### without sentence segemetation
lenth = len(train_x_characters)

for i in range(0,lenth):
    x = train_x_characters[i]
    y = train_y_characters[i]
    train_data.append((x, y))
    
## test 
#fd = pb.ConditionalFreqDist(train_data)
##

train_data = [train_data]

for i in range(0,len(test_x_characters)):
    x = test_x_characters[i]
    y = test_y_characters[i]
    test_data.append((x, y))
    
test_data = [test_data]
##without segemetation

##the processed 3rd cipher can not use this line
states = symbols = x_list
#hmm_trainer = hmm.HiddenMarkovModelTrainer(states, symbols)
hmm_trainer = hmm.HiddenMarkovModelTrainer()
##

#test with other estimators
#lid_estimator = lambda fd, bins: pb.LidstoneProbDist(fd, 0.1, bins)
#lid_estimator = lambda fd: pb.LidstoneProbDist(fd, 0.1, fd.B() + 1)
#pd = lambda fd, bins: pb.ConditionalProbDist(fd, lid_estimator,bins)
#hmm_model = hmm_trainer.train_supervised(train_data, estimator = lid_estimator)

##Laplace Smoothing
if(laplace_improved == True):
    hmm_model = hmm_trainer.train_supervised(train_data, estimator = pb.LaplaceProbDist)
else:
    hmm_model = hmm_trainer.train_supervised(train_data)
##

##try different functions
result = hmm_model.best_path(test_x_characters)
#result = hmm_model.test(test_data)
#result = hmm_model._tag(test_x_characters)
result_acc = hmm_model.evaluate(test_data)
##
results = ('').join(result)
print(results)
print('The accuracy is '+str(result_acc))

##-lm
if(text_process == True):
    stopWords = []
    stopWords.append('``')
    stopWords.append('<')
    stopWords.append('br')
    stopWords.append('/')
    stopWords.append('>')
    stopWords.append('!')
    stopWords.append("''")
    stopWords.append('-')
    stopWords.append('(')
    stopWords.append(')')
    stopWords.append(' ')

    path = "./IMDB_Dataset.csv"
    texts = []
    with open(path,"r",encoding="utf-8",errors="ignore") as r:
        lines = csv.reader(r)
        for line in lines:
            regex = re.compile('[^a-zA-Z,. ]')
            line = regex.sub('', line[0])
            splitted = line.split(' ')
            processed_words = []
            for i in splitted:
                if i not in stopWords:
                    processed_words.append(i)
                comment = (' ').join(processed_words)
            comment = comment.lower()
            comment = comment.strip()
            comment.replace(" ","")
            comment = (' ').join(comment.split())
            texts.append(comment)
        
        
        texts = texts[1:]
    texts = texts + train_y_sentence

    texts_sentences = []
    pattern = re.compile('([a-z \,]+\.)')

    for x in texts:
        results = pattern.findall(comment)
        for j in results:
            texts_sentences.append(j)
        
    starting = pb.FreqDist()
    transitional = pb.ConditionalFreqDist()
    emissional = pb.ConditionalFreqDist()
    pi = pb.FreqDist()

    for row in test_y_sentence:
        pi[row[0]] +=1

    for row in texts_sentences:
        lasts = None
        for ch in list(row):
            if(lasts is not None):
                transitional[lasts][ch] += 1
                lasts = ch
            
    for row in train_data:
        for pair in row:
            emissional[pair[1]][pair[0]] += 1
        
        states = symbols = x_list
        
    estimator = pb.LaplaceProbDist
    n = len(symbols)
    pi = estimator(pi, n)
    a = pb.ConditionalProbDist(transitional, estimator, n)
    b = pb.ConditionalProbDist(emissional, estimator ,n)

    hmm_tagger = hmm.HiddenMarkovModelTagger(states, symbols, a, b, pi)

    result = hmm_tagger.best_path(test_x_characters)
    result_acc = hmm_tagger.evaluate(test_data)

    results = ('').join(result)
    print(results)
    print('The accuracy is '+str(result_acc))

##