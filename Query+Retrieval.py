#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys,re,os,string
from math import sqrt
from numpy.lib.scimath import logn
import json
import nltk
from nltk.stem.isri import ISRIStemmer

'''--------------------------------------------------------------'''
with open('f:/ir/inverted_index.json') as f:
    inv= json.load(f)
f.close()
with open('f:/ir/docs.json') as f:
    d= json.load(f)
f.close()

asw = open('stopwords.txt').read()
aswUinicode = asw.decode('utf-8')
arabicStopWords =aswUinicode.split()
arabic_punct = ''' ` ÷ × ؛ < > _ ( ) * & ^ % ] [ ـ ، / : " ؟ . , ' { } ~ ¦ + | !  ”  …  “ –   ـ  '''
arabic_diacritics = '''  َ     ُ       ِ      ّ       ً      ٌ       ٍ      ْ     '''
arabic_punctUnicode = arabic_punct.decode('utf-8')
arabic_punct = arabic_punct.split()
arabic_punctUnicode = arabic_punctUnicode.split()
arabic_diacritics_unicode = arabic_diacritics.decode('utf-8')
arabic_diacritics = arabic_diacritics.split()
arabic_diacritics_unicode = arabic_diacritics_unicode.split()

english_punt = list(string.punctuation)
english_puntUnicode = list(string.punctuation.decode('utf-8'))
# Arabic punctuations and dicritis 
punctuations = set( arabic_punct + arabic_punctUnicode + arabic_diacritics + arabic_diacritics_unicode)
# remove punctcutions
def remove_punct(word):
	for c in word: return ''.join(ch for ch in word if not ch in punctuations) # remove punctuation
arstemmer = ISRIStemmer()
def remove_diacritics(text):
	result = arstemmer.norm(text, num=1) #  remove diacritics which representing Arabic short vowels
	return result
def process_text(text, removePunct=True, removeSW=True, removeNum=False):
	text = remove_diacritics(text)# remove arabic diacritics
	word_list = nltk.tokenize.wordpunct_tokenize(text.lower())
	if removePunct:
		word_list = [ w for w in word_list if not w in punctuations ]
		word_list = [ remove_punct(w) for w in word_list ]
	if removeSW: word_list = [ w for w in word_list if not w in arabicStopWords ]
	if removeNum: word_list = [ w for w in word_list if not w.isdigit() ]
	word_list = [ w for w in word_list if w]# remove empty words
	return word_list

# takes a word list and returns the root for each Arabic words
arstemmer = ISRIStemmer()
def isri_heavy(word):	
	root=arstemmer.stem(word)
	return root

# takes a word list and perform light stemming for each Arabic words
def isri_light(word):
	word = arstemmer.norm(word, num=1)      #  remove diacritics which representing Arabic short vowels  
	if not word in arstemmer.stop_words:    # exclude stop words from being processed
	    word = arstemmer.pre32(word)        # remove length three and length two prefixes in this order
	    word = arstemmer.suf32(word)        # remove length three and length two suffixes in this order
	    word = arstemmer.waw(word)          # remove connective ‘و’ if it precedes a word beginning with ‘و’
	    word = arstemmer.norm(word, num=2)  # normalize initial hamza to bare alif
	lstem=word
	return lstem

def query_vector(query):
    vector_space=[]
    tokens = {}
    docs_dict={}
    word_list =query.split()
    words=" "
    for word in word_list:
        word=filter(lambda x: x.isalpha(), word)
        word=word.replace("AR","")
        if (word !="") and (word !="D"):
            words=" ".join([words,word])
    text=words
    print "The Query is:",text
    word_list1 = process_text(text) # remove diacritics and punctcutions, stopwords, and tokenize text

    terms=[]
    for wordx in word_list1:    
        stemAr = isri_light(wordx)
        #print "stemAr=",stemAr
        #stemAr = isri_heavy(wordx)
        #print "stemAr=",stemAr
        terms.append(wordx)             
    return terms


def query_tfidf(doc_content,doc_count):
          tf={}
          tfidf={}
          list_of_term1=[]
          for term1 in doc_content:
              tf[term1] = tf.get(term1, 0) + 1 
              list_of_term1.append(term1)
          c1=list(set(list_of_term1))

          for token, freq in tf.iteritems():
              for term1 in doc_content:
                 if token==term1:
                      df=get_df_inverted(token)
                      if df !=0:
                          idf = logn(10,float(doc_count) / df)
                          tfidf[token]= freq*idf        
          tfidf=normalize(tfidf)
          query={"tokens":c1,"tfidf":tfidf}         
          return query

def normalize(features):
    if features != {}:
        x=sqrt(sum(i**2 for i in features.itervalues()))
        if x !=0:
            norm = 1.0 / x
            for k, v in features.iteritems():
                features[k] = v* norm
    return features

#Get document frequency df from inverted index, this enters into account query terms weights 
def get_df_inverted(term):
    if term in inv:     
            inverted_list=inv[term]
    else:
            inverted_list= None            
    if inverted_list!= None:
        #inverted_list=eval(inverted_list)
        df=inverted_list[0]
    else:
        df=0
    return df

#Get document identifiers docID from inverted index, this enters into account query terms weights
def get_docs_id_inverted(term):
    if term in inv:     
            inverted_list=inv[term]
    else:
            inverted_list= None            
    if inverted_list!= None:
        #inverted_list=eval(inverted_list)
        docs_id=[]
        for i in range (0,len(inverted_list[1])):
                docs_id.append(inverted_list[1][i][0])
    else:
        docs_id=[]
    return docs_id    

#Get No. of document in the collection from VSM, this enters into account query terms weights
def get_doc_count():
    doc_count=len(d.keys())
    return doc_count

#Get a document vector from VSM for matching
def get_doc_vector(doc_id):
    vector=d[str(doc_id)]
    return vector

# Using cosine for calculate similarity
def cosin_similarity(aa,bb):
    cos = 0.0
    a_tfidf =aa
    for token, tfidf in bb.iteritems():
        if token in a_tfidf:
                cos += tfidf * a_tfidf[token]                
    return cos

#Main program..................................
def main(args):
        #request = str(raw_input('Enter no.of queries? '))
        #request =u'قسم علوم الحاسوب'       
        request =u'فرع الذكاء الاصطناعي'
        #request =u'فرع امنية الحاسوب'
        
        doc_count=get_doc_count()
        doc_content=query_vector(request)
        query=query_tfidf(doc_content,doc_count)
        a=query["tfidf"]
        query_terms=query["tokens"]
        distance=[]
        docs_id_list=set([])
        for term in query_terms:
            docs_id=get_docs_id_inverted(term)
            docs_id_list=docs_id_list|set(docs_id)
            #print docs_id_list

        for doc_id in list(docs_id_list):
               b=get_doc_vector(doc_id)
               cos=cosin_similarity(a,b)
               if cos> 0:
                   distance.append((doc_id,cos))
            
        results=distance
        results.sort(key=lambda x: x[1])
        results=[result[0] for result in results]
        results=results[-10:]
        results.reverse()
        print "Results=",results

if __name__ == '__main__':
    main(sys.argv)

