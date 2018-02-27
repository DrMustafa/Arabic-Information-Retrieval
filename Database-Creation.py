#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys,re,os,string
from math import sqrt
from numpy.lib.scimath import logn
from time import clock
import nltk
import json
from nltk.stem.isri import ISRIStemmer
'''--------------------------------------------------------------'''
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
	word = arstemmer.norm(word, num=1)        
	if not word in arstemmer.stop_words:    
	    word = arstemmer.pre32(word)        
	    word = arstemmer.suf32(word)        
	    word = arstemmer.waw(word)          
	    word = arstemmer.norm(word, num=2)  
	lstem=word
	return lstem

def parseCollection(collFile):
    ''' returns the id, title and text of the next page in the collection '''
    doc_file=[]
    doc=[]
    for line in collFile:     
        if line!='</DOC>\n':
            doc.append(line)
        else:
            curPage=''.join(doc)
            page_seq=re.search('<DOC_ORDER>(.*?)</DOC_ORDER>', curPage, re.DOTALL)
            pagetitle=re.search('<HEADLINE>(.*?)</HEADLINE>', curPage, re.DOTALL)
            pagetext=re.search('<TEXT>(.*?)</TEXT>', curPage, re.DOTALL)
            if page_seq ==None or pagetitle ==None or pagetext==None : return {}
            d={}
            d['DOCSEQ']=page_seq.group(1)
            d['TITLE']=pagetitle.group(1)
            d['TEXT']=pagetext.group(1)
            doc=[]
            doc_file.append(d)
    return doc_file

def getFileNames(directory) :
  l = os.listdir(directory)
  l2 = []
  for name in l :
    if name[-4:] != ".dtd" and name != "README":
      if os.path.isdir(directory + "/" + name):
        print "%s is dir" % name
        l2.extend(getFileNames(directory + "/" + name + "/"))
      else :
        l2.append( directory + "/" + name)
  return l2    

def gen_documents(docFile):
    doc= open(docFile,'r')
    pagedict={}
    pagedict_list=parseCollection(doc)
    return pagedict_list
  
def documents_index():
    
    vector_space=[]
    tokens = {}
    docFileNames  = getFileNames("f:/ir/data")
    docs_dict={}
    get_documents=[]
    docFileName=docFileNames[0]
    docFile_list=gen_documents(docFileName)
    for pagedict in docFile_list:
        lines='\n'.join((pagedict['TITLE'],pagedict['TEXT']))
        pageseq=int(pagedict['DOCSEQ'])
        text=lines.decode('utf-8')              
        text=text.split()
        word_list=[]
        for i in text:
            if not i in arabicStopWords:
                word_list.append(i)
        text=' '.join(word_list)
        word_list1 = process_text(text) # remove diacritics and punctcutions, stopwords, and tokenize text
       
        terms=[]
        for wordx in word_list1:            
            stemAr = isri_light(wordx)
            #print "stemAr=",stemAr
            #stemAr = isri_heavy(wordx)
            #print "stemAr=",stemAr
            terms.append(wordx)           
        docs_dict[pageseq]=terms
        
    for doc_id, doc_content in docs_dict.items():
      tf={}
      list_of_term1=[]
      for term1 in doc_content:
          tf[term1] = tf.get(term1, 0) + 1
          list_of_term1.append(term1)
      c1=list(set(list_of_term1))
      for token, freq in tf.iteritems():
          tokens.setdefault(token, []).append((doc_id,freq))
      document={"doc_id":doc_id,"tokens":c1,"tfidf":{}}
      vector_space.append(document)  
    doc_count=len(vector_space)
    dinv={}
    for token, docs in tokens.iteritems():
        dinv[token]=[len(docs),docs]    
        l_docs=len(docs)
        idf = logn(10,float(doc_count) / float(l_docs))
        for doc_id, tf in docs:
            tfidf=tf*idf
            for i in vector_space:
                if (i['doc_id']==doc_id) and (token in i['tokens']):
                    i['tfidf'][token]=tfidf
    with open('f:/ir/inverted_index.json', 'w') as f:
        json.dump(dinv, f)
    f.close()
    '''with open('f:/inverted_index.json') as f:
        inv= json.load(f)
    f.close()'''

    for doc in vector_space:
        doc["tfidf"] = normalize(doc["tfidf"])
    return vector_space

def normalize(features):
    if features != {}:
        x=sqrt(sum(i**2 for i in features.itervalues()))
        if x !=0:
            norm = 1.0 / sqrt(sum(i**2 for i in features.itervalues()))
            for k, v in features.iteritems():
                features[k] = v * norm
    return features
        
'''Main program ...............................'''
def main(args):
    documents=documents_index()
    docs={}
    for document in documents:
        docs[document["doc_id"]]=document["tfidf"]
    with open('f:/ir/docs.json','w') as f:
        json.dump(docs,f)
    f.close()
    print "The inverted index is created......"
    

if __name__ == '__main__':
    main(sys.argv)

