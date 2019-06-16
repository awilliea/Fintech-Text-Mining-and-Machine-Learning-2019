import numpy as np
import pandas as pd
from collections import Counter
import re
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sn
import pickle
import matplotlib.pyplot as plt
import jieba
from gensim.models import Word2Vec
import multiprocess
import pickle
import ast

from nltk.corpus import stopwords
import nltk

def add_key_words_to_jieba(key_words):
    for word in key_words:
        try:
            word = word.lower()
            jieba.add_word(word)
        except:
            print(word)
def process_sentence(sentence,stopWords,punct):
    cut_sentence = ' '.join(jieba.cut(sentence))
    split_words = cut_sentence.split(' ')
    
    # Remove None
    rmNone_words = list(filter(None, split_words))
    
    # Remove stopwords
    rmStopWords_words = list(filter(lambda a: a not in stopWords and a != '\n', rmNone_words))
    
    # Remove punctuation
    filterpuntl = lambda l: list(filter(lambda x: x not in punct, l))
    rmPunc_words = filterpuntl(rmStopWords_words)
    
    return rmPunc_words
def get_all_cut_sentence(input_):
    item2_df_text,stopWords,punct,ver = input_
#     cut_sentences = []
    with open(f'cut_sentences_{ver}_15000.pkl','rb') as f:
        cut_sentences  = pickle.load(f)
    for i,text in enumerate(item2_df_text):
        if(i % 5000 == 0):
            print(i)
        if(i <= 15000):
            continue
        else:
            cut_sentences.append(process_sentence(text,stopWords,punct))
            if(i % 5000 == 0):
                with open(f'cut_sentences_{ver}_{i}.pkl','wb') as f:
                    pickle.dump(cut_sentences,f)
    with open(f'cut_sentences_{ver}_{i}.pkl','wb') as f:
        pickle.dump(cut_sentences,f)
    return cut_sentences

def filter_scoresum(df,threshold):
    scoresum_list = []
    df_score = df['score'].values
    for score in df_score:
        scoresum_list.append(sum(ast.literal_eval(score)))
    df['scoresum'] = scoresum_list
    return df[df['scoresum'] > threshold].copy()
def get_extradic(df_sen,sentiment_names,model):
    dic_sentiment_extra = []
    
    # each sentiment
    for sentiment in sentiment_names:
        sentiment_words = [i.lower() for i in df_sen[sentiment].dropna().values.reshape((-1))]
        extra_words = {}
        
        # each sentiment words
        for word in sentiment_words:
            # there is the word in word2vec model
            try:
                similar_words = list(filter(lambda x: x not in sentiment_words,[tu[0] for tu in model.wv.most_similar(word, topn=10)]))
                
                for s_word in similar_words:
                    # each silimiar words
                    if s_word in extra_words:
                        extra_words[s_word] += 1
                    else:
                        extra_words[s_word] = 1
            # not contain the word
            except:
                print('Not contain',word)
        dic_sentiment_extra.append(extra_words)
    return dic_sentiment_extra
def extract_importance_sentimentwords(dic_sentiment_extra,sentiment_names,threshold = 10):
    dic_sentiment = {}
    for i,dic in enumerate(dic_sentiment_extra):
        dic_sentiment[sentiment_names[i]] = []
        temp = sorted(dic.items(),key=lambda kv: kv[1],reverse=True)
        filter_words = list(filter(lambda x:x[1]>=threshold,temp))
        for word in filter_words:
            dic_sentiment[sentiment_names[i]].append(word[0].upper())
        dic_sentiment[sentiment_names[i]] = np.array(dic_sentiment[sentiment_names[i]])
    return dic_sentiment

if __name__=='__main__':
    nltk.download("stopwords")
    stopWords = stopwords.words('english')
    df = pd.read_excel('LoughranMcDonald_MasterDictionary_2018.xlsx')
    score = pd.read_csv('./score_QK.csv')
    key_words = df['Word'].values
    add_key_words_to_jieba(key_words)
    punct = set(u''':!),$=.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')
    filter_score = filter_scoresum(score,100)
    length = filter_score.shape[0]
    all_text = filter_score['text'].values
    
    # split works 
    quarter = int(length/4)

    text_0, text_1, text_2, text_3 = all_text[:quarter], all_text[quarter:2*quarter],\
                                                    all_text[quarter*2:quarter*3], all_text[quarter*3:]
    works = [(text_0,stopWords,punct,1),(text_1,stopWords,punct,2),(text_2,stopWords,punct,3),(text_3,stopWords,punct,4)]
    ans = []
    pool = multiprocess.Pool(processes=4)
    ans.append(pool.map(get_all_cut_sentence,works))
    all_cut_sentences = ans[0][0]+ans[0][1]+ans[0][2]+ans[0][3]
    with open('all_cut_sentences.pkl','wb') as f:
        pickle.dump(all_cut_sentences,f)
        
    # Word vectors    
    model = Word2Vec(all_cut_sentences,iter=20,size=300,sg=1) # skip gram, vector size=250
    model.save('word2vec.model')
    embedding_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
    word2idx = {}
    vocab_list = [(word, model.wv[word]) for word, _ in model.wv.vocab.items()]
    for i, vocab in enumerate(vocab_list):
        word, vec = vocab
        embedding_matrix[i + 1] = vec
        word2idx[word] = i + 1
    
    # count words for ezch sentiment    
    df_sen = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx',sheet_name = None)
    sentiment_names = ['Negative','Positive','Uncertainty','Litigious','StrongModal','WeakModal','Constraining']
    dic_sentiment_extra = get_extradic(df_sen,sentiment_names,model)
    dic_final_sentiment = extract_importance_sentimentwords(dic_sentiment_extra,sentiment_names)
    with open('dic_final_extra_sentiment.pkl','wb') as f:
        pickle.dump(dic_final_sentiment,f)