import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
import re
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sn
import pickle
import matplotlib.pyplot as plt
import multiprocess
import pickle

def get_CountWords(df_sen,upper_txt,dic_extra_sentiment=None):
    '''
    input:
        df_sen: Orderdictionary, from Mcdonald's SentimentWordLists
        upper_txt: string, upper txt for finiancial report 
    output:
        dic_list: list of dictionary, record each word frequency for every sentiment
        freq_list: list of ints, total frequency for each sentiment
    '''
    # count words from a document
    dic_list = []
    freq_list = []

    for emotion,df in df_sen.items():
        if(emotion != 'Documentation'):
            dic = {}
            freq = 0
            if(dic_extra_sentiment):
                sentiment_words = np.append(df.dropna().values.reshape((-1)),dic_extra_sentiment[emotion])
            else:
                sentiment_words = df.dropna().values.reshape((-1))
                
            for key in sentiment_words:
                num = len(re.findall(key,upper_txt))
                dic[key] = num
                freq += num
            dic_list.append(dic)
            freq_list.append(freq)
    return dic_list, freq_list
def get_alldoc_CountWords(input_):
    item2_df,df_sen,index,dic_extra_sentiment = input_
    company_dic_list = {}
    company_score_list = {}
    sentiment_scores_list = []
    ans = item2_df.copy()
    
    if(dic_extra_sentiment):
        filename_dic = f'company_cutwords_sentiment_dic_{index}_extra_QK.pkl'
        filename_score = f'company_cutwords_score_dic_{index}_extra_QK.pkl'
        filename_all = f'item2_{index}_extra_QK.csv'
    else:
        filename_dic = f'company_cutwords_sentiment_dic_{index}_QK.pkl'
        filename_score = f'company_cutwords_score_dic_{index}_QK.pkl'
        filename_all = f'item2_{index}_QK.csv'
        
    for i,txt in enumerate(item2_df['text'].values):
        if(i %1000 == 0):
            print(i)
        
        # to upper
        try:
            upper_txt = txt.upper()
        except:
            print(i,txt)
            assert (0 == 1)
        
        # cut words
        dic_list, freq_list = get_CountWords(df_sen,upper_txt,dic_extra_sentiment)
        sentiment_scores_list.append(str(freq_list))
        
        
        # save to dic
        company_dic_list[i] = dic_list
        company_score_list[i] = freq_list
        
    # save to dataframe
    
    
    with open(filename_dic,'wb') as f:
        pickle.dump(company_dic_list,f)
    with open(filename_score,'wb') as f:
        pickle.dump(company_score_list,f) 

    ans['score'] = np.array(sentiment_scores_list)
    ans.to_csv(filename_all)
        
    return ans

if __name__ == '__main__':
    
    # Load files 
    df_sen = pd.read_excel('LoughranMcDonald_SentimentWordLists_2018.xlsx',sheet_name = None)
    sentiment_names = ['Negative','Positive','Uncertainty','Litigious','StrongModal','WeakModal','Constraining']
    with open('dic_final_extra_sentiment.pkl','rb') as f:
        dic_extra_sentiment = pickle.load(f)
        
    item2_df = pd.read_csv('./sasa/cut_data_all_QK.csv')
    item2_df = item2_df.dropna()
    
    # split works
    length = item2_df.shape[0]
    quarter = int(length/4)

    item2_df_0, item2_df_1, item2_df_2, item2_df_3 = item2_df[:quarter], item2_df[quarter:2*quarter],\
                                                    item2_df[quarter*2:quarter*3], item2_df[quarter*3:]
    works = [(item2_df_0,df_sen,0,None),(item2_df_1,df_sen,1,None),(item2_df_2,df_sen,2,None),(item2_df_3,df_sen,3,None)]
    
    ans = []
    pool = multiprocess.Pool(processes=4)
    ans.append(pool.map(get_alldoc_CountWords,works))
    temp = pd.concat([ans[0][0],ans[0][1],ans[0][2],ans[0][3]],axis = 0)
    temp.to_csv('score_QK_extra.csv',index=False)