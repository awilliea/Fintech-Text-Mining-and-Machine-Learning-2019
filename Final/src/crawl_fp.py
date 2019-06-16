import numpy as np
import pandas as pd
from pyquery import PyQuery as pq
import time
import datetime
import requests
import urllib.request
import os
import calendar
import pickle
from lxml import etree
import sys

def crawl_fp_txt_url(company_url,fp_type = '10-Q'):
    finance_report_url = []
    finance_datetime = []
    mark = []
    res = requests.get(company_url)
    doc = pq(res.text)
    df_list = pd.read_html(company_url)
    
    skip_index = []
    # There are tables in the url
    if(len(df_list) >= 3):
        df = df_list[2]
        df_time = df.iloc[:,3].values
        for i,ty in enumerate(df.iloc[:,0].values):
            if(ty == fp_type):
                finance_datetime.append(datetime.datetime.strptime(df_time[i], '%Y-%m-%d'))
            else:
                skip_index.append(i)
    
    temp = 'https://www.sec.gov'

    for i,d in enumerate(doc('.tableFile2 #documentsbutton').items()):
        
        if(not i in skip_index):
            f_url = temp+d.attr('href')    
            lv1_res = requests.get(f_url)
            lv1_doc = pq(lv1_res.text)
            txt_num = lv1_doc('#secNum').text().split(' ')[-1]
            htm_pd = pd.read_html(f_url)
            htm_num_ = htm_pd[0].iloc[0,2]

            sel = etree.HTML(lv1_res.text)
            
            try:
                if (type(htm_num_) == str) :
                    htm_num = htm_num_.split(' ')[0]
                    url = sel.xpath(f"//a[contains(text(),'{htm_num}')] ")[0].attrib['href']
                    mark.append(0)
                else:
                    url = sel.xpath(f"//a[contains(text(),'{txt_num}.txt')] ")[0].attrib['href']
                    mark.append(1)
            except:
                print('Company',company_url)
                print(f_url)
                print(htm_num)
                print(df_time[i])
                print(txt_num)
            finance_report_url.append(url)
        
    if (len(finance_report_url) == 0):
        finance_report_url = 'Unfounded'
        
    return finance_report_url, finance_datetime, mark
def get_all_fp_url_htm(tickers,company_addr,fp_type = '10-Q'):
    # check directory
    if not os.path.isdir(f'./{fp_type}/fp_url_htm'):
        os.mkdir(f'./{fp_type}/fp_url_htm')
    
    # check files
    files = os.listdir(f'./{fp_type}/fp_url_htm')
    max_ver = 0

    for f in files:
        if 'company_fp' in f:
            version = int(f.split('_')[3].split('.')[0])
            if (version > max_ver ):
                max_ver = version
                max_fp_f = f
                max_dt_f = f.replace('fp','dt')
                max_mk_f = f.replace('fp','mk')
    if (max_ver == 0):     
        company_fp_dic = {}
        company_dt_dic = {}
        company_mk_dic = {}
    else:
        if (os.path.getsize(f"./{fp_type}/fp_url_htm/{max_fp_f}") > 0):
            with open(f'./{fp_type}/fp_url_htm/{max_fp_f}','rb') as f:
                company_fp_dic = pickle.load(f)
        else:
            company_fp_dic = {}
            
        if (os.path.getsize(f"./{fp_type}/fp_url_htm/{max_dt_f}") > 0):
            with open(f'./{fp_type}/fp_url_htm/{max_dt_f}','rb') as f:
                company_dt_dic = pickle.load(f)
        else:
            company_dt_dic = {} 
        
        if (os.path.getsize(f"./{fp_type}/fp_url_htm/{max_mk_f}") > 0):
            with open(f'./{fp_type}/fp_url_htm/{max_mk_f}','rb') as f:
                company_mk_dic = pickle.load(f)
        else:
            company_mk_dic = {} 
            
    length = len(company_fp_dic)   
    
    for i,key in enumerate(tickers[length:]):
        addr = company_addr[key]
        
        if (addr != 'Unfounded'):
            fp_addr, fp_datetime, fp_mark = crawl_fp_txt_url(addr,fp_type)
            
            company_fp_dic[key] = fp_addr
            company_dt_dic[key] = fp_datetime
            company_mk_dic[key] = fp_mark
        else:
            company_fp_dic[key] = 'Unfounded'
            company_dt_dic[key] = 'Unfounded'
            company_mk_dic[key] = 'Unfounded'
            
        if(len(company_fp_dic)%1000 == 0):
            num = len(company_fp_dic)/1000

            with open(f"./{fp_type}/fp_url_htm/company_fp_dic_{num}.pkl","wb") as f:
                pickle.dump(company_fp_dic,f)
                print(f'./{fp_type}/fp_url_htm/company_fp_dic_{num}.pkl',i+length)
                
            with open(f"./{fp_type}/fp_url_htm/company_dt_dic_{num}.pkl","wb") as f:
                pickle.dump(company_dt_dic,f)
                print(f'./{fp_type}/fp_url_htm/company_dt_dic_{num}.pkl',i+length)
            
            with open(f"./{fp_type}/fp_url_htm/company_mk_dic_{num}.pkl","wb") as f:
                pickle.dump(company_mk_dic,f)
                
    with open(f"./{fp_type}/fp_url_htm/company_fp_dic_final.pkl","wb") as f:
        pickle.dump(company_fp_dic,f)
        print(f'./{fp_type}/fp_url_htm/company_fp_dic_final.pkl',i+length)
        
    with open(f"./{fp_type}/fp_url_htm/company_dt_dic_final.pkl","wb") as f:
        pickle.dump(company_dt_dic,f)
        print(f'./{fp_type}/fp_url_htm/company_dt_dic_final.pkl',i+length)
        
    with open(f"./{fp_type}/fp_url_htm/company_mk_dic_final.pkl","wb") as f:
        pickle.dump(company_mk_dic,f)

def remove_repeat_ones(company_dt_dic,company_fp_dic,company_mk_dic,tickers):
    for tk in tickers:
        if(company_dt_dic[tk] != 'Unfounded'):
#             print(tk)
            dts = np.array(company_dt_dic[tk])
            fps = np.array(company_fp_dic[tk])
            mks = np.array(company_mk_dic[tk])
        
            length = dts.shape[0]
            repeat_indexs = []
            temp = dts.copy()

            for i in range(length-1):
                if(temp[i] == temp[i+1]):
                    repeat_indexs.append(i+1)

            company_dt_dic[tk] = np.delete(dts,repeat_indexs).tolist()
            company_fp_dic[tk] = np.delete(fps,repeat_indexs).tolist()
            company_mk_dic[tk] = np.delete(mks,repeat_indexs).tolist()
            
def check_repeat(company_dt_dic,tickers):
    for tk in tickers:
        if(company_dt_dic[tk] != 'Unfounded'):
            dts = np.array(company_dt_dic[tk])
            fps = np.array(company_fp_dic[tk])
            mks = np.array(company_mk_dic[tk])
        
            length = dts.shape[0]
            repeat_indexs = []
            temp = dts.copy()

            for i in range(length-1):
                if(temp[i] == temp[i+1]):
                    repeat_indexs.append(i+1)
            assert repeat_indexs == []
    print(True)
    
def check_index(company_fp_dic,company_dt_dic,tickers):
    '''
    Check alignment for url and datetime.
    '''
    errors = []
    for tk in tickers:
        dt_ele = company_dt_dic[tk]
        fp_ele = company_fp_dic[tk]
        
        if(company_dt_dic[tk] == 'Unfounded'):
            try:
                assert fp_ele == dt_ele
            except:
                print(tk)
                errors.append(tk)
        else:
            try:
                assert len(fp_ele) == len(dt_ele)
            except:
                print(tk)
                errors.append(tk)
    assert errors == []

if __name__ == '__main__':
    fp_type = sys.argv[1]
    
    # Load data
    with open('search_addr_final.pkl','rb') as f:
        search_addr = pickle.load(f)
    tickers = pd.read_csv('all_ticker.csv').iloc[:,1].values
    
    # Crawl
    get_all_fp_url_htm(tickers,company_ft_dic,fp_type)
    
    # process the data
    
    # datetime of each financial report
    with open(f'./{fp_type}/fp_url_htm/company_dt_dic_final.pkl','rb') as f:
        company_dt_dic = pickle.load(f)

    # htm or txt url of each financial report
    with open(f'./{fp_type}/fp_url_htm/company_fp_dic_final.pkl','rb') as f:
        company_fp_dic = pickle.load(f)

    # mark of each financial report
    # 0 : crawl from raw 1(htm or txt)  1: crawl from original txt
    with open(f'./{fp_type}/fp_url_htm/company_mk_dic_final.pkl','rb') as f:
        company_mk_dic = pickle.load(f)
    
    for key,value in company_dt_dic.items():
    if value == []:
        company_dt_dic[key] = 'Unfounded'
        
    with open(f'./{fp_type}/fp_url_htm/company_dt_dic_final.pkl','wb') as f:
        pickle.dump(company_dt_dic,f)
    
    # some test     
    remove_repeat_ones(company_dt_dic,company_fp_dic,company_mk_dic,tickers)
    check_index(company_fp_dic,company_dt_dic,tickers)
    check_repeat(company_dt_dic,tickers)
    
    # save the final results
    with open(f"./{fp_type}/fp_url_htm/company_fp_dic_final.pkl","wb") as f:
    pickle.dump(company_fp_dic,f)
        
    with open(f"./{fp_type}/fp_url_htm/company_dt_dic_final.pkl","wb") as f:
        pickle.dump(company_dt_dic,f)

    with open(f"./{fp_type}/fp_url_htm/company_mk_dic_final.pkl","wb") as f:
        pickle.dump(company_mk_dic,f)