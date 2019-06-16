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
from selenium import webdriver
from lxml import etree
import sys

def crawl_by_company(company_url,browser,financial_report_type):

    browser.get(company_url)
    browser.find_element_by_name('type').send_keys(financial_report_type)
    browser.find_element_by_xpath("//input[@value='Search']").click()
    time.sleep(1)
    b_url = browser.current_url
    
    finance_report_url = []
    res = requests.get(b_url)
    doc = pq(res.text)

    temp = 'https://www.sec.gov'
    for i,d in enumerate(doc('.tableFile2 #documentsbutton').items()):

        f_url = temp+d.attr('href')    
        browser.get(f_url)
        lv1_res = requests.get(f_url)
        lv1_doc = pq(lv1_res.text)
        txt_num = lv1_doc('#secNum').text().split(' ')[-1]
        browser.find_element_by_xpath(f"//a[contains(text(),'{txt_num}.txt')] ").click()
        time.sleep(1.5)
        print(browser.current_url)
        finance_report_url.append(browser.current_url)
       
    return finance_report_url

def crawl_10_Q_url(company_url,browser,financial_report_type):
    browser.get(company_url)
    browser.find_element_by_name('type').send_keys(financial_report_type)
    browser.find_element_by_xpath("//select[@id='count']").click()
    browser.find_element_by_xpath("//input[@value='Search']").click()
    time.sleep(1)
    return browser.current_url

def get_search_addr(chrome_path,tickers):
    
    files = os.listdir()
    max_ver = 0

    for f in files:
        if 'search' in f:
            version = int(f.split('_')[2].split('.')[0])
            if (version > max_ver ):
                max_ver = version
                max_f = f
    if (max_ver == 0):     
        search_addr = {}
    else:
        with open(max_f,'rb') as f:
            search_addr = pickle.load(f)
    length = len(search_addr)
    
    options = webdriver.ChromeOptions()
    prefs = {'download.default_directory': './data'}
    options.add_experimental_option('prefs', prefs)
    browser=webdriver.Chrome(chrome_path,options=options) 
    
    for i,tic in enumerate(tickers[length:]):
        browser.get('https://www.sec.gov/edgar/searchedgar/legacy/companysearch.html')
        browser.find_element_by_xpath("//input[@title='CIK']").send_keys(tic)
        browser.find_element_by_xpath("//input[@value='Find Companies']  ").click()
        c_url = browser.current_url
#         time.sleep(1)
        
        res = requests.get(c_url)
        doc = pq(res.text)
        message = doc('center').text()
        if (message == 'No matching Ticker Symbol.'):
#             print('Unfound')
            search_addr[tic] = 'Unfound'
        else:
            search_addr[tic] = c_url
            
        if(len(search_addr)%1000 == 0):
            num = len(search_addr)/1000
            
            with open(f"search_addr_{num}.pkl","wb") as f:
                pickle.dump(search_addr,f)
                print(f'Save search_addr_{num}.pkl',i+length)
    browser.quit()
    
    with open("search_addr_final.pkl","wb") as f:
        pickle.dump(search_addr,f)
        print('Save search_addr_final.pkl',i+length)
                
    return search_addr

def get_all_ft_url(tickers,search_addr,chrome_filepath,financial_report_type):
    
    # check directory
    if not os.path.isdir(f'./{financial_report_type}_url_'):
        os.mkdir(f'./{financial_report_type}_url_')
    
    # check files
    files = os.listdir(f'./{financial_report_type}_url_')
    max_ver = 0

    for f in files:
        if 'company' in f:
            version = int(f.split('_')[3].split('.')[0])
            if (version > max_ver ):
                max_ver = version
                max_f = f
    if (max_ver == 0):     
        company_ft_dic = {}
    else:
        # Check file size > 0
        if (os.path.getsize(f"./{financial_report_type}_url_/{max_f}") > 0):
            with open(f'./{financial_report_type}_url_/{max_f}','rb') as f:
                company_ft_dic = pickle.load(f)
        else:
            company_ft_dic = {}
    length = len(company_ft_dic)
    
    options = webdriver.ChromeOptions()
    prefs = {'download.default_directory': f'./{financial_report_type}_url_'}
    options.add_experimental_option('prefs', prefs)
    browser=webdriver.Chrome(chrome_filepath,options=options) 
    
    for i,key in enumerate(tickers[length:]):
        addr = search_addr[key]
#         print(key)
        
        # Extra check for updated tickers
        if(addr != 'Unfound'):
            res = requests.get(addr)
            doc = pq(res.text)
            message = doc('center').text()
            if (message == 'No matching Ticker Symbol.'):
                addr = 'Unfound'
        
        # main function for crawl
        if (addr != 'Unfound'):
            ft_addr = crawl_10_Q_url(addr,browser,financial_report_type)
            company_ft_dic[key] = ft_addr
        else:
            company_ft_dic[key] = 'Unfounded'
            
        if(len(company_ft_dic)%1000 == 0):
            num = len(company_ft_dic)/1000
            
            with open(f"./{financial_report_type}_url_/company_ft_dic_{num}.pkl","wb") as f:
                pickle.dump(company_ft_dic,f)
                print(f'./{financial_report_type}_url_/company_ft_dic_{num}.pkl',i+length)
    with open(f"./{financial_report_type}_url_/company_ft_dic_final.pkl","wb") as f:
        pickle.dump(company_ft_dic,f)
        print(f'./{financial_report_type}_url_/company_ft_dic_final.pkl',i+length)

if __name__ == '__main__':
    
    chrome_driver_path = sys.argv[1]
    
    tickers = pd.read_csv('all_ticker.csv').iloc[:,1].values
    seach_addr = get_search_addr(chrome_driver_path,tickers)
    get_all_ft_url(tickers,search_addr,chrome_driver_path,'10-K')
    get_all_ft_url(tickers,search_addr,chrome_driver_path,'10-Q')
    