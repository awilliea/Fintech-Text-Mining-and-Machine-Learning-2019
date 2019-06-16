import pickle
import pandas as pd
import numpy as np
import datetime as dt
import requests
from bs4 import BeautifulSoup
import re
import multiprocess
import ast

def filter_scoresum(df,threshold):
    scoresum_list = []
    df_score = df['score'].values
    for score in df_score:
        scoresum_list.append(sum(ast.literal_eval(score)))
    df['scoresum'] = scoresum_list
    return df[df['scoresum'] < threshold].iloc[:,0]
def get_CrawlIndex(cutDataAndNeedtoCrawl):
    QK = cutDataAndNeedtoCrawl['QK'].values
    texts = cutDataAndNeedtoCrawl['text'].values
    index = cutDataAndNeedtoCrawl.index
    crawl_index = []
    
    for i,types in enumerate(QK):
        if(types == 'Q'):
            if(type(texts[i]) != str):
                crawl_index.append(1)
            else:
                crawl_index.append(0)
        else:
            crawl_index.append(0)
    return crawl_index
def cut_item2(input_data_):
    input_data, idx, need_recrawl_index = input_data_
    output=input_data.copy()
    output["text"]="NA"
    else_deal=[]
    for i in range(input_data.shape[0]):
        if(need_recrawl_index[i] == 1):
            try:
                index_now=input_data.index[i]
                url="https://www.sec.gov"+input_data["url"].iloc[i]
                res = requests.get(url)
                soup = BeautifulSoup(res.text, 'html.parser')
                soup_txt = soup.get_text()
                soup_txt=soup_txt.replace('\xa0',' ').replace('&nbsp;',' ').replace('&#160;',' ').replace('\n',' ').replace('\x93',' ').replace('\x94',' ').replace(':',' ')
                soup_txt=soup_txt.split(" ")
                while '' in soup_txt:
                    soup_txt.remove('')
                space=" "
                soup_txt=space.join(soup_txt)
                soup_txt=soup_txt.lower()

                pattern=re.compile("item 2. manage")
                result_iter1=pattern.finditer(soup_txt)
                result_item2 = [ m1.span() for m1 in result_iter1]
                if len(result_item2) == 0:
                    pattern=re.compile("item 2 – manage")
                    result_iter1=pattern.finditer(soup_txt)
                    result_item2 = [ m1.span() for m1 in result_iter1]
                if len(result_item2) == 0:
                    pattern=re.compile("item 2.manage")
                    result_iter1=pattern.finditer(soup_txt)
                    result_item2 = [ m1.span() for m1 in result_iter1]
                if len(result_item2) == 0:
                    pattern=re.compile("item 2 - manage")
                    result_iter1=pattern.finditer(soup_txt)
                    result_item2 = [ m1.span() for m1 in result_iter1]
                if len(result_item2) == 0:
                    pattern=re.compile("item 2 — manage")
                    result_iter1=pattern.finditer(soup_txt)
                    result_item2 = [ m1.span() for m1 in result_iter1]
                if len(result_item2) == 0:
                    pattern=re.compile("item 2: manage")
                    result_iter1=pattern.finditer(soup_txt)
                    result_item2 = [ m1.span() for m1 in result_iter1]
                if len(result_item2) == 0:
                    pattern=re.compile("management's discussion and analysis")
                    result_iter1=pattern.finditer(soup_txt)
                    result_item2 = [ m1.span() for m1 in result_iter1]
                if len(result_item2) == 0:
                    pattern=re.compile("management’s discussion and analysis")
                    result_iter1=pattern.finditer(soup_txt)
                    result_item2 = [ m1.span() for m1 in result_iter1]
                if len(result_item2) == 0:
                    pattern=re.compile("managements' discussion and analysis")
                    result_iter1=pattern.finditer(soup_txt)
                    result_item2 = [ m1.span() for m1 in result_iter1]
                if len(result_item2) == 0:
                    pattern=re.compile("management's discussion of operations and financial condition")
                    result_iter1=pattern.finditer(soup_txt)
                    result_item2 = [ m1.span() for m1 in result_iter1]
                if len(result_item2) == 0:
                    pattern=re.compile("item 2. m anage")
                    result_iter1=pattern.finditer(soup_txt)
                    result_item2 = [ m1.span() for m1 in result_iter1]
                #print(len(result_item2))

                pattern=re.compile("item 3. quant")
                result_iter1=pattern.finditer(soup_txt)
                result_item3 = [ m1.span() for m1 in result_iter1]
                if len(result_item3) == 0:
                    pattern=re.compile("item 3 – quant")
                    result_iter1=pattern.finditer(soup_txt)
                    result_item3 = [ m1.span() for m1 in result_iter1]
                if len(result_item3) == 0:
                    pattern=re.compile("item 3.quant")
                    result_iter1=pattern.finditer(soup_txt)
                    result_item3 = [ m1.span() for m1 in result_iter1]
                if len(result_item3) == 0:
                    pattern=re.compile("item 3 - quant")
                    result_iter1=pattern.finditer(soup_txt)
                    result_item3 = [ m1.span() for m1 in result_iter1]
                if len(result_item3) == 0:
                    pattern=re.compile("item 3 — quant")
                    result_iter1=pattern.finditer(soup_txt)
                    result_item3 = [ m1.span() for m1 in result_iter1]
                #print(len(result_item3))

                pattern=re.compile("part ii")
                result_iter1=pattern.finditer(soup_txt)
                result_part2=[ m1.span() for m1 in result_iter1]
                #print(len(result_part2))

                if len(result_item2) == 1:
                    if len(result_item3) == 1:
                        output["text"].loc[index_now]=soup_txt[result_item2[0][0]:result_item3[0][0]]
                    elif len(result_item3) == 2:
                        output["text"].loc[index_now]=soup_txt[result_item2[0][0]:result_item3[1][0]]
                    else:    
                        if len(result_part2) != 0:
                            output["text"].loc[index_now]=soup_txt[result_item2[0][0]:result_part2[-1][0]]
                        else:
                            output["text"].loc[index_now]=soup_txt[result_item2[0][0]:]


                elif len(result_item2) == 2:
                    if len(result_item3) == 2:
                        #(2,2)
                        output["text"].loc[index_now]=soup_txt[result_item2[1][0]:result_item3[1][0]]
                    elif len(result_item3) == 1:
                        #(2,1)
                        output["text"].loc[index_now]=soup_txt[result_item2[1][0]:result_item3[0][0]]
                    else:
                        if len(result_part2) >= 2:
                            output["text"].loc[index_now]=soup_txt[result_item2[1][0]:result_part2[1][0]]
                        elif len(result_part2) == 1:
                            output["text"].loc[index_now]=soup_txt[result_item2[1][0]:result_part2[0][0]]
                        else:
                            output["text"].loc[index_now]=soup_txt[result_item2[1][0]:]


                elif len(result_item2) == 3:
                    if len(result_item3) == 1:
                        output["text"].loc[index_now]=soup_txt[result_item2[-1][0]:result_item3[0][0]]
                    elif len(result_item3) >= 2:
                        output["text"].loc[index_now]=soup_txt[result_item2[-1][0]:result_item3[-1][0]]
                    else:  
                        if len(result_part2) != 0:
                            output["text"].loc[index_now]=soup_txt[result_item2[-1][0]:result_part2[-1][0]]
                        else:
                            output["text"].loc[index_now]=soup_txt[result_item2[-1][0]:]

                elif len(result_item2) > 3:
                    if len(result_item3) == 1:
                        output["text"].loc[index_now]=soup_txt[result_item2[1][0]:result_item3[0][0]]
                    elif len(result_item3) >= 2:
                        output["text"].loc[index_now]=soup_txt[result_item2[1][0]:result_item3[1][0]]
                    else:  
                        if len(result_part2) != 0:
                            output["text"].loc[index_now]=soup_txt[result_item2[1][0]:result_part2[-1][0]]
                        else:
                            output["text"].loc[index_now]=soup_txt[result_item2[1][0]:]               
                else:
                    print('*****************wronggggg*****************')
                    print(url)
                    print(i)
                    print("result_item2 = "+str(len(result_item2)))
                    print("result_item3 = "+str(len(result_item3)))
                    print("result_part2 = "+str(len(result_part2)))
                    else_deal.append(i)

            except:
                print("----------------unsolved--------------")
                print(url)
                print(i)

        if(i % 5000 == 0 and i != 0):
            print(i)
            multiple_fix = int(i/10000)
            output.to_csv(f'cut_data_{idx}_{multiple_fix}_10Q_9.csv')
            
    output.to_csv(f'cut_data_{idx}_10Q_9.csv')
    
    return output

if __name__ =='__main__':
    all_company_data=pd.read_csv("../sasa/all_company_data_ver2.csv")
    cutDataAndNeedtoCrawl = pd.read_csv('../sasa/cutDataAndNeedtoCrawl.csv')
    need_recrawl_index = get_CrawlIndex(cutDataAndNeedtoCrawl)
    need_recrawl_index = np.array(need_recrawl_index)
    
    # split data to 8 chunks
    length = int(need_recrawl_index.shape[0]/8)
    need_recrawl_index0, need_recrawl_index1,need_recrawl_index2,need_recrawl_index3,\
    need_recrawl_index4,need_recrawl_index5,need_recrawl_index6,need_recrawl_index7 \
    = need_recrawl_index[:length], need_recrawl_index[length:length*2], need_recrawl_index[length*2:length*3],\
    need_recrawl_index[length*3:length*4], need_recrawl_index[length*4:length*5], need_recrawl_index[length*5:length*6],\
    need_recrawl_index[length*6:length*7] , need_recrawl_index[length*7:] 
    
    length = int(all_company_data.shape[0]/8)
    company_data0, company_data1, company_data2, company_data3,company_data4,company_data5,company_data6,company_data7 = \
    all_company_data.iloc[:length,:], all_company_data.iloc[length:length*2,:],all_company_data.iloc[length*2:length*3,:]\
    ,all_company_data.iloc[length*3:length*4,:],all_company_data.iloc[length*4:length*5,:],all_company_data.iloc[length*5:length*6,:],all_company_data.iloc[length*6:length*7,:]\
    ,all_company_data.iloc[length*7:,:]

    works = [(company_data0,0,need_recrawl_index0),(company_data1,1,need_recrawl_index1),(company_data2,2,need_recrawl_index2),\
             (company_data3,3,need_recrawl_index3),(company_data4,4,need_recrawl_index4),(company_data5,5,need_recrawl_index5)\
            ,(company_data6,6,need_recrawl_index6),(company_data7,7,need_recrawl_index7)]

    ans = []
    pool = multiprocess.Pool(processes=8)
    ans.append(pool.map(cut_item2,works))
    output = pd.concat([ans[0][0],ans[0][1],ans[0][2],ans[0][3],ans[0][4],ans[0][5],ans[0][6],ans[0][7]],axis = 0)
    output.to_csv('cut_data_all_10Q_9.csv')
    
