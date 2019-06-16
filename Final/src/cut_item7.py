import pickle
import pandas as pd
import numpy as np
import datetime as dt
import requests
from bs4 import BeautifulSoup
import re
import multiprocess
import ast
def cut_item7(input_data_):
    input_data, idx = input_data_
    output=input_data.copy()
    output["text"]="NA"
    else_deal=[]
    for i in range(input_data.shape[0]):
        if (input_data["QK"].iloc[i] =="K"):
            try:
                index_now=input_data.index[i]
                url="https://www.sec.gov"+input_data["url"].iloc[i]
                res = requests.get(url,timeout=1)
                soup = BeautifulSoup(res.text, 'html.parser')
                soup_txt = soup.get_text()
                soup_txt=soup_txt.replace('\xa0',' ').replace('&nbsp;',' ').replace('&#160;',' ').replace('\n',' ').replace('\x93',' ').replace('\x94',' ').replace(':',' ')
                soup_txt=soup_txt.split(" ")
                while '' in soup_txt:
                    soup_txt.remove('')
                space=" "
                soup_txt=space.join(soup_txt)
                soup_txt=soup_txt.lower()

                pattern=re.compile("item 7. manage")
                result_iter1=pattern.finditer(soup_txt)
                result_item2 = [ m1.span() for m1 in result_iter1]
                if len(result_item2) == 0:
                    pattern=re.compile("item 7 – manage")
                    result_iter1=pattern.finditer(soup_txt)
                    result_item2 = [ m1.span() for m1 in result_iter1]
                if len(result_item2) == 0:
                    pattern=re.compile("item 7.manage")
                    result_iter1=pattern.finditer(soup_txt)
                    result_item2 = [ m1.span() for m1 in result_iter1]
                if len(result_item2) == 0:
                    pattern=re.compile("item 7 - manage")
                    result_iter1=pattern.finditer(soup_txt)
                    result_item2 = [ m1.span() for m1 in result_iter1]
                if len(result_item2) == 0:
                    pattern=re.compile("item 7 — manage")
                    result_iter1=pattern.finditer(soup_txt)
                    result_item2 = [ m1.span() for m1 in result_iter1]
                if len(result_item2) == 0:
                    pattern=re.compile("item 7: manage")
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
                    pattern=re.compile("item 7. m anage")
                    result_iter1=pattern.finditer(soup_txt)
                    result_item2 = [ m1.span() for m1 in result_iter1]
                #print(len(result_item2))

                pattern=re.compile("item 7a. quant")
                result_iter1=pattern.finditer(soup_txt)
                result_item3 = [ m1.span() for m1 in result_iter1]
                if len(result_item3) == 0:
                    pattern=re.compile("item 7a – quant")
                    result_iter1=pattern.finditer(soup_txt)
                    result_item3 = [ m1.span() for m1 in result_iter1]
                if len(result_item3) == 0:
                    pattern=re.compile("item 7a.quant")
                    result_iter1=pattern.finditer(soup_txt)
                    result_item3 = [ m1.span() for m1 in result_iter1]
                if len(result_item3) == 0:
                    pattern=re.compile("item 7a - quant")
                    result_iter1=pattern.finditer(soup_txt)
                    result_item3 = [ m1.span() for m1 in result_iter1]
                if len(result_item3) == 0:
                    pattern=re.compile("item 7a — quant")
                    result_iter1=pattern.finditer(soup_txt)
                    result_item3 = [ m1.span() for m1 in result_iter1]
                #print(len(result_item3))

                pattern=re.compile("part iii")
                result_iter1=pattern.finditer(soup_txt)
                result_part2=[ m1.span() for m1 in result_iter1]
                #print(len(result_part2))

                #print(url)
                #print("i="+str(i))
                #print("item7="+str(len(result_item2)))
                #print("item7A="+str(len(result_item3)))
                #print("part3="+str(len(result_part2)))


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
        if(i%10000 == 0):
            print(i)
            multiple_fix = int(i/10000)
            output.to_csv(f'cut_data_{idx}_{multiple_fix}_10K_v2.csv')

    output.to_csv(f'cut_data_{idx}_10K_v2.csv')
    return output

if __name__ == '__main__':
    
    # Load data
    
    # datetime of each financial report
    with open('../10-K/fp_url_htm/company_dt_dic_final.pkl','rb') as f:
        company_dt_dic = pickle.load(f)

    # htm or txt url of each financial report
    with open('../10-K/fp_url_htm/company_fp_dic_final.pkl','rb') as f:
        company_fp_dic = pickle.load(f)
    
    all_company_data=pd.read_csv("../sasa/all_company_data_ver2.csv")
    
    # Split works
    
    length = int(all_company_data.shape[0]/4)
    company_data0, company_data1, company_data2, company_data3 = all_company_data.iloc[:length,:], all_company_data.iloc[length:length*2,:],\
                                                all_company_data.iloc[length*2:length*3,:],all_company_data.iloc[length*3:,:]
    works = [(company_data0,0),(company_data1,1),(company_data2,2),(company_data3,3)]
    ans = []
    pool = multiprocess.Pool(processes=4)
    ans.append(pool.map(cut_item7,works))
    output = pd.concat([ans[0][0],ans[0][1],ans[0][2],ans[0][3]],axis = 0)
    output.to_csv(f'cut_data_all_10K_v2.csv')
