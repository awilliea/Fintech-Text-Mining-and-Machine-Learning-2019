# HW2

### Data
每個tickers所對應的10-Q txt網址放在10-Q_url這個資料夾內
若要取用可以直接打以下程式

``` python
with open('./10-Q_url/company_ft_dic_final.pkl','rb') as f:
    dic = pickle.load(f)
```
擷取出來的dic個格式大概長這樣： {ticker1:10-Q txt url1, ticker2:10-Q txt url2, ...}
