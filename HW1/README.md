# HW1
### Required packages
* pandas：讀入csv, excel 資料、做時間過濾及處理
* numpy：用來做資料確認（看看是否有NAN資料非常多的ETF）
* os：下載資料的路徑處理 
* time：爬蟲間隔 
* datetime：將string轉為此資料形式，使他可以做比較
* request：網路爬蟲 
* urllib.request：做檔案下載時會用到 
* calendar：要做每月月底的時間處理時會用到
* pyquery：做簡單的爬蟲以及css選擇器
* selenium：做相對較複雜的爬蟲時會用到
* pickle：將之前爬下來的網址做儲存

### Some error situations
1. 請先將Emerging Asia Pacific ETF List (114).csv放入local資料夾內，否則會出現錯誤（因為檔案位置已寫死）
2. python版本需要在3.6以上，否則f-string會無法使用（可在cmd上執行 conda update python 來更新python版本）
3. 確認自己的電腦有安裝chrome，且版本需要70以上，否則會出現錯誤（可在google 網址中打 chrome://version/  來檢查版本）
4. 需安裝chrome driver，並記住其所在位置(可在此下載 https://sites.google.com/a/chromium.org/chromedriver/downloads)
5. 網路速度較慢者，可把get_single_symbol中的wait參數調大一點，不然下載下來的數據可能會不完整或是重複下載

### Current problems
NFTY、KALL、YAO這三檔ETF在homepage以及yahoo_finance都無法取得完整的資料（只從2018開始）

### 流程圖
整體流程圖

![Imgur](https://i.imgur.com/PWKxBP1.png)


爬yahoo_finance 流程圖

![Imgur](https://i.imgur.com/SIaVFTC.png)
