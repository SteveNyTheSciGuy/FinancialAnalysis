# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 12:37:07 2020

@author: Steven
"""

import bs4 as bs
from collections import Counter
import csv
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import numpy as np
import pandas as pd
import pandas_datareader as web
import os
import requests
from sklearn import svm, neighbors
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
import tensorflow as tf
matplotlib.style.use('ggplot')

params={}
ticker="AAPL"
LIST="dowjones"  #List to iterate over, current avaiable: sp500, dowjones, nasdaq
TYPE="Adj Close" #Type of stock data to investiage: Adj Close, Open, Close, High, Low, Volume
NORM="global" #Type of normilzation to use: None, local (norm/indv stock), global (norm to entire list)
rollavg=1
start=dt.datetime(2000,1,1)
end=dt.datetime(2020,10,21)
    
params={"ticker":ticker,"LIST":LIST,"TYPE":TYPE,"NORM":NORM,"rollavg":rollavg,"start":start,"end":end}




def process_labels_1day(ticker,days=7,LIST='sp500',TYPE='Adj Close'):
    stock_df=pd.read_csv('DataFrames/{}_joined_{}.csv'.format(LIST,TYPE),index_col=0)
    tickers=stock_df.columns.values.tolist()
    
    stock_df.fillna(0,inplace=True)
    
    for i in range(1,days+1):
        stock_df['{}_{}d'.format(ticker,i)]=(stock_df[ticker].shift(-i) - stock_df[ticker])/stock_df[ticker]
    
    stock_df.fillna(0,inplace=True)
    
    return tickers, stock_df
    
def extract_features(ticker):    
    tickers, stock_df = process_labels_1day(ticker)
    stock_df['{}_target'.format(ticker)]=list(map(buy_sell_hold,stock_df['{}_1d'.format(ticker)],stock_df['{}_2d'.format(ticker)],stock_df['{}_3d'.format(ticker)],stock_df['{}_4d'.format(ticker)],stock_df['{}_5d'.format(ticker)],stock_df['{}_6d'.format(ticker)],stock_df['{}_7d'.format(ticker)]))
    vals=stock_df['{}_target'.format(ticker)].values.tolist()
    str_vals=[str(i) for i in vals]
    print('Data spread: ',Counter(str_vals))

    stock_df.fillna(0,inplace=True)
    stock_df=stock_df.replace([np.inf,-np.inf],np.nan)
    stock_df.dropna(0,inplace=True)
    
    df_vals=stock_df[[ticker for ticker in tickers]].pct_change()
    df_vals=df_vals.replace([np.inf,-np.inf],0)
    df_vals.fillna(0,inplace=True)
    
    X=df_vals.values
    y=stock_df['{}_target'.format(ticker)].values
    
    return X,y,stock_df

def implement_ml(ticker):
    X,y,df=extract_features(ticker)
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25)
    
    #clf=neighbors.KNeighborsClassifier()
    clf=VotingClassifier([('lsvc',svm.LinearSVC()),('knn',neighbors.KNeighborsClassifier()),('rfor',RandomForestClassifier())])
    
    
    clf.fit(X_train,y_train)
    confidence=clf.score(X_test,y_test)
    print('Confidence: ', confidence)
    predictions=clf.predict(X_test)
    print('Predictions: ', Counter(predictions))
    
    return confidence

def buy_sell_hold(*args):
    cols=[c for c in args]
    req_buy=0.0230
    req_sell=0.0190
    for col in cols:
        if col>req_buy:
            return 1
        if col<-req_sell:
            return -1
    return 0
 
def extract_labels(training_df, testing_df, days_before=7, days_predicted=1,params=params):
    train_X=[]
    train_Y=[]
    test_X=[]
    test_Y=[]
    
    
    
    
    
    # for i in range(training_df.shape[0]-days_before-days_predicted):
    #     train_x=training_df[i:i+days_before].values
    #     train_X.append(train_x)
    #     train_y=training_df.iloc[i+days_before+days_predicted].values
    #     train_Y.append(train_y)
    
    # for i in range(testing_df.shape[0]-days_before-days_predicted):
    #     test_x=testing_df[i:i+days_before].values
    #     test_X.append(test_x)
    #     test_y=testing_df.iloc[i+days_before+days_predicted].values
    #     test_Y.append(test_y)        
        
    return train_X, train_Y, test_X, test_Y


def deep_nn_1(train_X, train_Y, test_X, test_Y):
    
    model=tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu',input_shape=[7]),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='relu'),
        ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),loss='binary_crossentropy',metrics=['acc'])
    print(model.summary())
    
    history = model.fit(train_X, train_Y, epochs=10)
    
    
    return history



 
def split_data(test_ratio=1/20,dev_ratio=0,params=params):
    stock_df=normaile_stocks(LIST=params['LIST'],TYPE=params['TYPE'],NORM=params['NORM'],params=params)    
    size=stock_df.shape
    num_days=np.int(size[0]*test_ratio)
    
    training_df=stock_df.iloc[:-num_days][:]
    testing_df=stock_df.iloc[-num_days:][:]
   
    return training_df, testing_df

def normaile_stocks(LIST=None,TYPE=None,NORM="global",params=params):
    if LIST:
        params={"LIST":LIST}
        print("Used given LIST: {}".format(LIST))
    elif "LIST" in params:
        LIST=params["LIST"]
        print("Used LIST in params: {}".format(LIST))
        
    if TYPE:
        params={"TYPE":TYPE}
        print("Used given TYPE: {}".format(TYPE))
    elif "TYPE" in params:
        TYPE=params["TYPE"]
        print("Used TYPE in params: {}".format(TYPE))    

    filename='DataFrames/{}_joined_{}_{}.csv'.format(LIST,TYPE,NORM)
    
    stock_df_norm=[]
    if os.path.exists(filename):
        print("{} already exists".format(filename))
        stock_df_norm=pd.read_csv(filename)
    else:
        stock_df=compile_stocks(LIST,TYPE,params)
        max_val=stock_df.max()
        min_val=stock_df.min()
        
        if NORM == "global":
            max_val=stock_df.max().max()
            min_val=stock_df.min().min()

        stock_df_norm=(stock_df-min_val)/(max_val-min_val)

    return stock_df_norm
    
def corrolate_stocks(LIST=None,TYPE=None,params=params):
    
    if LIST:
        params={"LIST":LIST}
        print("Used given LIST: {}".format(LIST))
    elif "LIST" in params:
        LIST=params["LIST"]
        print("Used LIST in params: {}".format(LIST))
        
    if TYPE:
        params={"TYPE":TYPE}
        print("Used given TYPE: {}".format(TYPE))
    elif "TYPE" in params:
        TYPE=params["TYPE"]
        print("Used TYPE in params: {}".format(TYPE))    
    
    if not os.path.exists('DataFrames/{}_joined_{}.csv'.format(LIST,TYPE)):
        stock_df=compile_stocks(LIST=LIST,TYPE=TYPE)
    else:
        stock_df=pd.read_csv('DataFrames/{}_joined_{}.csv'.format(LIST,TYPE))
    
    corrolation=stock_df.corr()

    data=corrolation.values
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    heatmap=ax.pcolor(data,cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0])+0.5,minor=False)
    ax.set_yticks(np.arange(data.shape[1])+0.5,minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    column_labels=corrolation.columns
    row_labels=corrolation.index
    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    plt.show()

def compile_stocks(LIST=None,TYPE=None,params=params):

    if LIST:
        params={"LIST":LIST}
        print("Used given LIST: {}".format(LIST))
    elif "LIST" in params:
        LIST=params["LIST"]
        print("Used LIST in params: {}".format(LIST))
        
    if TYPE:
        params={"TYPE":TYPE}
        print("Used given TYPE: {}".format(TYPE))
    elif "TYPE" in params:
        TYPE=params["TYPE"]
        print("Used TYPE in params: {}".format(TYPE))    

    ticks=pd.read_csv('Lists/{}list.csv'.format(LIST),header=None,index_col=False).values.tolist()
    tickers=[val for sublist in ticks for val in sublist]    
    
    stock_df=pd.DataFrame()
    
    print('Compiling Stocks, Please Be Patient')
    for cnt,ticker in enumerate(tickers):
        try:
            data=pd.read_csv('Stocks/{}.csv'.format(ticker))
            data.set_index('Date',inplace=True)
            data.rename(columns={TYPE:ticker},inplace=True)
            stock_labels=['Adj Close','Open','High','Low','Close','Volume']
            stock_labels.remove(TYPE)
            data.drop(stock_labels,1,inplace=True)
            
            if stock_df.empty:
                stock_df=data
            else:
                stock_df=stock_df.join(data,how='outer')
                
            if cnt % 25 == 0: print(cnt)
        except:
            print('{} Stock cannot be found on Yahoo'.format(ticker))
    
    if not os.path.exists('DataFrames'):
        os.mkdir('DataFrames')
        
    print(stock_df.head())
    stock_df.to_csv('DataFrames/{}_joined_{}.csv'.format(LIST,TYPE))
    return stock_df

def plot_indv_stock(ticker,params=params):
    yaxis=params['TYPE']
    rollavg=params['rollavg']
           
    stock_info=pd.read_csv('Stocks/'+ticker+'.csv',parse_dates=True,index_col=0)
    stock_info['ma']=stock_info[yaxis].rolling(window=rollavg,min_periods=0).mean()
    stock_info.dropna(inplace=True)
    
    ax1=plt.subplot2grid((6,1),(0,0,),rowspan=5,colspan=1)
    ax2=plt.subplot2grid((6,1),(5,0,),rowspan=5,colspan=1,sharex=ax1)
    ax1.plot(stock_info.index,stock_info[yaxis],)
    ax1.plot(stock_info.index,stock_info['ma'])
    ax2.bar(stock_info.index,stock_info['Volume'])
    ax1.set_title(ticker)
    ax2.set_xlabel('Time')
    ax1.set_ylabel('Stock $')
    ax2.set_ylabel('Vol')
    
    plt.show()

def plot_indv_ohlc(ticker):
    stock_info= pd.read_csv('Stocks/'+ticker+'.csv',parse_dates=True,index_col=0)
    stock_info_ohlc=stock_info['Adj Close'].resample('10D').ohlc()
    stock_info_vol=stock_info['Volume'].resample('10D').sum()
    
    stock_info_ohlc.reset_index(inplace=True)
    print(stock_info_ohlc.head)
    
    stock_info_ohlc['Date']=stock_info_ohlc['Date'].map(matplotlib.dates.date2num)
    
    ax1=plt.subplot2grid((6,1),(0,0,),rowspan=5,colspan=1)
    ax2=plt.subplot2grid((6,1),(5,0,),rowspan=5,colspan=1,sharex=ax1)
    ax1.xaxis_date()
    
    candlestick_ohlc(ax1,stock_info_ohlc.values,width=2,colorup='g')
    ax2.fill_between(stock_info_vol.index.map(matplotlib.dates.date2num),stock_info_vol.values,0)
    ax1.set_title(ticker)
    ax2.set_xlabel('Time')
    ax1.set_ylabel('Stock $')
    ax2.set_ylabel('Vol')
    plt.show()

def get_indv_stock(ticker):
    
    stock_info=[]
    if not os.path.exists('Stocks/{}.csv'.format(ticker)):
        try:
            stock_info=web.get_data_yahoo(ticker,start=start,end=end)
            stock_info.to_csv('Stocks/{}.csv'.format(ticker))
        except:
            print('{} Stock not found on Yahoo'.format(ticker))
    else:
        print('Already had {} stock'.format(ticker))
        stock_info=pd.read_csv('Stocks/{}.csv'.format(ticker))    
    
    print(stock_info)
    return stock_info

def get_stocks(LIST=None,update=False,params=params):
    
    if LIST:
        params={"LIST":LIST}
        print("Used given LIST: {}".format(LIST))
    elif "LIST" in params:
        LIST=params["LIST"]
        print("Used LIST in params: {}".format(LIST))
    
    if update:
        tickers=get_lists(LIST)
    else:
        try:
            ticks=pd.read_csv('Lists/{}list.csv'.format(LIST),header=None,index_col=False).values.tolist()
            tickers=[val for sublist in ticks for val in sublist]
        except:
            print('List not opened')
            
    if not os.path.exists('Stocks'):
        os.mkdir("Stocks")

    for ticker in tickers:
        print(ticker)
        get_indv_stock(ticker)

def get_lists(LIST=None,params=params):
    try:
        if LIST:
            params={"LIST":LIST}
            print("Used given LIST: {}".format(LIST))
        elif "LIST" in params:
            LIST=params["LIST"]
            print("Used LIST in params: {}".format(LIST))
    except:
        LIST="sp500"
        params={"LIST":LIST}
        print("Used default sp500 LIST")
              
    tickers=[]
    if LIST == 'sp500':
        resp=requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')      
        soup=bs.BeautifulSoup(resp.text,'lxml')
        table=soup.find('table',{'class':'wikitable sortable'})
    
        for row in table.findAll('tr')[1:]:
            ticker=row.findAll('td')[0].text
            tickers.append(ticker.strip('\n').replace('.',""))
         
    elif LIST == 'dowjones':
        pass
            
    elif LIST == 'nasdaq':
        ticks=web.nasdaq_trader.get_nasdaq_symbols(retry_count=3, timeout=30, pause=None)
        tickers=ticks['NASDAQ Symbol'].values
    else:
        print(LIST,"Does not exist currently")
        return
    
        
    if os.path.exists('Lists'):
        pass
    else:
        os.mkdir('Lists')
        print("List Directory Created")
   
    with open("Lists/{}sp500list.csv".format(LIST),"w") as f:
        writer=csv.writer(f)
        writer.writerow(tickers)
    
    print(tickers)
    return tickers, params

#Code to run
if __name__ == "__main__":
    # params={}
    
    # LIST="sp500"  #List to iterate over, current avaiable: sp500, dowjones, nasdaq
    # TYPE="Adj Close" #Type of stock data to investiage: Adj Close, Open, Close, High, Low, Volume
    # NORM="global" #Type of normilzation to use: None, local (norm/indv stock), global (norm to entire list)
    # rollavg=1
    # start=dt.datetime(2000,1,1)
    # end=dt.datetime(2020,10,21)
    
    # params={"LIST":LIST,"TYPE":TYPE,"NORM":NORM,"rollavg":rollavg,"start":start,"end":end}
    #tickers=get_lists()
    #stock=get_indv_stock("mmm")
    #get_stocks(LIST='sp500')
    #plot_indv_stock('AAPL')
    #plot_indv_ohlc('AAPL')
    #stock_df=compile_stocks(LIST='dowjones',TYPE='Adj Close')
    #corrolate_stocks(LIST='dowjones',TYPE='Adj Close')
    #tickers, stock_df = process_labels_1day(ticker='UPS',days=7,LIST='sp500',TYPE='Adj Close')
    #X,y,stock_df=extract_features('AAPL')
    #confidence = implement_ml('AAPL')
    #stock_df_norm=normaile_stocks(LIST="dowjones",TYPE="Adj Close",NORM="local",params=params)
    training_df, testing_df=split_data(test_ratio=1/20,dev_ratio=0,params=params)
    train_X, train_Y, test_X, test_Y=extract_labels(training_df, testing_df, days_before=7, days_predicted=1,params=params)

    history=deep_nn_1(train_X, train_Y, test_X, test_Y)





