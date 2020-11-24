import numpy as np
import pandas as pd

from tqdm import tqdm
import os
import time

from collections import defaultdict
from sklearn.model_selection import train_test_split

company_df = pd.read_csv('LimitedComp/dis_comp.csv')
company_list = list(company_df['Company'])
consider_years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']

all_stock_path = './Stocks'
news_data = './news_day.csv'

threshold = 0.015
UP, STAY, DOWN = 2, 1, 0
test_size = 0.2

def get_news_data(news_data, consider_years, company_list):
    res = defaultdict(dict)
    news_df = pd.read_csv(news_data)

    stocks = list(news_df['stock'])
    date = list(news_df['day'])
    title = list(news_df['title'])

    comp_set = set(company_list)
    years_set = set(consider_years)
    for i in tqdm(range(len(stocks))):
        if stocks[i] in comp_set and date[i][:4] in years_set:
            if date[i] not in res[stocks[i]]:
                res[stocks[i]][date[i]] = [title[i]]
            else:
                res[stocks[i]][date[i]].append(title[i])
    return res

def build_path(company, stock_path=all_stock_path):
    return stock_path + '/' + company.lower() + '.us.txt'

def get_label(rate, thres):
    if rate>=thres:
        return UP
    elif rate<=-thres:
        return DOWN
    else:
        return STAY

def build_dataset(company_list, news_dic, min_title, max_title):
    res = []
    column_names = ['stock', 'date', 'title', 'label']

    for comp in tqdm(company_list):
        comp_path = build_path(comp)
        df_temp = pd.read_csv(comp_path)

        date = list(df_temp['Date'])
        open_price = list(df_temp['Open'])
        close_price = list(df_temp['Close'])

        for i, dat in enumerate(date):
            if dat in news_dic[comp] and open_price[i]!=0:
                if min_title<=len(news_dic[comp][dat])<=max_title:
                    rate = (close_price[i]-open_price[i])/open_price[i]
                    label = get_label(rate, threshold)
                    group_news = '; '.join(x for x in news_dic[comp][dat])
                    res.append([comp, dat, group_news, label])
    
    res_df = pd.DataFrame(columns=column_names, data=res)
    return res_df

news_dic = get_news_data(news_data, consider_years, company_list)
print("News data obtained.")
price_news_df = build_dataset(company_list, news_dic, 2, 5)
print("Price data and news data are merged.\n")

print("The whole dataset has %d rows." % (len(price_news_df)))
train_df, test_df = train_test_split(price_news_df, test_size=test_size)

print("There are %d rows in training set; %d rows in test set." % (len(train_df), len(test_df)))
train_df.to_csv('LimitedComp/train_limit_news.csv', index=False)
test_df.to_csv('LimitedComp/test_limit_news.csv', index=False)