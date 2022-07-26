#Before running ensure nltk is installed

import os
#used to visualise the graphs
import matplotlib.pyplot as plt
#BeautifulSoup used to parse data from website
from bs4 import BeautifulSoup
#Pandas library to store data in DataFrame objects
import pandas as pd
#Used to perform sentiment analysis on the news headlines
import nltk
nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#requests library to get the data
from urllib.request import urlopen, Request


def sentiment(codes):
    '''
    Extract and Store the Data
    '''
    #to parse a webside, the stock ticker is added to this URL
    web_url = 'https://finviz.com/quote.ashx?t='
    #create an empty dictionary
    news_tables = {}
    #these are the stocks we are analysing
    tickers = codes
    #make iterations, extracting news data for one of the stocks per iteration
    for tick in tickers:
        # add ticker onto URL
        url = web_url + tick
        #request the data
        req = Request(url=url,headers={"User-Agent": "Chrome"})
        response = urlopen(req)
        #parse the html
        html = BeautifulSoup(response,"html.parser")
        #we are looking for headings, found in the HTML of the webpage in a table under the id of 'news-table'
        news_table = html.find(id='news-table')
        #add it into the dictionary with the key being the ticker
        news_tables[tick] = news_table


    '''
    Code to parse date, time and headlines into a Python List
    '''
    #these will go into the empty new_list
    news_list = []
    #loop over the news
    for file_name, news_table in news_tables.items():
        #iterate over all the <tr> tages in news_table containing the headline
        for i in news_table.findAll('tr'):
            #.get_text() function extracts text placed within the <tr> tag, but only text within the <a> tag
            text = i.a.get_text()
            #.split() function splits text placed in <td> tag into a list
            date_scrape = i.td.text.split()
            #if the length of this split data = 1 , time will be loaded as the only element
            if len(date_scrape) == 1:
                time = date_scrape[0]
            #otherwise date will be loaded as the first element and time as the second
            else:
                date = date_scrape[0]
                time = date_scrape[1]
            tick = file_name.split('_')[0]
            news_list.append([tick, date, time, text])

    '''
    Sentiment Analysis Section
    '''
    vader = SentimentIntensityAnalyzer()
    columns = ['ticker', 'date', 'time', 'headline']
    news_df = pd.DataFrame(news_list, columns=columns)
    scores = news_df['headline'].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    news_df = news_df.join(scores_df, rsuffix='_right')
    news_df['date'] = pd.to_datetime(news_df.date).dt.date
    #this is an optional piece of 5 line code that removes totally neutral news
    for index, row in news_df.iterrows():
        if int(row.neu) == 1:
            news_df = news_df.drop(index)
        else:
            pass
    #.head() function returns rows of the dataframe
    print(news_df.head(n=len(news_df)))
    #the compound columns gives us the sentiment scores
    #1 is positive, -1 is negative

    '''
    Visualise the Sentiment Scores
    '''
    plt.rcParams['figure.figsize'] = [10, 6]
    mean_scores = news_df.groupby(['ticker','date']).mean()
    #.unstack() function helps to unstack the ticker column
    mean_scores = mean_scores.unstack()
    #.transpose() obtains the cross-section of compound in the columns axis
    mean_scores = mean_scores.xs('compound', axis="columns").transpose()
    #.plot() and set the kind of graph to 'bar'
    mean_scores.plot(kind = 'bar')
    plt.grid()
    plt.show()


'''
Test of the code with 4 stocks
'''
testsentiment = sentiment(['GOOG', 'AMZN', 'TSLA', 'AAPL'])
