import nltk
import numpy as np
import pandas as pd
import pickle
import pprint
import project_helper
import re

from tqdm import tqdm


# Download stopwords and wordnet for further use
nltk.download('stopwords')
nltk.download('wordnet')


# Load the cik ticker file and create the mapping for further use
cik_df= pd.read_csv('cik_ticker.csv', sep="|")

cik_map = {}

for i in range(len(cik_df)):
    cik_map[cik_df.iloc[i]['Ticker']] = cik_df.iloc[i]['CIK']


# Load the company list
company_df = pd.read_csv('dis_comp.csv')
all_comp = []
for i in range(len(company_df)):
    all_comp.append(company_df.iloc[i]['Company'])


# Create a cik lookup mapping for the company we selected
cik_lookup = {}

for i, comp in enumerate(company):
    if comp in cik_map.keys():
        cik_lookup[comp] = cik_map[comp]


# Use SecAPI to get the 10Ks data
sec_api = project_helper.SecAPI()


# Pull a lost of filled 10-ks from the API for each company
from bs4 import BeautifulSoup

def get_sec_data(cik, doc_type, start=0, count=60):
    rss_url = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany' \
        '&CIK={}&type={}&start={}&count={}&owner=exclude&output=atom' \
        .format(cik, doc_type, start, count)
    sec_data = sec_api.get(rss_url)
    feed = BeautifulSoup(sec_data.encode('ascii'), 'xml').feed
    entries = [
        (
            entry.content.find('filing-href').getText(),
            entry.content.find('filing-type').getText(),
            entry.content.find('filing-date').getText())
        for entry in feed.find_all('entry', recursive=False)]

    return entries


# Pull the data, and show one of the examples
example_ticker = 'AMZN'
sec_data = {}

for ticker, cik in cik_lookup.items():
    sec_data[ticker] = get_sec_data(cik, '10-K')


# Download fillings from the urls we get in last step
raw_fillings_by_ticker = {}

for ticker, data in sec_data.items():
    raw_fillings_by_ticker[ticker] = {}
    for index_url, file_type, file_date in tqdm(data, desc='Downloading {} Fillings'.format(ticker), unit='filling'):
        if (file_type == '10-K'):
            file_url = index_url.replace('-index.htm', '.txt').replace('.txtl', '.txt')            
            
            raw_fillings_by_ticker[ticker][file_date] = sec_api.get(file_url)


# Get documents from the fillings
# To return a list of documents from a filling
def get_documents(text):
    extracted_docs = []
    doc_start_pattern = re.compile(r'<DOCUMENT>')
    doc_end_pattern = re.compile(r'</DOCUMENT>')   
    doc_start_is = [x.end() for x in doc_start_pattern.finditer(text)]
    doc_end_is = [x.start() for x in doc_end_pattern.finditer(text)]
    for doc_start_i, doc_end_i in zip(doc_start_is, doc_end_is):
            extracted_docs.append(text[doc_start_i:doc_end_i])
    return extracted_docs


# Extract the documents for the company
filling_documents_by_ticker = {}

for ticker, raw_fillings in raw_fillings_by_ticker.items():
    filling_documents_by_ticker[ticker] = {}
    for file_date, filling in tqdm(raw_fillings.items(), desc='Getting Documents from {} Fillings'.format(ticker), unit='filling'):
        filling_documents_by_ticker[ticker][file_date] = get_documents(filling)



# Define the function to get documents according to the type
def get_document_type(doc):
    type_pattern = re.compile(r'<TYPE>[^\n]+')
    doc_type = type_pattern.findall(doc)[0][len('<TYPE>'):] 
    return doc_type.lower()


# Only get the 10-k documents for the companies we selected
ten_ks_by_ticker = {}

for ticker, filling_documents in filling_documents_by_ticker.items():
    ten_ks_by_ticker[ticker] = []
    for file_date, documents in filling_documents.items():
        for document in documents:
            if get_document_type(document) == '10-k':
                ten_ks_by_ticker[ticker].append({
                    'cik': cik_lookup[ticker],
                    'file': document,
                    'file_date': file_date})


# Clean up - remove the html tags and lowercase all the text
from w3lib.html import remove_tags

def remove_html_tags(text):
    text = remove_tags(text)
    return text

def clean_text(text):
    text = text.lower()
    text = remove_html_tags(text)
    return text

for ticker, ten_ks in ten_ks_by_ticker.items():
    for ten_k in tqdm(ten_ks, desc='Cleaning {} 10-Ks'.format(ticker), unit='10-K'):
      #if ten_k['file']:
        ten_k['file_clean'] = clean_text(ten_k['file'])


# Lemmatize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def lemmatize_words(words):
    lemmatized_words = [WordNetLemmatizer().lemmatize(word, 'v') for word in words]
    return lemmatized_words

word_pattern = re.compile('\w+')

for ticker, ten_ks in ten_ks_by_ticker.items():
    for ten_k in tqdm(ten_ks, desc='Lemmatize {} 10-Ks'.format(ticker), unit='10-K'):
        ten_k['file_lemma'] = lemmatize_words(word_pattern.findall(ten_k['file_clean']))


# Remove Stopwords
from nltk.corpus import stopwords

lemma_english_stopwords = lemmatize_words(stopwords.words('english'))

for ticker, ten_ks in ten_ks_by_ticker.items():
    for ten_k in tqdm(ten_ks, desc='Remove Stop Words for {} 10-Ks'.format(ticker), unit='10-K'):
        ten_k['file_lemma'] = [word for word in ten_k['file_lemma'] if word not in lemma_english_stopwords]

# Here, the keys for each ten_k is ['cik', 'file', 'file_date', 'file_clean', 'file_lemma'].


# Transform the Data Format - from Dict to DataFrame
ten_ks_df_dict = {'date': [], 'company': [], 'ticker': [], 'doc': []}

for ticker, ten_ks in ten_ks_by_ticker.items():
    for ten_k in ten_ks:
        ten_ks_df_dict['date'].append(ten_k['file_date'])
        ten_ks_df_dict['company'].append(ticker)
        ten_ks_df_dict['ticker'].append(cik_lookup[ticker])
        #ten_ks_df_dict['lemma'].append(ten_k['file_lemma'])
        ten_ks_df_dict['doc'].append(' '.join(ten_k['file_lemma']))

ten_ks_df = pd.DataFrame(ten_ks_df_dict)


# Save the output data
ten_ks_df.to_csv('output.csv', index = False)
