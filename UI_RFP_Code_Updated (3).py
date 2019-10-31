
# coding: utf-8

# In[148]:


#Imort modules
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import wordpunct_tokenize
from scipy import spatial
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import re
from string import digits
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder 
import math
from tika import parser


#Model definition

def cor(dataset, th):
    col_corr = set()
    corr_mat = dataset.corr()
    for i in range(len(corr_mat.columns)):
        for j in range(i):
            if (corr_mat.iloc[i,j] >= th) and (corr_mat.columns[j] not in col_corr):
                colname = corr_mat.columns[i]
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname]
    return(dataset)


def computeIDFDict(countDict):
    idfDict = {}
    for word in countDict:
        idfDict[word] = math.log(len(word) / countDict[word])
    return idfDict

def SL_rfp(rfp_name,filepath=''):
    #rfp_name = 'AXA Call centre'
    #rfp_name = "Syndicate Bank"
    #filepath = r"C:\Users\vsinha1\Documents\RFP\New-RFP\syndicate bank RFP.pdf"
    
    
    df_1 = pd.read_excel(r"C:\Users\vsinha1\Documents\RFP\FS_NON_FS_Tika_data.xlsx",sheet_name="Sheet1")
    df_1 = df_1[df_1['RFP_Data'].notnull()]
    df_1 = df_1[df_1['Service Line'].isin(['AI Engineering','Data_Science'])]
    if rfp_name in list(df_1['RFP']):
        data = df_1[df_1['RFP']==rfp_name][['RFP','Service Line','Sub Service Line','Sector','Sub-Sector']].values[0]
    else:
        parsedPDF = parser.from_file(filepath)
        #Appending data of testing RFP
        df = df_1.append({'RFP':rfp_name,'RFP_Data':parsedPDF['content']},ignore_index=True)
        tokens = []
        stop = set(stopwords.words('english'))
        lmtzr = WordNetLemmatizer()
        stemmer = SnowballStemmer("english")
        w = re.compile("[A-Za-z_]",re.I)
        d = re.compile("\d+")
        tf_idf = {}
        doc_tokenize = {}
        doc_no = {}
        for i in range(len(df.index)):
            try:
                remove_digits = str.maketrans('', '', digits)
                res = (df.loc[i,'RFP_Data']).translate(remove_digits)
                res = re.sub(r'_', '', res)
                review_list = wordpunct_tokenize(res.lower())
                noun_review = [lmtzr.lemmatize(word) for word in (review_list) if word not in punctuation]
                rm_stop_review = [review.strip for review in noun_review if review not in stop ]#and review not in stop_list
                doc_no_int = [review for review in noun_review if review not in stop]#and review not in stop_list
                tokens.append(' '.join(doc_no_int))
            except Exception as e:
                print(e)
        tfidf_vectorizer = TfidfVectorizer(norm=None, ngram_range=(1,2))
        new_docs = tokens
        new_term_freq_matrix = tfidf_vectorizer.fit_transform(new_docs)
        new_term_freq_matrix = new_term_freq_matrix.todense()
        df_tf = pd.DataFrame(new_term_freq_matrix)
        val = df_tf.copy()
        val.columns = tfidf_vectorizer.get_feature_names()
        km = []
        for i in val.columns:
            if sum(val[i])<40:
                km.append(i)
            else:
                continue
        val_new = val.drop(columns=km)
        val_cor = cor(val_new,0.5)
        num = val_cor.shape[0]
        test = val_cor.iloc[num-1]
        val_cor = val_cor.drop([num-1])
        val_cor = val_cor.reset_index(drop=True)
        target = df_1['Service Line']
        val_cor['Service_Line'] = LabelEncoder().fit_transform(target.astype(str))
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2
        X, y = val_cor.drop('Service_Line',axis=1),val_cor['Service_Line']

        # #Chi-Square
        select = SelectKBest(chi2, k=200)
        select.fit(X,y)
        X_new = select.transform(X)
        data = (val_cor.columns[select.get_support(indices=True)].tolist())

        #TF calculation of chi-squared feature name
        tf_d = []
        for i in range(len(tokens)):
            tf_val = {}
            for j in data:
                if j in tokens[i]:
                    if j in tf_val:
                        tf_val[j] += 1
                    else:
                        tf_val[j] = 1
            for word in tf_val:
                tf_val[word] = tf_val[word] / len(data)
            tf_d.append(tf_val)

        #TF-IDF calculation

        import math
        idfDict = []
        for i in range(len(tf_d)):
            idfDict.append(computeIDFDict(tf_d[i]))

        #Dataframe creation of TF-IDF
        fg=pd.DataFrame(idfDict)
        fg.fillna(0,inplace=True)
        test_new = fg.iloc[num-1]
        fg = fg.drop([num-1])
        fg = fg.reset_index(drop=True)

        fg['Service Line'] = val_cor['Service_Line']
        X,y=fg.drop('Service Line',axis=1),fg['Service Line']

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=52)
        from sklearn.naive_bayes import MultinomialNB
        clf = MultinomialNB()
        clf.fit(X_train, y_train)

        y_pred = clf.predict([test_new])

        #SSL and Sectors
        sector = pd.read_excel(r"C:\Users\vsinha1\Documents\RFP\sector_FS_NonFS_data.xlsx")
        sec = sector.loc[sector['RFP']==rfp_name]['Sector'].values[0]
        sub_sec = sector.loc[sector['RFP']==rfp_name]['Sub-Sector'].values[0]
        ssl = sector.loc[sector['RFP']==rfp_name]['Sub Service Line'].values[0]

        if y_pred == 0:
            df.loc[df['RFP']==rfp_name,'Service Line'] = 'AI Engineering'
            df.loc[df['RFP']==rfp_name,'Sub Service Line']=ssl
            df.loc[df['RFP']==rfp_name,'Sector']=sec
            df.loc[df['RFP']==rfp_name,'Sub-Sector']=sub_sec
        elif y_pred == 1:
            df.loc[df['RFP']==rfp_name,'Service Line'] = 'Data_Science'
            df.loc[df['RFP']==rfp_name,'Sub Service Line']=ssl
            df.loc[df['RFP']==rfp_name,'Sector']=sec
            df.loc[df['RFP']==rfp_name,'Sub-Sector']=sub_sec

        data = df[df['RFP']==rfp_name][['RFP','Service Line','Sub Service Line','Sector','Sub-Sector']].values[0]
    return data


# In[141]:


df_1 = pd.read_excel(r"C:\Users\vsinha1\Documents\RFP\FS_NON_FS_Tika_data.xlsx",sheet_name="Sheet1")
df_1 = df_1[df_1['RFP_Data'].notnull()]
df_1 = df_1[df_1['Service Line'].isin(['AI Engineering','Data_Science'])]


# In[147]:


'DBS' in list(df_1['RFP'])
#df_1[df_1['RFP']=='DBS'][['RFP','Service Line','Sub Service Line','Sector','Sub-Sector']].values[0]


# In[149]:


SL_rfp("DBS")


# In[2]:


#Label Similarity
def longest_common_substring(s1, s2):
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return s1[x_longest - longest: x_longest]

def similarity(s1, s2):
     return 2. * len(longest_common_substring(s1, s2)) / (len(s1) + len(s2)) * 100
    
def cal_dis(val,df,rfp):
    sl = {}
    l = []
    #i = len(df)-1
    i = df.loc[df['RFP']==rfp].index[0]
    for j in range(len(df)):
        l.append(similarity(df[val][i],df[val][j]))
    sl[i]=l
    return sl

from operator import itemgetter

def final_product(rfp):
    df = pd.read_excel(r"C:\Users\vsinha1\Documents\RFP\sector_FS_NonFS_data.xlsx")
    df = df[df['Sub-Sector'].notnull()]
    df = df.reset_index(drop=True)
    num = df.loc[df['RFP']==rfp].index[0]+1
    #df = ssl_model(r"C:\Users\vsinha1\Documents\RFP\FS_NON_FS_Tika_data.xlsx","Sheet7",rfp)
    #df = ssl_model(r"C:\Users\vsinha1\Documents\RFP\FS_NON_FS_Tika_data.xlsx","Sheet7",rfp)
    sl = cal_dis('Service Line',df,rfp)
    ssl = cal_dis('Sub Service Line',df,rfp)
    sec = cal_dis('Sector',df,rfp)
    sub_sec = cal_dis('Sub-Sector',df,rfp)
    row_sum = {}
    #row_sum[num-1]=[(sl[num-1][j]+ssl[num-1][j]+sec[num-1][j]+sub_sec[num-1][j])/4 for j in range(len(sl[num-1]))]
    row_sum[num-1]=[(sl[num-1][j]+ssl[num-1][j])/2 for j in range(len(sl[num-1]))]

    index_val = {}
    index_val[num-1]=sorted({k:v for k,v in enumerate(row_sum[num-1])}.items(), key=itemgetter(1),reverse=True)[:10]
    new_df = pd.DataFrame(columns=['ind','RFP','SL','SSL','Sector','Sub_Sector','Sim_val'])
    ind,rfp,sl,ssl,sec,sub,val=[],[],[],[],[],[],[]
    i=0
    for j,k in zip(list(dict(list(index_val.values())[i]).keys()),list(dict(list(index_val.values())[i]).values())):
        ind.append(i)
        rfp.append(df.iloc[j]['RFP'])
        sl.append(df.iloc[j]['Service Line'])
        ssl.append(df.iloc[j]['Sub Service Line'])
        sub.append(df.iloc[j]['Sector'])
        sec.append(df.iloc[j]['Sub-Sector'])
        val.append(k)
    new_df['ind']=ind
    new_df['RFP']=rfp
    new_df['SL']=sl
    new_df['SSL']=ssl
    new_df['Sector']=sec
    new_df['Sub_Sector']=sub
    new_df['Sim_val']=val
    #data = new_df[new_df['ind']==num-1]
    return new_df


# In[154]:


final_product('Farmers oasis')


# In[31]:


final_product("DBS")


# In[3]:


#Word2vec
from gensim.models import ldamodel
import gensim.corpora;
def avg_feature_vector(sentence, model, num_features, index2word_set): 
    words = sentence.split() 
    feature_vec = np.zeros((num_features, ), dtype='float32') 
    n_words = 0 
    for word in words: 
        if word in index2word_set: 
            n_words += 1 
            feature_vec = np.add(feature_vec, model[word]) 
    if (n_words > 0): 
        feature_vec = np.divide(feature_vec, n_words) 
    return feature_vec

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary = True)

index2word_set = set(model.wv.index2word)
import numpy as np


# In[155]:


#Content Similarity
#Content Similarity
import sys;
from nltk.corpus import stopwords;
import nltk;
from gensim.models import ldamodel
import gensim.corpora;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;
lmtzr = WordNetLemmatizer()
import pickle;
from nltk.corpus import stopwords
from string import punctuation
stop_words = set(stopwords.words('english'))
import re,os
import sklearn.feature_extraction.text as text
import numpy as np
import scipy

def topic_model(rfp):
    #final_df = final_product(rfp)
    if rfp=='Syndicate Bank':
        df = pd.read_excel(r"test_scope.xlsx",sheet_name = "Sheet8")
        df = df[pd.notnull(df['Scope'])]
        df = df.reset_index(drop = True)


        #vectorizer = text.CountVectorizer(input='df', stop_words='english', max_df=0.95)
        text = df[['Scope']]

        text = text.astype('str')
        tokens = []
        for i in range(0,len(text)):
            word_tokens = re.sub('[^a-zA-Z]', ' ', text['Scope'][i])
            word_tokens = word_tokens.lower()
            word_tokens= word_tokens.split()
            word_tokens=[word for word in word_tokens if word not in stop_words and word in model.vocab]
            word_tokens= ' '.join(word_tokens)
            tokens.append(word_tokens) 
        df['tokens']=tokens

        vectorizer = TfidfVectorizer(smooth_idf=False,ngram_range=(1,2))
        dtm = vectorizer.fit_transform(tokens)
        dtm_df = pd.DataFrame(dtm.todense())
        dtm_df.columns = vectorizer.get_feature_names()
        #dtm = vectorizer.fit_transform(x_counts).toarray()
        vocab = np.array(text['Scope'])
        from sklearn import decomposition
        num_topics = 2
        num_top_words = 100
        clf = decomposition.NMF(n_components=num_topics, random_state=None)
        doctopic = clf.fit_transform(dtm)
        topic_words = []
        for topic in clf.components_:
             word_idx = np.argsort(topic)[::-1][0:num_top_words]
             topic_words.append([dtm_df.columns[i] for i in word_idx])
        doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)
        #novel_names = []
        #for fn in df:
             #basename = os.path.basename(fn)
            # name, ext = os.path.splitext(basename)
            # name = name.rstrip('0123456789')
            # novel_names.append(name)

        novel_names = np.asarray(df.index)
        doctopic_orig = doctopic.copy()
        num_groups = len(set(novel_names))
        doctopic_grouped = np.zeros((num_groups, num_topics))
        for i, name in enumerate(sorted(set(novel_names))):
            doctopic_grouped[i, :] = np.mean(doctopic[novel_names == name, :], axis=0)
        NMF_doc_topics = pd.DataFrame(doctopic)
        # print(topic_words[0])
        # print(topic_words[1])
        # print(topic_words[2])
        #l=topic_words[0]+topic_words[1]
        topic_words = set(topic_words[0]+topic_words[1])
#         print(topic_words[0])
#         print(topic_words[1])
        #dts = chisquare_method(tokens,topic_words)
        #df['Do the topics match']=dts
        #return df[['RFP_Name','SL','SSL','Sector','Sub Sector','Do the topics match']]
        df.rename(columns={'RFP_Name':'RFP'},inplace=True)
        return {'tokens':tokens,'topic_words':topic_words,'df':df}
    elif rfp=='RBI-2019':
        df = pd.read_excel(r"test_scope.xlsx",sheet_name = "Sheet10")
        df = df[pd.notnull(df['Scope'])]
        df = df.reset_index(drop = True)


        #vectorizer = text.CountVectorizer(input='df', stop_words='english', max_df=0.95)
        text = df[['Scope']]

        text = text.astype('str')
        tokens = []
        for i in range(0,len(text)):
            word_tokens = re.sub('[^a-zA-Z]', ' ', text['Scope'][i])
            word_tokens = word_tokens.lower()
            word_tokens= word_tokens.split()
            word_tokens=[word for word in word_tokens if word not in stop_words and word in model.vocab]
            word_tokens= ' '.join(word_tokens)
            tokens.append(word_tokens) 
        df['tokens']=tokens

        vectorizer = TfidfVectorizer(smooth_idf=False,ngram_range=(1,2))
        dtm = vectorizer.fit_transform(tokens)
        dtm_df = pd.DataFrame(dtm.todense())
        dtm_df.columns = vectorizer.get_feature_names()
        #dtm = vectorizer.fit_transform(x_counts).toarray()
        vocab = np.array(text['Scope'])
        from sklearn import decomposition
        num_topics = 2
        num_top_words = 100
        clf = decomposition.NMF(n_components=num_topics, random_state=None)
        doctopic = clf.fit_transform(dtm)
        topic_words = []
        for topic in clf.components_:
             word_idx = np.argsort(topic)[::-1][0:num_top_words]
             topic_words.append([dtm_df.columns[i] for i in word_idx])
        doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)
        #novel_names = []
        #for fn in df:
             #basename = os.path.basename(fn)
            # name, ext = os.path.splitext(basename)
            # name = name.rstrip('0123456789')
            # novel_names.append(name)

        novel_names = np.asarray(df.index)
        doctopic_orig = doctopic.copy()
        num_groups = len(set(novel_names))
        doctopic_grouped = np.zeros((num_groups, num_topics))
        for i, name in enumerate(sorted(set(novel_names))):
            doctopic_grouped[i, :] = np.mean(doctopic[novel_names == name, :], axis=0)
        NMF_doc_topics = pd.DataFrame(doctopic)
        # print(topic_words[0])
        # print(topic_words[1])
        # print(topic_words[2])
        #l=topic_words[0]+topic_words[1]
        topic_words = set(topic_words[0]+topic_words[1])
#         print(topic_words[0])
#         print(topic_words[1])
        #dts = chisquare_method(tokens,topic_words)
        #df['Do the topics match']=dts
        #return df[['RFP_Name','SL','SSL','Sector','Sub Sector','Do the topics match']]
        df.rename(columns={'Sec':'Sector','Sub-Sector':'Sub Sector'},inplace=True)
        return {'tokens':tokens,'topic_words':topic_words,'df':df}
    elif rfp=='DBS':
        df = pd.read_excel(r"test_scope.xlsx",sheet_name = "Sheet11")
        df = df[pd.notnull(df['Scope'])]
        df = df.reset_index(drop = True)


        #vectorizer = text.CountVectorizer(input='df', stop_words='english', max_df=0.95)
        text = df[['Scope']]

        text = text.astype('str')
        tokens = []
        stopwords_1 = ['migration']
        for i in range(0,len(text)):
            word_tokens = re.sub('[^a-zA-Z]', ' ', text['Scope'][i])
            word_tokens = word_tokens.lower()
            word_tokens= word_tokens.split()
            word_tokens=[word for word in word_tokens if word not in stop_words and word not in stopwords_1 and word in model.vocab]
            word_tokens= ' '.join(word_tokens)
            tokens.append(word_tokens) 
        df['tokens']=tokens

        vectorizer = TfidfVectorizer(smooth_idf=False,ngram_range=(1,2))
        dtm = vectorizer.fit_transform(tokens)
        dtm_df = pd.DataFrame(dtm.todense())
        dtm_df.columns = vectorizer.get_feature_names()
        #dtm = vectorizer.fit_transform(x_counts).toarray()
        vocab = np.array(text['Scope'])
        from sklearn import decomposition
        num_topics = 5
        num_top_words = 50
        clf = decomposition.NMF(n_components=num_topics, random_state=None)
        doctopic = clf.fit_transform(dtm)
        topic_words = []
        for topic in clf.components_:
             word_idx = np.argsort(topic)[::-1][0:num_top_words]
             topic_words.append([dtm_df.columns[i] for i in word_idx])
        doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)
        #novel_names = []
        #for fn in df:
             #basename = os.path.basename(fn)
            # name, ext = os.path.splitext(basename)
            # name = name.rstrip('0123456789')
            # novel_names.append(name)

        novel_names = np.asarray(df.index)
        doctopic_orig = doctopic.copy()
        num_groups = len(set(novel_names))
        doctopic_grouped = np.zeros((num_groups, num_topics))
        for i, name in enumerate(sorted(set(novel_names))):
            doctopic_grouped[i, :] = np.mean(doctopic[novel_names == name, :], axis=0)
        NMF_doc_topics = pd.DataFrame(doctopic)
        # print(topic_words[0])
        # print(topic_words[1])
        # print(topic_words[2])
        #l=topic_words[0]+topic_words[1]
        topic_words = set(topic_words[0]+topic_words[1])
#         print(topic_words[0])
#         print(topic_words[1])
        #dts = chisquare_method(tokens,topic_words)
        #df['Do the topics match']=dts
        #return df[['RFP_Name','SL','SSL','Sector','Sub Sector','Do the topics match']]
        df.rename(columns={'Sec':'Sector','Sub-Sector':'Sub Sector'},inplace=True)
        return {'tokens':tokens,'topic_words':topic_words,'df':df}
    elif rfp=='RBI_CIMS':
        df = pd.read_excel(r"test_scope.xlsx",sheet_name = "Sheet12")
        df = df[pd.notnull(df['Scope'])]
        df = df.reset_index(drop = True)


        #vectorizer = text.CountVectorizer(input='df', stop_words='english', max_df=0.95)
        text = df[['Scope']]

        text = text.astype('str')
        tokens = []
        for i in range(0,len(text)):
            word_tokens = re.sub('[^a-zA-Z]', ' ', text['Scope'][i])
            word_tokens = word_tokens.lower()
            word_tokens= word_tokens.split()
            word_tokens=[word for word in word_tokens if word not in stop_words and word in model.vocab]
            word_tokens= ' '.join(word_tokens)
            tokens.append(word_tokens) 
        df['tokens']=tokens

        vectorizer = TfidfVectorizer(smooth_idf=False,ngram_range=(1,2))
        dtm = vectorizer.fit_transform(tokens)
        dtm_df = pd.DataFrame(dtm.todense())
        dtm_df.columns = vectorizer.get_feature_names()
        #dtm = vectorizer.fit_transform(x_counts).toarray()
        vocab = np.array(text['Scope'])
        from sklearn import decomposition
        num_topics = 3
        num_top_words = 100
        clf = decomposition.NMF(n_components=num_topics, random_state=None)
        doctopic = clf.fit_transform(dtm)
        topic_words = []
        for topic in clf.components_:
             word_idx = np.argsort(topic)[::-1][0:num_top_words]
             topic_words.append([dtm_df.columns[i] for i in word_idx])
        doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)
        #novel_names = []
        #for fn in df:
             #basename = os.path.basename(fn)
            # name, ext = os.path.splitext(basename)
            # name = name.rstrip('0123456789')
            # novel_names.append(name)

        novel_names = np.asarray(df.index)
        doctopic_orig = doctopic.copy()
        num_groups = len(set(novel_names))
        doctopic_grouped = np.zeros((num_groups, num_topics))
        for i, name in enumerate(sorted(set(novel_names))):
            doctopic_grouped[i, :] = np.mean(doctopic[novel_names == name, :], axis=0)
        NMF_doc_topics = pd.DataFrame(doctopic)
        # print(topic_words[0])
        # print(topic_words[1])
        # print(topic_words[2])
        #l=topic_words[0]+topic_words[1]
        topic_words = set(topic_words[0]+topic_words[1])
#         print(topic_words[0])
#         print(topic_words[1])
        #dts = chisquare_method(tokens,topic_words)
        #df['Do the topics match']=dts
        #return df[['RFP_Name','SL','SSL','Sector','Sub Sector','Do the topics match']]
        df.rename(columns={'Sec':'Sector','Sub-Sector':'Sub Sector'},inplace=True)
        return {'tokens':tokens,'topic_words':topic_words,'df':df}
    elif rfp=='Farmers oasis':
        df = pd.read_excel(r"test_scope.xlsx",sheet_name = "Sheet13")
        df = df[pd.notnull(df['Scope'])]
        df = df.reset_index(drop = True)


        #vectorizer = text.CountVectorizer(input='df', stop_words='english', max_df=0.95)
        text = df[['Scope']]

        text = text.astype('str')
        tokens = []
        for i in range(0,len(text)):
            word_tokens = re.sub('[^a-zA-Z]', ' ', text['Scope'][i])
            word_tokens = word_tokens.lower()
            word_tokens= word_tokens.split()
            word_tokens=[word for word in word_tokens if word not in stop_words and word in model.vocab]
            word_tokens= ' '.join(word_tokens)
            tokens.append(word_tokens) 
        df['tokens']=tokens

        vectorizer = TfidfVectorizer(smooth_idf=False,ngram_range=(1,2))
        dtm = vectorizer.fit_transform(tokens)
        dtm_df = pd.DataFrame(dtm.todense())
        dtm_df.columns = vectorizer.get_feature_names()
        #dtm = vectorizer.fit_transform(x_counts).toarray()
        vocab = np.array(text['Scope'])
        from sklearn import decomposition
        num_topics = 3
        num_top_words = 100
        clf = decomposition.NMF(n_components=num_topics, random_state=None)
        doctopic = clf.fit_transform(dtm)
        topic_words = []
        for topic in clf.components_:
             word_idx = np.argsort(topic)[::-1][0:num_top_words]
             topic_words.append([dtm_df.columns[i] for i in word_idx])
        doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)
        #novel_names = []
        #for fn in df:
             #basename = os.path.basename(fn)
            # name, ext = os.path.splitext(basename)
            # name = name.rstrip('0123456789')
            # novel_names.append(name)

        novel_names = np.asarray(df.index)
        doctopic_orig = doctopic.copy()
        num_groups = len(set(novel_names))
        doctopic_grouped = np.zeros((num_groups, num_topics))
        for i, name in enumerate(sorted(set(novel_names))):
            doctopic_grouped[i, :] = np.mean(doctopic[novel_names == name, :], axis=0)
        NMF_doc_topics = pd.DataFrame(doctopic)
        # print(topic_words[0])
        # print(topic_words[1])
        # print(topic_words[2])
        #l=topic_words[0]+topic_words[1]
        topic_words = set(topic_words[0]+topic_words[1])
#         print(topic_words[0])
#         print(topic_words[1])
        #dts = chisquare_method(tokens,topic_words)
        #df['Do the topics match']=dts
        #return df[['RFP_Name','SL','SSL','Sector','Sub Sector','Do the topics match']]
        df.rename(columns={'Sec':'Sector','Sub-Sector':'Sub Sector'},inplace=True)
        return {'tokens':tokens,'topic_words':topic_words,'df':df}


# In[565]:


len(topic_model('RBI-2019')['topic_words'])


# In[156]:


# d=topic_model('Syndicate Bank')
# d
topic_model('Farmers oasis')


# In[187]:


def chisquare_method(rfp):
    if rfp=='Syndicate Bank':
        data = topic_model(rfp)
        tokens = data['tokens']
        topic_words = data['topic_words']
        df = data['df']
        d1,d2,d3,d4,d5,d6={},{},{},{},{},{}
        word_freq_new = {'Syn':d1,'bow':d2,'Soc':d3,'hsbc':d4,'sbic':d5,'fe':d6}

        for i in range(len(tokens)):
            for j in set(topic_words):
                if i ==0:
                    d1[j]=tokens[i].count(j)
                elif i==1:
                    d2[j]=tokens[i].count(j)
                elif i==2:
                    d3[j]=tokens[i].count(j)
                elif i==3:
                    d4[j]=tokens[i].count(j)
                elif i==4:
                    d5[j]=tokens[i].count(j)
                elif i==5:
                    d6[j]=tokens[i].count(j)
        word_freq_new=pd.DataFrame(word_freq_new)
        word_freq_new['words']=word_freq_new.index
        df_w2vec_dis_new = pd.DataFrame()
        topic_words=topic_words
        for i in set(topic_words):
            avg= avg_feature_vector(i,model,num_features=300,index2word_set=index2word_set)
            df_w2vec_dis_new[i]=avg

        df_w2vec_dis_new=df_w2vec_dis_new.T
        from sklearn.cluster import KMeans
        num_clusters = 5

        cluster = {}

        km = KMeans(n_clusters=num_clusters)
        km.fit(df_w2vec_dis_new)
        clusters_syn = km.labels_.tolist()

        df_w2vec_dis_new['Cluster']=clusters_syn
        df_w2vec_dis_new['words'] = df_w2vec_dis_new.index
        d_new_1 = df_w2vec_dis_new[['words','Cluster']]

        df_new_1=d_new_1.set_index('words').join(word_freq_new.set_index('words'))
        nd1 = pd.DataFrame()

        l = []
        for i in range(5):
            d = df_new_1[df_new_1['Cluster']==i].shape[0]
            #d = df_new_1.shape[0]
            n=(sum(df_new_1[df_new_1['Cluster']==i]['Syn'])/d,sum(df_new_1[df_new_1['Cluster']==i]['bow'])/d,sum(df_new_1[df_new_1['Cluster']==i]['Soc'])/d,sum(df_new_1[df_new_1['Cluster']==i]['hsbc'])/d,sum(df_new_1[df_new_1['Cluster']==i]['sbic'])/d,sum(df_new_1[df_new_1['Cluster']==i]['fe'])/d)
            #print(n)
            l.append(n)
        nd1= pd.DataFrame(l)
        nd1=nd1.replace(0.000000,0.01)
        nd1.columns=[ 'Syn', 'bow', 'Soc', 'hsbc', 'sbic','fe']
        m = [ 'bow', 'Soc', 'hsbc', 'sbic','fe']
        pval = []
        for i in m:
            o_v = scipy.array(nd1['Syn'])
            e_v = scipy.array(nd1[i])
            pval.append(scipy.stats.chisquare(o_v,e_v)[1])

        dt = []
        for val in pval:
            if val>0.05:
                dt.append('Y')
            else:
                dt.append('N')

        return {'value':dt,'rfp':m,'df':df,'pval':pval}
    elif rfp=='RBI-2019':
        data = topic_model(rfp)
        tokens = data['tokens']
        topic_words = data['topic_words']
        df = data['df']
        d1,d2,d3,d4,d5,d6={},{},{},{},{},{}
        word_freq_new = {'rbi':d1,'bnpp':d2,'boi':d3,'dbs':d4,'db':d5,'hsbc':d6}

        for i in range(len(tokens)):
            for j in set(topic_words):
                if i ==0:
                    d1[j]=tokens[i].count(j)
                elif i==1:
                    d2[j]=tokens[i].count(j)
                elif i==2:
                    d3[j]=tokens[i].count(j)
                elif i==3:
                    d4[j]=tokens[i].count(j)
                elif i==4:
                    d5[j]=tokens[i].count(j)
                elif i==5:
                    d6[j]=tokens[i].count(j)
        word_freq_new=pd.DataFrame(word_freq_new)
        word_freq_new['words']=word_freq_new.index
        df_w2vec_dis_new = pd.DataFrame()
        topic_words=topic_words
        for i in set(topic_words):
            avg= avg_feature_vector(i,model,num_features=300,index2word_set=index2word_set)
            df_w2vec_dis_new[i]=avg

        df_w2vec_dis_new=df_w2vec_dis_new.T
        from sklearn.cluster import KMeans
        num_clusters = 5

        cluster = {}

        km = KMeans(n_clusters=num_clusters)
        km.fit(df_w2vec_dis_new)
        clusters_syn = km.labels_.tolist()

        df_w2vec_dis_new['Cluster']=clusters_syn
        df_w2vec_dis_new['words'] = df_w2vec_dis_new.index
        d_new_1 = df_w2vec_dis_new[['words','Cluster']]

        df_new_1=d_new_1.set_index('words').join(word_freq_new.set_index('words'))
        nd1 = pd.DataFrame()

        l = []
        for i in range(5):
            d = df_new_1[df_new_1['Cluster']==i].shape[0]
            #d = df_new_1.shape[0]
            n=(sum(df_new_1[df_new_1['Cluster']==i]['rbi'])/d,sum(df_new_1[df_new_1['Cluster']==i]['bnpp'])/d,sum(df_new_1[df_new_1['Cluster']==i]['boi'])/d,sum(df_new_1[df_new_1['Cluster']==i]['dbs'])/d,sum(df_new_1[df_new_1['Cluster']==i]['db'])/d,sum(df_new_1[df_new_1['Cluster']==i]['hsbc'])/d)
            #print(n)
            l.append(n)
        nd1= pd.DataFrame(l)
        nd1=nd1.replace(0.000000,0.01)
        nd1.columns=[ 'rbi', 'bnpp', 'boi', 'dbs', 'db','hsbc']
        m = [ 'bnpp', 'boi', 'dbs', 'db','hsbc']
        pval = []
        for i in m:
            o_v = scipy.array(nd1['rbi'])
            e_v = scipy.array(nd1[i])
            pval.append(scipy.stats.chisquare(o_v,e_v)[1])

        dt = []
        for val in pval:
            if val>0.05:
                dt.append('Y')
            else:
                dt.append('N')

        return {'value':dt,'rfp':m,'df':df,'pval':pval}
    elif rfp=='DBS':
        data = topic_model(rfp)
        tokens = data['tokens']
        topic_words = data['topic_words']
        df = data['df']
        d1,d2,d3,d4,d5={},{},{},{},{}
        word_freq_new = {'bar':d1,'dbs':d2,'deu':d3,'hcsc':d4,'hsbc':d5}

        for i in range(len(tokens)):
            for j in set(topic_words):
                if i ==0:
                    d1[j]=tokens[i].count(j)
                elif i==1:
                    d2[j]=tokens[i].count(j)
                elif i==2:
                    d3[j]=tokens[i].count(j)
                elif i==3:
                    d4[j]=tokens[i].count(j)
                elif i==4:
                    d5[j]=tokens[i].count(j)
               
        word_freq_new=pd.DataFrame(word_freq_new)
        word_freq_new['words']=word_freq_new.index
        df_w2vec_dis_new = pd.DataFrame()
        topic_words=topic_words
        for i in set(topic_words):
            avg= avg_feature_vector(i,model,num_features=300,index2word_set=index2word_set)
            df_w2vec_dis_new[i]=avg

        df_w2vec_dis_new=df_w2vec_dis_new.T
        from sklearn.cluster import KMeans
        num_clusters = 5

        cluster = {}

        km = KMeans(n_clusters=num_clusters)
        km.fit(df_w2vec_dis_new)
        clusters_syn = km.labels_.tolist()

        df_w2vec_dis_new['Cluster']=clusters_syn
        df_w2vec_dis_new['words'] = df_w2vec_dis_new.index
        d_new_1 = df_w2vec_dis_new[['words','Cluster']]

        df_new_1=d_new_1.set_index('words').join(word_freq_new.set_index('words'))
        nd1 = pd.DataFrame()

        l = []
        for i in range(5):
            d = df_new_1[df_new_1['Cluster']==i].shape[0]
            #d = df_new_1.shape[0]
            n=(sum(df_new_1[df_new_1['Cluster']==i]['bar'])/d,sum(df_new_1[df_new_1['Cluster']==i]['dbs'])/d,sum(df_new_1[df_new_1['Cluster']==i]['deu'])/d,sum(df_new_1[df_new_1['Cluster']==i]['hcsc'])/d,sum(df_new_1[df_new_1['Cluster']==i]['hsbc'])/d)
            #print(n)
            l.append(n)
        nd1= pd.DataFrame(l)
        nd1=nd1.replace(0.000000,0.01)
        nd1.columns=[ 'bar', 'dbs', 'deu', 'hcsc','hsbc']
        m = [ 'bar', 'deu', 'hcsc','hsbc']
        pval = []
        for i in m:
            o_v = scipy.array(nd1['dbs'])
            e_v = scipy.array(nd1[i])
            pval.append(scipy.stats.chisquare(o_v,e_v)[1])

        dt = []
        for val in pval:
            if val>0.05:
                dt.append('Y')
            else:
                dt.append('N')

        return {'value':dt,'rfp':m,'df':df,'pval':pval}
    elif rfp=='RBI_CIMS':
        data = topic_model(rfp)
        tokens = data['tokens']
        topic_words = data['topic_words']
        df = data['df']
        d1,d2,d3,d4,d5,d6={},{},{},{},{},{}
        word_freq_new = {'jlt':d1,'rbi_cims':d2,'rbs':d3,'sbi':d4,'adib':d5,'met':d6}

        for i in range(len(tokens)):
            for j in set(topic_words):
                if i ==0:
                    d1[j]=tokens[i].count(j)
                elif i==1:
                    d2[j]=tokens[i].count(j)
                elif i==2:
                    d3[j]=tokens[i].count(j)
                elif i==3:
                    d4[j]=tokens[i].count(j)
                elif i==4:
                    d5[j]=tokens[i].count(j)
                elif i==5:
                    d6[j]=tokens[i].count(j)
        word_freq_new=pd.DataFrame(word_freq_new)
        word_freq_new['words']=word_freq_new.index
        df_w2vec_dis_new = pd.DataFrame()
        topic_words=topic_words
        for i in set(topic_words):
            avg= avg_feature_vector(i,model,num_features=300,index2word_set=index2word_set)
            df_w2vec_dis_new[i]=avg

        df_w2vec_dis_new=df_w2vec_dis_new.T
        from sklearn.cluster import KMeans
        num_clusters = 5

        cluster = {}

        km = KMeans(n_clusters=num_clusters)
        km.fit(df_w2vec_dis_new)
        clusters_syn = km.labels_.tolist()

        df_w2vec_dis_new['Cluster']=clusters_syn
        df_w2vec_dis_new['words'] = df_w2vec_dis_new.index
        d_new_1 = df_w2vec_dis_new[['words','Cluster']]

        df_new_1=d_new_1.set_index('words').join(word_freq_new.set_index('words'))
        nd1 = pd.DataFrame()

        l = []
        for i in range(5):
            d = df_new_1[df_new_1['Cluster']==i].shape[0]
            #d = df_new_1.shape[0]
            n=(sum(df_new_1[df_new_1['Cluster']==i]['jlt'])/d,sum(df_new_1[df_new_1['Cluster']==i]['rbi_cims'])/d,sum(df_new_1[df_new_1['Cluster']==i]['rbs'])/d,sum(df_new_1[df_new_1['Cluster']==i]['sbi'])/d,sum(df_new_1[df_new_1['Cluster']==i]['adib'])/d,sum(df_new_1[df_new_1['Cluster']==i]['met'])/d)
            #print(n)
            l.append(n)
        nd1= pd.DataFrame(l)
        nd1=nd1.replace(0.000000,0.01)
        nd1.columns=[ 'jlt', 'rbi_cims', 'rbs', 'sbi', 'adib','met']
        m = [ 'jlt', 'rbs', 'sbi', 'adib','met']
        pval = []
        for i in m:
            o_v = scipy.array(nd1['rbi_cims'])
            e_v = scipy.array(nd1[i])
            pval.append(scipy.stats.chisquare(o_v,e_v)[1])

        dt = []
        for val in pval:
            if val>0.05:
                dt.append('Y')
            else:
                dt.append('N')

        return {'value':dt,'rfp':m,'df':df,'pval':pval}
    elif rfp=='Farmers oasis':
        data = topic_model(rfp)
        tokens = data['tokens']
        topic_words = data['topic_words']
        df = data['df']
        d1,d2,d3,d4,d5,d6={},{},{},{},{},{}
        word_freq_new = {'far':d1,'united':d2,'morgan':d3,'pro':d4,'wells':d5,'hvhc':d6}

        for i in range(len(tokens)):
            for j in set(topic_words):
                if i ==0:
                    d1[j]=tokens[i].count(j)
                elif i==1:
                    d2[j]=tokens[i].count(j)
                elif i==2:
                    d3[j]=tokens[i].count(j)
                elif i==3:
                    d4[j]=tokens[i].count(j)
                elif i==4:
                    d5[j]=tokens[i].count(j)
                elif i==5:
                    d6[j]=tokens[i].count(j)
        word_freq_new=pd.DataFrame(word_freq_new)
        word_freq_new['words']=word_freq_new.index
        df_w2vec_dis_new = pd.DataFrame()
        topic_words=topic_words
        for i in set(topic_words):
            avg= avg_feature_vector(i,model,num_features=300,index2word_set=index2word_set)
            df_w2vec_dis_new[i]=avg

        df_w2vec_dis_new=df_w2vec_dis_new.T
        from sklearn.cluster import KMeans
        num_clusters = 5

        cluster = {}

        km = KMeans(n_clusters=num_clusters)
        km.fit(df_w2vec_dis_new)
        clusters_syn = km.labels_.tolist()

        df_w2vec_dis_new['Cluster']=clusters_syn
        df_w2vec_dis_new['words'] = df_w2vec_dis_new.index
        d_new_1 = df_w2vec_dis_new[['words','Cluster']]

        df_new_1=d_new_1.set_index('words').join(word_freq_new.set_index('words'))
        nd1 = pd.DataFrame()

        l = []
        for i in range(5):
            d = df_new_1[df_new_1['Cluster']==i].shape[0]
            #d = df_new_1.shape[0]
            n=(sum(df_new_1[df_new_1['Cluster']==i]['far'])/d,sum(df_new_1[df_new_1['Cluster']==i]['united'])/d,sum(df_new_1[df_new_1['Cluster']==i]['morgan'])/d,sum(df_new_1[df_new_1['Cluster']==i]['pro'])/d,sum(df_new_1[df_new_1['Cluster']==i]['wells'])/d,sum(df_new_1[df_new_1['Cluster']==i]['hvhc'])/d)
            #print(n)
            l.append(n)
        nd1= pd.DataFrame(l)
        nd1=nd1.replace(0.000000,0.01)
        nd1.columns=[ 'far', 'united', 'morgan', 'pro', 'wells','hvhc']
        m = [ 'united', 'morgan', 'pro', 'wells','hvhc']
        pval = []
        for i in m:
            o_v = scipy.array(nd1['far'])
            e_v = scipy.array(nd1[i])
            pval.append(scipy.stats.chisquare(o_v,e_v)[1])

        dt = []
        for val in pval:
            if val>0.05:
                dt.append('Y')
            else:
                dt.append('N')

        return {'value':dt,'rfp':m,'df':df,'pval':pval}
    


# In[200]:


chisquare_method('Syndicate Bank')


# In[307]:


def final_output(rfp):
    #rfp='RBI-2019'
    dt =['Y']
    pval = chisquare_method(rfp)['pval']
    #pval.insert(0,1)
    #pval_float = [(float(i)) for i in pval]
    fg = pd.DataFrame()
    ft = pd.DataFrame()
    for i in range(5):
        fg[i]=(chisquare_method(rfp)['value'])
        ft[i]=(chisquare_method(rfp)['pval'])
    fg.index=chisquare_method(rfp)['rfp']
    fg=fg.T
    
    ft.index=chisquare_method(rfp)['rfp']
    ft=ft.T
    
    val = []
    for i in chisquare_method(rfp)['rfp']:
        try:
            #print(fg[i].value_counts()['N'])
            if 'N' in fg[i].value_counts() and fg[i].value_counts()['N']>3:
                dt.append('N')
                val.append(float(min(ft[i])))
    #             #elif 'N' in fg[i].value_counts() and fg[i].value_counts()['N']>=2:
            else:
                dt.append('Y')
                k = [i for i in ft[i] if i>0.05 ]
                val.append(float(min(k)))
        except Exception as e:
            e
    df = chisquare_method(rfp)['df']
    #return ft
    val.insert(0,1)
    df['Do the topics match']=dt
    df['Relevance']=val
    df['Relevance'].apply(float)
    df=df.rename(columns={'RFP':'RFP_Name','SL':'Service Line','SSL':'Sub Service Line'})
    return df[['RFP_Name','Service Line','Sub Service Line','Sector','Sub Sector','Do the topics match','Relevance']]


# In[311]:


final_output('RBI_CIMS')


# In[280]:


p['Relevance Percentage'].apply(float)*100


# In[261]:


dp=final_output('Syndicate Bank')


# In[238]:


dp['fe'].mean()*100


# In[239]:


dp.loc[5]=dp.mean()*100


# In[243]:


dp.loc[5]


# In[152]:


p.rename(columns={'RFP':'RFP_Name','SL':'Service Line','SSL':'Sub Service Line'})


# In[32]:


final_output('RBI_CIMS')


# In[167]:


l=[1,2,3]


# In[168]:


l.insert(0,5)


# In[169]:


l


# In[170]:


0>0.05

