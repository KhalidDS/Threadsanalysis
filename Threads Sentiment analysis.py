#!/usr/bin/env python
# coding: utf-8

# <h1> Hi</h1>

# In[1]:


#install google play scraper
get_ipython().system('pip install google-play-scraper')


# In[2]:


from google_play_scraper import app
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings(action='ignore')
import emoji
import contractions
import re
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
from textstat import flesch_reading_ease 
import demoji


# In[3]:


#from google_play_scraper import Sort, reviews_all


#threads_reviews = reviews_all(
   # 'com.instagram.barcelona',
   # sleep_milliseconds=0, # defaults to 0
   # lang='en', # defaults to 'en'
    
   # sort=Sort.NEWEST, # defaults to Sort.MOST_RELEVANT
#)


# In[4]:


#df = pd.DataFrame(np.array(threads_reviews),columns=['review'])


#df = df.join(pd.DataFrame(df.pop('review').tolist()))



#df


# In[5]:


#df.to_csv('threads_reviews.csv', index=False)
df=pd.read_csv('threads_reviews.csv')
df


# In[6]:


#show count of  null or missing values in each column
df.isnull().sum()


# In[7]:


#show number of duplicated rows
df.duplicated().sum()


# In[8]:


#Print out the most frequent value in the "country" and "children" columns
print('The most frequent App version is : ', df["appVersion"].mode().iloc[0])


# In[9]:


#Replace the missing value in "children" and "country" columns with the most frequent values.
df["appVersion"].fillna(df["appVersion"].mode().iloc[0], inplace = True)


# In[10]:


#show count of  null or missing values in each column
df.isnull().sum()


# In[11]:


#Delete unwanted columns
df.drop(['reviewId', 'userName','userImage','thumbsUpCount','reviewCreatedVersion','at','replyContent','repliedAt'], axis = 1, inplace = True)


# In[12]:


df.head()


# In[13]:


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('words')


# In[14]:


sid = SentimentIntensityAnalyzer()
words = set(nltk.corpus.words.words())


# In[15]:


positive_scores = []
negative_scores = []
neutral_scores = []
compound_scores = []


    
for index, row in df.iterrows():
    contents = row['content']
    
    sentiment_scores = sid.polarity_scores(contents)
    
    positive_score = sentiment_scores['pos']
    negative_score = sentiment_scores['neg']
    neutral_score = sentiment_scores['neu']
    compound_score = sentiment_scores['compound']
    
    positive_scores.append(positive_score * 100)
    negative_scores.append(negative_score * 100)
    neutral_scores.append(neutral_score * 100)
    compound_scores.append(compound_score * 100)


# In[16]:


df['POSITIVE SCORE'] = positive_scores
df['NEGATIVE SCORE'] = negative_scores
df['NEUTRAL SCORE'] = neutral_scores
df['COMPOUND SCORE'] = compound_scores


df.head()


# 
# 
# plt.figure(figsize=(10,6))
# plt.hist(df['COMPOUND SCORE'], bins=20)
# plt.xlabel('Compound Score')
# plt.ylabel('Frequency')
# plt.title('Distribution of Compund')
# plt.show()

# In[18]:


positive_threshold = 0.2
negative_threshold = -0.2

df['Sentiment']= ''

for index, row in df.iterrows():
    compound_score = row['COMPOUND SCORE']
    
    if compound_score > positive_threshold:
        df.at[index, 'Sentiment'] = 'Positive'
    elif compound_score < negative_threshold:
            df.at[index, 'Sentiment'] = 'Negative'
    else:
                df.at[index, 'Sentiment'] = 'Neutral'


# In[19]:


df.sample(5)


# In[20]:


df['Sentiment'].value_counts()


# In[21]:


dff=df[['content','Sentiment']]


# In[22]:


demoji.download_codes()


# In[23]:


def remove_emojis(text):
    return demoji.replace(text, '')


# In[24]:


#Applying the remove_emojis function
dff['content'] = dff['content'].apply(remove_emojis)


# In[25]:


dff


# In[26]:


nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[27]:


corpus = []
ps=PorterStemmer()
for i in range(len(dff)):
    #Removing special charcters from text
    review = re.sub('[^a-zA-Z]', ' ',dff['content'][i])
    # Converting text to lowercase
    review = review.lower()
    #splitting text
    review = review.split()
    # stemming and remove stop words
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #joining  the words into a complete text
    review = ' '.join(review)
    #Appending text into the list corpus
    corpus.append(review)
    


# In[28]:


#convert labels into numerical format
dff['Sentiment'] = dff['Sentiment'].map({'Positive':1,'Neutral':0,'Negative':-1})


# In[29]:


#model training

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

#load and split data into train and test set
x_train, x_test, y_train, y_test = train_test_split(corpus, dff['Sentiment'], test_size=0.2, random_state=42)

#dfine a list of n values to evalute
n_values = [1, 2, 3]

#Iterate over each value of n

for n in n_values:
    #create countervectorizer with current n value
    vectorizer = CountVectorizer(ngram_range=(n, n))
    
    #fit and transform the training set
    x_train_transformed= vectorizer.fit_transform(x_train)
    
    #create a logistic regression model
    model = LogisticRegression(max_iter=1000)
    
    #Cross-validation
    cv_scores = cross_val_score(model, x_train_transformed, y_train, cv=5)
    
    #fit the model on the training set
    model.fit(x_train_transformed, y_train)
    
    #transform the test set
    
    x_test_transformed = vectorizer.transform(x_test)
    
    #Evalute the model on the test set
    test_accuracy = model.score(x_test_transformed, y_test)
    
    #results
    
    print(f"Results for n={n}")
    print("Cross-validation scores:", cv_scores)
    print("Mean CV accuracy:", cv_scores.mean())
    print("Test accuracy:", test_accuracy)
    print()


# In[30]:


from sklearn.feature_extraction.text import CountVectorizer

# instance of Countervectorizer with n-grams
n=1
vectorizer1 = CountVectorizer(ngram_range=(n, n))


# In[31]:


#fit and tranform the data

x = vectorizer1.fit_transform(corpus)
y=dff['Sentiment']


# In[32]:


feature_names = vectorizer1.get_feature_names_out()

print(feature_names)
print('No. of feature_words: ', len(feature_names))


# In[33]:


#train test split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x , y,test_size=0.2,random_state=42)


# In[34]:


#import libarary

from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[35]:


lr.fit(x_train,y_train)


# In[36]:


y_pred=lr.predict(x_test)


# In[48]:


#cheack the accuracy
y_pred = model.predict(x_test_transformed)

# Convert y_test and y_pred to integer type if needed
y_test = y_test.astype(int)
y_pred = y_pred.astype(int)

# Print evaluation metrics
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[39]:


from sklearn.ensemble import RandomForestClassifier
treemodel=RandomForestClassifier()


# In[40]:


treemodel.fit(x_train,y_train)


# In[41]:


y_pred=treemodel.predict(x_test)


# In[42]:


#cheack the accuracy

print("Accuracy score : \n",accuracy_score(y_pred,y_test))
print("Confusion Matrix : \n",confusion_matrix(y_pred,y_test))
print("Classfication Report : \n",classification_report(y_pred,y_test))


# In[43]:


from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()


# In[44]:


naive_bayes.fit(x_train,y_train)


# In[45]:


y_pred=naive_bayes.predict(x_test)


# In[46]:


#cheack the accuracy

print("Accuracy score : \n",accuracy_score(y_pred,y_test))
print("Confusion Matrix : \n",confusion_matrix(y_pred,y_test))
print("Classfication Report : \n",classification_report(y_pred,y_test))


# In[52]:


dff


# In[67]:


train = dff
train.set_axis(['content', 'Sentiment'], axis=1, inplace=True)
train


# In[69]:



from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
vectorizer=CountVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=0.01,max_df=0.3)
svc = LinearSVC()
pipeline = make_pipeline(vectorizer, svc)


# In[72]:



pipeline.fit(train.Sentiment, train.Sentiment)


# In[77]:


import eli5
# see weights for each feature.
eli5.show_weights(svc, vec=vectorizer, top=40)


# In[83]:





# In[84]:


df.to_csv('threadspreNLP.csv', index=False)


# In[86]:


dff.to_csv('threadspostNLP.csv', index=False)


# In[ ]:




