import pandas as pd
import streamlit as st

import sklearn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import pickle

import seaborn as sns

# load data
df = pd.read_csv("drug_data.csv")

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.model_selection import train_test_split

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from contractions import contractions_dict


# Define the process_review function
def process_review(review):
    # Remove hashtags
    review = review.replace('#', '')
    
    # Expand contractions
    words = review.split()
    words = [contractions_dict[word] if word in contractions_dict else word for word in words]
    review = ' '.join(words)
    
    # Tokenize the text
    tokens = word_tokenize(review)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if not token in stop_words]
    
    # Remove punctuation
    tokens = [token for token in tokens if token.isalpha()]
    
    # Return the processed tokens
    return ' '.join(tokens)

# Clean the drug reviews
df['processed_review'] = df['review'].apply(process_review)



X = df[['drugName', 'condition','processed_review']]

import pandas as pd
# Create a new column for sentiment
df['Sentiment'] = ''

# Assign sentiment based on rating
df.loc[df['rating'] < 3, 'Sentiment'] = 'negative'
df.loc[df['rating'] > 7, 'Sentiment'] = 'positive'
df.loc[(df['rating'] >= 3) & (df['rating'] <= 7), 'Sentiment'] = 'neutral'

# Print the updated dataset with sentiment column
print(df)

y = df['Sentiment']


# One-hot encode the categorical features
ohe = OneHotEncoder()
X = ohe.fit_transform(X)

# Label encode the target variable
le = LabelEncoder()
y  = le.fit_transform(y)

# train SVM model
model = SVC(kernel="linear")
model.fit(X, y)

# save model as pickle file
from sklearn.svm import SVM
import pickle 
import pickle 
svm_model = SVC(kernel='linear')
pickle_out = open("model.pkl", mode = "wb") 
pickle.dump(svm_model, pickle_out) 
pickle_out.close()
# Create sidebar for filtering data
st.sidebar.subheader("Filter Reviews")
condition = st.sidebar.selectbox("Select Condition", df['Condition'].unique())
rating = st.sidebar.selectbox("Select Rating", df['Rating'].unique())

# Filter data based on user selection
filtered_data = df[(df['Condition'] == condition) & (df['Rating'] == rating)]

# Display filtered data
st.write("### Patient Reviews")
st.write(filtered_data)

# Visualize data
st.write("### Review Distribution")
sns.histplot(data=filtered_data, x='Review', kde=True)

st.pyplot()
