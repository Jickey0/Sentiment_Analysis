from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np 
import pandas as pd
import joblib

# read my csv file using pandas
my_df = pd.read_csv("data_sets/IMDB Dataset.csv")

# seperate our reviews as positive and negative
positive = my_df[my_df['sentiment'] == 'positive']
negative = my_df[my_df['sentiment'] == 'negative']

# Concatenate positive and negative reviews using pandas
my_df_concat = pd.concat([positive, negative])

# https://www.youtube.com/watch?v=QAZc9xsQNjQ&t=1s&ab_channel=CS50
# Use Random Under-Sampling to avoid overfitting the dataset
rus = RandomUnderSampler(random_state = 0)

# https://imbalanced-learn.org/stable/over_sampling.html
# resample the dataset
my_df_sep_resampled = rus.fit_resample(my_df_concat[['review']], my_df_concat['sentiment'])

# sperate the resampled data set
df_review_sep = my_df_sep_resampled[0]
df_review_sep['sentiment'] = my_df_sep_resampled[1]

# seperate the data into a training and testing group
train, test = train_test_split(df_review_sep, test_size =0.35, random_state = 38)

# seperate training and testing datasets
train_x = train['review']
train_y = train['sentiment']
test_x = test['review']
test_y = test['sentiment']

# create a term frequeny vs inverse document frequency function
tfidf = TfidfVectorizer(stop_words = 'english')
train_x_vector = tfidf.fit_transform(train_x)

# use the training data to improve/train the model
test_x_vector = tfidf.transform(test_x)

# seperate the dataframe with sparse
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sparse.from_spmatrix.html
#pd.DataFrame.sparse.from_spmatrix(train_x_vector, index = train_x.index, columns = tfidf.get_feature_names_out())

# define our logistic regression model using sklearn
log_reg = LogisticRegression()

# create the model by "fitting" the dataset
# https://www.analyticsvidhya.com/blog/2022/07/sentiment-analysis-using-python/
log_reg.fit(train_x_vector, train_y)

# Save our TF-IDF word to vector tool and our log_reg model
np.save('models/TFIDF_Vectorize', tfidf)
joblib.dump(log_reg, 'models/IMDB_log_reg.joblib')