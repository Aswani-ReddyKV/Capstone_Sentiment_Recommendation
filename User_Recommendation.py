import numpy as np
import pandas as pd
import os


import pickle

import warnings
warnings.filterwarnings("ignore")

class User_Recommendation:
    def __init__(self):

        self.clean_df = pd.read_pickle("pickle_files/preprocessed-dataframe.pkl")
        self.clean_df_recommended = self.clean_df[['id','name','reviews_complete_text', 'user_sentiment']]

        self.user_final_rating = pd.read_pickle("pickle_files/user_final_rating.pkl")

        file = open("pickle_files/tfidf-vectorizer.pkl",'rb')
        self.vectorizer = pickle.load(file)
        file.close()

        try:
            # Use 'with' for safer file handling
            with open("pickle_files/sentiment-classification-xg-boost-model.pkl", 'rb') as file:
                self.model = pickle.load(file)
                print("Model loaded successfully.")
        except FileNotFoundError:
            print("Error: Pickle file not found.")
        except AttributeError as e:
            print(f"AttributeError during loading: {e}")
            print("This often means a class definition (like ModelFactory) was not found.")
            print("Ensure all custom classes used by the pickled object are defined or imported.")
        except Exception as e:
            print(f"An unexpected error occurred during loading: {e}")

    def get_top5_user_recommendations(self, user):
        if user in self.user_final_rating.index:
            # get the top 20  recommedation using the user_final_rating
            top20_reco = list(self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)
            # get the product recommedation using the orig data used for trained model
            common_top20_reco = self.clean_df_recommended[self.clean_df['id'].isin(top20_reco)]
            # Apply the TFIDF Vectorizer for the given 20 products to convert data in reqd format for modeling
            X =  self.vectorizer.transform(common_top20_reco['reviews_complete_text'].values.astype(str))

            # Using the model from param to predict
            # self.model.set_test_data(X)
            common_top20_reco['sentiment_pred']= self.model.predict(X)

            # Create a new dataframe "pred_df" to store the count of positive user sentiments
            temp_df = common_top20_reco.groupby(by='name').sum()
            # Create a new dataframe "pred_df" to store the count of positive user sentiments
            sent_df = temp_df[['sentiment_pred']]
            sent_df.columns = ['positive_sentiment_count']
            # Create a column to measure the total sentiment count
            sent_df['total_sentiment_count'] = common_top20_reco.groupby(by='name')['sentiment_pred'].count()
            # Calculate the positive sentiment percentage
            sent_df['positive_sentiment_percent'] = np.round(sent_df['positive_sentiment_count']/sent_df['total_sentiment_count']*100,2)
            # Return top 5 recommended products to the user
            result = sent_df.sort_values(by='positive_sentiment_percent', ascending=False)[:5]
            return result
        else:
            print(f"User name {user} doesn't exist")


def get_top5_recommendationsNew(user):
    user_recommendation = User_Recommendation()
    recommendation_df = user_recommendation.get_top5_user_recommendations(user=user).reset_index()
    print(recommendation_df)
    return recommendation_df