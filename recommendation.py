import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt


class popularity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.popularity_recommendations = None

    # Create the popularity based recommender system model
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        # Get a count of cvs for each unique job as recommendation score
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        train_data_grouped.rename(columns={'user_id': 'score'}, inplace=True)

        # Sort the cvs based upon recommendation score
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending=[0, 1])

        # Generate a recommendation rank based upon score
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')

        # Get the top 10 recommendations
        self.popularity_recommendations = train_data_sort.head(10)

    # Use the popularity based recommender system model to
    # make recommendations
    def recommend(self, user_id):
        user_recommendations = self.popularity_recommendations

        # Add job_id column for which the recommendations are being generated
        user_recommendations['user_id'] = user_id

        # Bring job_id column to the front
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]

        return user_recommendations


# Class for Item similarity based Recommender System model
class item_similarity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.cvs_dict = None
        self.rev_cvs_dict = None
        self.item_similarity_recommendations = None

    # Get unique cvs corresponding to a given job
    def get_user_items(self, user):
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_items = list(user_data[self.item_id].unique())

        return user_items

    # Get unique jobss for a given cvs
    def get_item_users(self, item):
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set(item_data[self.user_id].unique())

        return item_users

    # Get unique cvs in the training data
    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.item_id].unique())

        return all_items

    # Construct cooccurence matrix
    def construct_cooccurence_matrix(self, user_jobs, all_items):

        ####################################
        # Get users for all cvs in user_cvs.
        ####################################
        user_cvs_users = []
        for i in range(0, len(user_jobs)):
            user_cvs_users.append(self.get_item_users(user_jobs[i]))

        ###############################################
        # Initialize the item cooccurence matrix of size
        # len(user_cvs) X len(cvs)
        ###############################################
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_jobs), len(all_items))), float)

        #############################################################
        # Calculate similarity between user cvs and all unique cvs
        # in the training data
        #############################################################
        for i in range(0, len(all_items)):
            # Calculate unique jobs (users) of cvs (item) i
            cvs_i_data = self.train_data[self.train_data[self.item_id] == all_items[i]]
            users_i = set(cvs_i_data[self.user_id].unique())

            for j in range(0, len(user_jobs)):

                # Get unique job (users) of cvs (item) j
                users_j = user_cvs_users[j]

                # Calculate intersection of jobs of cvs i and j
                users_intersection = users_i.intersection(users_j)

                # Calculate cooccurence_matrix[i,j] as Jaccard Index
                if len(users_intersection) != 0:
                    # Calculate union of jobs of cvs i and j
                    users_union = users_i.union(users_j)

                    cooccurence_matrix[j, i] = float(len(users_intersection)) / float(len(users_union))
                else:
                    cooccurence_matrix[j, i] = 0

        return cooccurence_matrix

    # Use the cooccurence matrix to make top recommendations
    def generate_top_recommendations(self, user, cooccurence_matrix, all_items, user_jobs):
        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix))

        # Calculate a weighted average of the scores in cooccurence matrix for all user cvs.
        user_sim_scores = cooccurence_matrix.sum(axis=0) / float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()

        # Sort the indices of user_sim_scores based upon their value
        # Also maintain the corresponding score
        sort_index = sorted(((e, i) for i, e in enumerate(list(user_sim_scores))), reverse=True)

        # Create a dataframe from the following
        columns = ['job_id', 'cv', 'score', 'rank']
        # index = np.arange(1) # array of numbers for the number of samples
        df = pd.DataFrame(columns=columns)

        # Fill the dataframe with top 10 item based recommendations
        rank = 1
        for i in range(0, len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_items[sort_index[i][1]] not in user_jobs and rank <= 10:
                df.loc[len(df)] = [user, all_items[sort_index[i][1]], sort_index[i][0], rank]
                rank = rank + 1

        # Handle the case where there are no recommendations
        if df.shape[0] == 0:
            print("The current job has no cvs for training the item similarity based recommendation model.")
            return -1
        else:
            return df

    # Create the item similarity based recommender system model
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    # Use the item similarity based recommender system model to
    # make recommendations
    def recommend(self, user):

        ########################################
        # A. Get all unique cvs for this job
        ########################################
        user_jobs = self.get_user_items(user)

        print("No. of unique cvs for the job: %d" % len(user_jobs))

        ######################################################
        # B. Get all unique items (cvs) in the training data
        ######################################################
        all_items = self.get_all_items_train_data()

        print("no. of unique cvs in the training set: %d" % len(all_items))

        ###############################################
        # C. Construct item cooccurence matrix of size
        # len(job_ids) X len(cvs)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_jobs, all_items)

        #######################################################
        # D. Use the cooccurence matrix to make recommendations
        #######################################################
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_items, user_jobs)

        return df_recommendations

    # Get similar items to given items
    def get_similar_items(self, item_list):

        user_jobs = item_list

        ######################################################
        # B. Get all unique items in the training data
        ######################################################
        all_items = self.get_all_items_train_data()

        print("no. of unique cvs in the training set: %d" % len(all_items))

        ###############################################
        # C. Construct item cooccurence matrix of size
        # len(job_ids) X len(cvs)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_jobs, all_items)

        #######################################################
        # D. Use the cooccurence matrix to make recommendations
        #######################################################
        user = ""
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_items, user_jobs)


        return df_recommendations


job_df = pd.read_csv('/home/student/job_pool.csv')
cv_df = pd.read_csv('/home/student/cv_pool.csv')
job_cv_df=pd.read_csv('/home/student/job_cv.csv')
merged_df=pd.merge(cv_df, job_df, on="Designation", how="inner")
#mergerd_df = pd.merge(job_df,job_cv_df)
print(merged_df.shape)
print(merged_df.head())

cv_grouped = merged_df.groupby(['CV_Id']).agg({'count': 'count'}).reset_index()
total_sum= cv_grouped['count'].sum()
cv_grouped['percent']=cv_grouped['count'].div(total_sum)*100
print(cv_grouped['percent'])
dat=cv_grouped.sort_values(['count','CV_Id'], ascending = [0,1])
print(dat)

jobs = merged_df['Job_Id'].unique()
cvs =  merged_df['CV_Id'].unique()

print(jobs)

train_data, test_data = train_test_split(dat, test_size = 0.20, random_state=0)
pm = popularity_recommender_py()
pm.create(train_data,'Job_Id','CV_Id')
test_id=jobs[1]
pm.recommend(test_id)




#print("success")
