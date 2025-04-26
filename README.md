# Sentiment Based Product Recommendation

### Problem Statement

The e-commerce business is quite popular today. Here, you do not need to take orders by going to each customer. A company launches its website to sell the items to the end consumer, and customers can order the products that they require from the same website. Famous examples of such e-commerce companies are Amazon, Flipkart, Myntra, Paytm and Snapdeal.

 

Suppose you are working as a Machine Learning Engineer in an e-commerce company named 'Ebuss'. Ebuss has captured a huge market share in many fields, and it sells the products in various categories such as household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products and health care products.

 

With the advancement in technology, it is imperative for Ebuss to grow quickly in the e-commerce market to become a major leader in the market because it has to compete with the likes of Amazon, Flipkart, etc., which are already market leaders.

### Solution
Build a sentiment-based product recommendation system, which includes the following tasks.

* Data sourcing and sentiment analysis
* Building a recommendation system
* Improving the recommendations using the sentiment analysis model
* Deploying the end-to-end project with a user interface

* github link: https://github.com/Aswani-ReddyKV/Capstone_Sentiment_Recommendation

### Solution Approach

* The dataset and attribute descriptions are provided in the dataset folder for reference.
* Initial steps include Data Cleaning, Visualization, and Text Preprocessing (NLP) on the dataset. TF-IDF Vectorization is employed to convert textual data (review_title + review_text) into numerical vectors, measuring the relative importance of words across documents.
* Addressing the Class Imbalance Issue: SMOTE Oversampling technique is applied to balance the distribution of classes before model training.
* Machine Learning Classification Models: Various models such as Logistic Regression, Naive Bayes, and Tree Algorithms (Decision Tree, Random Forest, XGBoost) are trained on the vectorized data and target column (user_sentiment). The objective is to classify sentiment as positive (1) or negative (0). The best model is chosen based on evaluation metrics including Accuracy, Precision, Recall, F1 Score, and AUC. XGBoost emerges as the top performer.
* Collaborative Filtering Recommender System: Utilizing both User-User and Item-Item approaches, a recommender system is developed. Evaluation is performed using the RMSE metric.
SentimentBasedProductRecommendation.ipynb: This Jupyter notebook contains the code for Sentiment Classification and Recommender Systems.
* Product Sentiment Prediction: Top 20 products are filtered using the recommender system. For each product, user_sentiment is predicted for all reviews, and the top 5 products with higher positive user sentiment are selected (model.py).
* Model Persistence and Deployment: Machine Learning models are saved in pickle files within the pickle directory. A Flask API (app.py) is developed to interface and test these models. The User Interface is set up using Bootstrap and Flask Jinja templates (templates/index.html) without additional custom styles.