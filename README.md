# Sentiment Analysis of Product Review
## Context:
- The objective of this project is to develop a classification model to predict whether an upcoming product will elicit positive or negative sentiment based on customer feedback and related data.
## Background:
- A customer review is feedback provided by a customer based on their experience with a product or service they have purchased or used. These reviews serve as a valuable form of consumer insight, particularly on e-commerce and online shopping platforms. Research indicates that 90% of consumers read online reviews before making a purchase, and 88% trust these reviews as much as personal recommendations, highlighting their significant impact on purchasing decisions.
## WorkFlow
![image](https://github.com/user-attachments/assets/641833b1-2f27-4d79-9f7c-aae4c7d783cb)
## Data preprocessing and cleaning 
1. I concnated "Summary" and "ReviewText" column and created a new column "Reviews"
```
process_reviews['reviews']=process_reviews['reviewText']+process_reviews['summary']
process_reviews=process_reviews.drop(['reviewText', 'summary'], axis=1)
process_reviews.head()
```
2. Created a "Sentiment" column based on "Overall" column
3. Created a "Date", "Year" and "Month" columns from "ReviewTime" column
```
# Splitting the date 
new1 = process_reviews["date"].str.split(" ", n = 1, expand = True) 
# adding month to the main dataset 
process_reviews["month"]= new1[0] 
# adding day to the main dataset 
process_reviews["day"]= new1[1] 
process_reviews=process_reviews.drop(['date'], axis=1)
process_reviews.head()
```
4. Created a "helpful_rate" column using "Overall" column
5. Dropped 'reviewerName','unixReviewTime' from dataset
6. Then, created "word_count" column to count the words in "Reviews" column.
7. Removed stop words from dataset
```
process_reviews['reviews'] = process_reviews['reviews'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
process_reviews.head()
```
## EDA 
I created assumptions based on prior analysis:
- Higher the helpful rate the sentiment becomes positive
- There will be many negative sentiment reviews in the 2013 and 2014 year
- There will be more reviews at the starting of a month
  
1. Created violinplot using sentiment and helpfulness column
![image](https://github.com/user-attachments/assets/a42a0792-e761-44bb-9fec-e5b2302fc2a0)
**Insights:**
- From the plot we can observe that more number of positive reviews are having high helpful rate. We got deceived by the mean value, it's better to look at a plot rather than taking some measures of central tendency under such situation. Our first assumption is correct!

2. A plot was created to analyze the trend and distribution of sentiment counts, providing insights into how sentiments vary over the selected dimension.
![image](https://github.com/user-attachments/assets/c6f29f3b-e38f-44d6-83ff-55b39976ee8e)
**Insights:**
- From the plot we can clearly see the rise in positive reviews from 2010. Reaching its peak around 2013 and there is a dip in 2014, All the review rates were dropped at this time. Negative and neutral reviews are very low as compared to the positive reviews. Our second assumption is wrong!

3. A plot was generated to examine the daily distribution of review counts, providing insights into the volume of reviews submitted on each day.
![image](https://github.com/user-attachments/assets/145a4b74-4847-4b1b-bd48-c3c0daeb101b)
**Insights:**
- The review counts are more or less uniformly distributed.There isn't much variance between the days. But there is a huge drop at the end of month. Our third assumption is wrong.

## Feature Engineering 
- Created three new features "polarity", "review length" and "word count":
```
process_reviews['polarity'] = process_reviews['reviews'].map(lambda text: TextBlob(text).sentiment.polarity)
process_reviews['review_len'] = process_reviews['reviews'].astype(str).apply(len)
process_reviews['word_count'] = process_reviews['reviews'].map(lambda x: len(str(x).split()))
```
**1. Sentiment Polarity Distribution**
```
process_reviews['polarity'].iplot(
    kind='hist',
    bins=50,
    xTitle='polarity',
    linecolor='black',
    yTitle='count',
    title='Sentiment Polarity Distribution')
```
![image](https://github.com/user-attachments/assets/58d15b76-f254-4fcb-bef2-32684db91edb)
**Insights:**
The analysis revealed a significantly higher number of positive polarities compared to negative polarities, reflecting the prevalence of positive reviews in the dataset. The distribution of polarities aligns with the observed trend in review sentiments, confirming the dominance of positive feedback. While the polarity distribution appears to follow a normal pattern, it is not a standard normal distribution due to deviations in mean and standard deviation

**2. Review Rating Distribution**
```
process_reviews['overall'].iplot(
    kind='hist',
    xTitle='rating',
    linecolor='black',
    yTitle='count',
    title='Review Rating Distribution')
```
![image](https://github.com/user-attachments/assets/b7216555-d52f-4d75-898d-539816d07920)
**Insights**
It had large number of 5 ratings(nearly 7k) followed by 4,3,2,1. It's linear in nature.

**3. Review text world count distribution**
```
process_reviews['word_count'].iplot(
    kind='hist',
    bins=100,
    xTitle='word count',
    linecolor='black',
    yTitle='count',
    title='Review Text Word Count Distribution')
```
![image](https://github.com/user-attachments/assets/7ed7687c-7b20-4f25-89b5-4668d8b0bcc9)
**Insights:**
It is right skewed distribution with most of the words falling between 0-200 in a a review

**4. Review Text Length Distribution**
```
process_reviews['review_len'].iplot(
    kind='hist',
    bins=100,
    xTitle='review length',
    linecolor='black',
    yTitle='count',
    title='Review Text Length Distribution')
```
![image](https://github.com/user-attachments/assets/170f4e12-ce60-4b27-b587-b9a1cf5bba7e)
**Insights:**
It is a right skewed distribution where most of the lengths falls between 0-1000

- Created **N-gram analysis** to identify patterns in text data by analyzing the frequency and distribution of smaller text units.
- Lastly, Created wordcloud to analyze **positive, netural and negative words**.
- Performed **label encoding** on the target variable: sentiment.
```
label_encoder = preprocessing.LabelEncoder()  
process_reviews['sentiment']= label_encoder.fit_transform(process_reviews['sentiment']) 
process_reviews['sentiment'].unique()
```
- Performed stemming on the review dataframe.
```
ps = PorterStemmer()
corpus = []
for i in range(0, len(review_features)):
    review = re.sub('[^a-zA-Z]', ' ', review_features['reviews'][i])
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stop_words]
    review = ' '.join(review)
    corpus.append(review)    
```
- Utilized **TFIDF(Term Frequency — Inverse Document Frequency)** to measure the importance of each word in a sentence.
    - Used only top 5000 words from the reviews. 
```
tfidf_vectorizer = TfidfVectorizer(max_features = 5000, ngram_range=(2,2))
# TF-IDF feature matrix
X= tfidf_vectorizer.fit_transform(review_features['reviews'])
```
- Lastly, handled imbalanced data using SMOTE technique.
```
print(f'Original dataset shape : {Counter(y)}')
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print(f'Resampled dataset shape {Counter(y_res)}')
```
## Modeling
- Created multiple models and compared results. Utilized ** LogisticRegression, DecisionTreeClassifier, KNeighborsClassifier, SVC, BernoulliNB** algorithms. 
- Based on the resultes best performing model was LogisticRegression
![image](https://github.com/user-attachments/assets/915ffe1e-a815-41d7-90de-d0010be13898)
- Applied hyperparameter tuning on LogisticRegression
- Achieved 97% F-1 Score. 
![image](https://github.com/user-attachments/assets/18001486-0a60-4558-af20-167a61b800cd)
## Conclusion:
- **Data Preparation and Handling:** The process included splitting sentiments based on the overall score, cleaning the text, and customizing the stopword list to meet specific requirements.
- **Stopword Handling:** Standard stopword lists were avoided in sentiment analysis as they included words with negative connotations, which could have distorted the results. The stopword list was manually reviewed and adjusted for better accuracy.
- **N-gram Utilization:** Incorporating n-grams (e.g., bi-grams, tri-grams) significantly improved the accuracy of the sentiment analysis. Single words often failed to capture the contextual meaning effectively.
- **Neutral Sentiment Insights:** A large proportion of neutral reviews in the dataset turned out to be constructive critiques of products.
- **Recommendation:** Amazon could consider leveraging such neutral reviews as actionable feedback for sellers to improve their products.
- **Dataset Composition:** Most reviews in the dataset were related to string instruments, such as guitars, highlighting a potential niche focus.
- **Addressing Class Imbalance:** SMOTE (Synthetic Minority Oversampling Technique) was used to balance the dataset, resulting in significantly improved recall and F1 score.
- **Observation:** Before balancing, precision was good, but recall was low, which negatively impacted the F1 score. Balancing the target feature proved essential for robust performance.
- **Performance Metrics:** The F1 score was prioritized as the primary metric for evaluating the sentiment analysis model.
The analysis achieved an impressive average **F1 score of 96%**, indicating a high level of accuracy and balance in the classification results.






 



