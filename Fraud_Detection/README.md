<H1 align="center">Fraud Detection Take Home</H1>
<h3 align="center">By Jarrod Valentine</h3>

<H3 align="center">Introduction</H3>
As an e-commerce company, fraud is a major concern for our bottom line. Fradulent transactions erode not only our profits, but also our customers' trust in our ability to handle their money securely. The task here is to accurately identify whether a user has a high probability of using our site to perform some sort of fraudulent activity, e.g. money laundering, using stolen credit cards, etc.

<H3 align="center">Data</H3>
The data available to me to predict fraudulent activity for Company XYZ consists of two tables. The first table gives us information about roughly 150,000 users' first transactions, and which have been previously labeled as fraud. The second table gives us roughly 140,000 upper and lower bounds of IP addresses, along with their associated country names.

<H3 align="center">Exploratory Data Analysis</H3>
After some exploration of the data, I found that 9.36% of transactions were fraudulent. The average fraudulent purchase value is $36, while the average age of a user committing fraud is 33 years. 

<H5 align="center">IP Addresses and Countries</H5>
The first task in my analysis was to map each user's IP address to its country of origin, using data from both tables. I checked each IP address against the lower and upper bounds for each of the 140,000 IP bands to match each to a country. This was a computationally slow process, but could probably be greatly improved in the future using hashing tables. Of all the IP addresses matched, about 22,000 were from an "Unknown" country. With about 14,000 fraudulent transactions, 10 countries accounted for 80% of all fraud. 39% of fraudulent events were from IP addresses within the United States, and 13% were from an "Unknown" country. For this reason, I chose not to include a user's country as a feature in predicting fraud.

<p align="center"><img src="https://github.com/jkvalentine/Fraud_Detection/blob/master/images/percent_fraud_by_country.png?raw=true" width="800" /></p>

<H5 align="center">Feature Engineering</H5>
In my data exploration, I found that 42% of all fraudulent transactions were performed using the Chrome web browser. With only 5 browsers logged, this seemed significant, and so I used this as a feature in my model. 
</br>
</br>
I created the feature `time_to_purchase` by taking the difference between `purchase_time` and `signup_time` to see if this was correlated with fraud, and ,lo and behold, 56% of fraudulent transactions occurred within a period of 5 days between signup and purchase. The mean `time_to_purchase` was 57 days with a standard deviation of 36 days. With 62% of frauds occuring 1 standard deviation below the mean, this seems to be a strong indicator of fraud.
</br>
</br>
The sex of a user was not indicative of fraud, with males accounting for 58% of the user population and 59% of fraudulent transactions, so I chose not to use this as a feature in my model. 

<H3 align="center">The Model, Costs, and Predictions</H3>
To make my fraud prediction model, I chose a random forest classifier because ensemble models generally perform better than non-ensemble ones. The most important features in this model give us a good idea of what factors go into predicting a transaction as fraudulent. The most important features of this model are the purchase value, the age of the user, and the time between purchase and signup.
</br>
</br>
When considering metric of success of this model, we must consider the costs of predicting fraud when it actually occurs, predicting it when it hasn't occured, and failing to predict it when it has actually occured. If we predict that a transaction is not fraudulent when it actually is, we potentially lose money and our customer's trust. When we predict that a transaction is fraudulent when it is not, we have the potential to lose the customer's trust or their interest in using our site if this prediction interferes with their interaction with the site. For these reasons, we want to focus on recall to minimize false negative fraud predictions and on precision for false positive fraud predictions. The model currently predicts with a precision of 86% and a recall of 54%. The recall is slightly better than a random guess, but much better than a guess that adheres to the imbalanced class of 9% cases as fraud.
</br>
</br>
According to this model, a user who is older, whose time to first transaction is nearly immediate, and whose purchase value is higher than the average transaction price of 36 dollars has a higher probability of being fraud.

<H3 align="center">Future Work</H3>
This model has some definite room for improvement. In the future I would like to switch from a random forest classifier to a logistic regression model to offer our company a more interpretable model. This will also allow us to give a better breakdown of probability that a user will commit a fraudulent transaction and we will be able to communicate why we think the transaction is fraudulent, improving customer relations. 
</br>
</br>
I think the use of an ROC curve will help illuminate whether a random forest model or ligistic regression model will be most beneficial in this context. It will also help us determine the threshold that would be best for us to use given that we want to maximize our recall while minimizing our fall-out. 





