import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn import linear_model, preprocessing

# read csv into dataframe
csv_file = 'beer_reviews.csv'
df = pd.read_csv(csv_file)

# see dataframe headers
headers = list(df.columns)

print("\n----------------------------------------------------------------------------------------------------------\n")

# 1.     Which brewery produces the strongest beers by ABV%

# 1a. The strongest beer(s) by ABV% - the simple approach
# finding the max ABV in the whole dataset and back-tracing the brewery(s) which produces this beer(s)

ntop = 3

top_three_strongest = df.drop_duplicates(subset=['brewery_name', 'beer_name', 'beer_abv']).nlargest(ntop, columns='beer_abv').sort_values(by='beer_abv', ascending=False)

# just some re-indexing
top_three_strongest.reset_index(inplace=True)

print("The strongest beer in the dataset (highest ABV) has ABV {}%".format(df.beer_abv.max()))
print("The breweries producing the strongest {} beers are:".format(ntop))
for index, row in top_three_strongest.iterrows():
    print("#{} - Brewery: {}, Beer: {}, ABV: {}%".format(index+1, row.brewery_name, row.beer_name, row.beer_abv))
print('\n')
# 1b. The brewery(s) which produces the strongest beers by ABV% on average
# finding the average ABV of all beers produced by each brewery, then taking the max of the averages

brewery_avg_abv = df.groupby('brewery_name').beer_abv.mean()
top_three_brewery = brewery_avg_abv.nlargest(ntop).sort_values(ascending=False).reset_index()

print("The top {} breweries producing the strongest beers on average are:".format(ntop))
for index, row in top_three_brewery.iterrows():
    print("#{} - Brewery: {}, Average ABV: {:.2f}%".format(index+1, row.brewery_name, row.beer_abv))




print("\n----------------------------------------------------------------------------------------------------------\n")

# 2.     If you had to pick 3 beers to recommend using only this data, which would you pick?


# beer_avg_reviews = df.groupby('beer_name')['beer_name', 'review_overall', 'review_aroma', 'review_appearance', 'review_palate', 'review_taste'].mean()

# Only considering average overall scores is insufficient (perhaps due to small sample sizes for some beers)
# resulting in 615 beers with perfect 5.0 average overall scores
# the solution below filters out beers with too few reviews as defined by the sample size threshold, and picks the top rated beers amongst the remaining subset

# there does not seem to be much value to consider the other review scores besides overall i.e. aroma, appearance, palate and taste
# as it is assumed that they are encapsulated in the overall score i.e. overall = f(aroma, appearance, palate, taste) for some function f()
# and it is fairly reasonable to expect f() to be some kind of weighted/simple average (linear) function - more below

ntop = 3
max_score = 5
sample_size_threshold = 10

beer_review_counts = df.beer_name.value_counts().rename("review_count")
beer_avg_overall = df.groupby('beer_name').review_overall.mean()
avg_overall_counts = pd.concat([beer_avg_overall, beer_review_counts], axis=1)

# filters out those with < sample_size_threshold reviews
filtered_avg_overall = avg_overall_counts[avg_overall_counts.review_count >= sample_size_threshold]
top_avg_overall = filtered_avg_overall.nlargest(ntop, columns='review_overall')

# just some re-indexing and renaming
top_avg_overall.reset_index(inplace=True)
top_avg_overall.rename(columns={'index': 'beer_name'}, inplace=True)

print("The top {} beers recommended based on average overall scores and with at least {} reviews are:".format(ntop, sample_size_threshold))
for index, row in top_avg_overall.iterrows():
    print("#{} - Beer: {}, Average Overall Score: {:.2f}/{}, Total Reviews: {}".format(index+1, row.beer_name, row.review_overall, max_score, row.review_count))


    

print("\n----------------------------------------------------------------------------------------------------------\n")

# 3.     Which of the factors (aroma, taste, appearance, palate) are most important in determining the overall quality of a beer?

# 3a. As a superficial model-free (nonparametric) approach, only the pairwise correlation coefficients will be calculated
# and the factors will be ranked by them (as a superficial metric for importance) accordingly
# however it must be noted that correlation does not imply causation or even importance
# a visual plot of the relationship(s) could be investigated and the correlation matrix could be considered

review_headers = ['review_overall', 'review_aroma', 'review_taste', 'review_appearance', 'review_palate']
beer_review_scores = df[review_headers]
correlation_matrix = beer_review_scores.corr()
correlation_vector = correlation_matrix.review_overall.drop(labels='review_overall').sort_values(ascending=False).rename('correlation_coefficient')

# just some re-indexing and renaming
correlation_vector = correlation_vector.to_frame()
correlation_vector.reset_index(inplace=True)
correlation_vector.rename(columns={'index': 'factor'}, inplace=True)

print("The factors determining the overall quality of a beer ranked in decreasing importance (correlation) are:")
for index, row in correlation_vector.iterrows():
    print("#{} - Factor: {}, Correlation: {:.2f}".format(index+1, str(row.factor)[7:].capitalize(), row.correlation_coefficient))
print('\n')

# 3b. Here the standard multivariable/multiple linear regression is considered
# but the methodology could certainly be expanded to include other machine learning models

'''

# the data was not split into training and testing as in typical ML routines
X_train = np.array(beer_review_scores.loc[:, beer_review_scores.columns != 'review_overall'])
X_train = preprocessing.scale(X_train)

y_train = np.array(beer_review_scores.review_overall)

clf = linear_model.LinearRegression()
clf.fit(X_train, y_train)
clf.coef_
'''

# analysis of variance (ANOVA) and hypothesis testing could (should) be done to check model assumptions and validity for more robustness

X = beer_review_scores.loc[:, beer_review_scores.columns != 'review_overall']
X = sm.add_constant(X)
y = beer_review_scores.review_overall

model = sm.OLS(y, X).fit()
model.summary()
model_coefficients = model.params.drop(labels='const').sort_values(ascending=False).rename('model_coefficient')

# just some re-indexing and renaming
model_coefficients = model_coefficients.to_frame()
model_coefficients.reset_index(inplace=True)
model_coefficients.rename(columns={'index': 'factor'}, inplace=True)

print("The factors determining the overall quality of a beer ranked in decreasing importance (linear model) are:")
for index, row in model_coefficients.iterrows():
    print("#{} - Factor: {}, Model Coefficient: {:.2f}".format(index+1, str(row.factor)[7:].capitalize(), row.model_coefficient))




print("\n----------------------------------------------------------------------------------------------------------\n")

# 4.     Lastly, if I typically enjoy a beer due to its aroma and appearance, which beer style should I try?

# the approach adopted will be similar to # 2. - it is assumed that enjoyment is solely due to aroma and appearance only
# i.e. enjoyment = f(aroma, appearance) for some function f() and it is fairly reasonable to expect f() to be some kind of weighted/simple average (linear) function
# filtering beer styles with too few reviews is probably unnecessary as there is ample data (the beer style with the fewest reviews still has 241 reviews)

beer_style_avg = df.groupby('beer_style')['review_aroma', 'review_appearance'].mean()

# user-defined weight on aroma representing preference for aroma relative to appearance
ntop = 3
w_aroma = 0.5

beer_style_avg['weighted_avg'] = beer_style_avg.apply(lambda row: w_aroma*row.review_aroma +(1-w_aroma)*row.review_appearance, axis=1)
top_three_styles = beer_style_avg.nlargest(ntop, columns='weighted_avg').sort_values(by='weighted_avg', ascending=False)

# just some re-indexing and renaming
top_three_styles.reset_index(inplace=True)
top_three_styles.rename(columns={'index': 'beer_style'}, inplace=True)

print("If one enjoys a beer with a {} % preference for aroma and {} % preference for appearance,".format(100*w_aroma, 100*(1-w_aroma)))
print("the top {} beer styles to be tried are:".format(ntop))

for index, row in top_three_styles.iterrows():
    print("#{} - Style: {}, Weighted Average Score: {:.2f}".format(index+1, row.beer_style, row.weighted_avg))





print("\n----------------------------------------------------------------------------------------------------------\n")
