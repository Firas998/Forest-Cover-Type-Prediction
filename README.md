# Introduction

In this project, we will be using machine learning techniques to
classify forest cover types based on a set of features such as
elevation, soil type, and wilderness area. The dataset for this
competition consists of cartographic variables and wilderness area
boundaries for an area of 464,000 hectares in Colorado. The goal is to
predict the cover type for each 30 x 30 meter patch of land in the test
set.

#### 

There are seven cover types in the dataset: Spruce/Fir, Lodgepole Pine,
Ponderosa Pine, Cottonwood/Willow, Aspen, Douglas-fir, and Krummholz.
Accurately predicting the cover type for a given patch of land can have
important practical applications, such as aiding in conservation efforts
and resource management.

#### 

We will be using a variety of machine learning algorithms to build and
compare models for this classification task. We will also be exploring
the different features in the dataset and how they influence the
prediction of the cover type.

# Dataset

The dataset for this problem includes twelve features among which ten
are numerical and two are catrgorical.

The numerical values are presented in the following table:\

::: center
![Numerical features
description](numerical_data_explained.png){width="40%"}
:::

The two categorical features are the wilderness area (4 categories) and
the soil type (40 categories). Since the data is already one-hot
encoded, we have 44 binary columns in our dataset representing the
categorical data.\

## Data Analysis

We first start by checking for null values or duplicates in our dataset.
None are found.\
We then check that the dataset is balanced.

::: center
![Number of samples per cover type](uniform.png){width="30%"}
:::

After that we divide the dataset between numerical and categorical
features.

### Numerical features

#### Correlations

We then start off by looking at the covariance matrix, to see how
correlated each two numerical features are :\

#### Interpretation of the correlation matrix

What is most striking are the correlations between the different
\"Hillshade\" variables as well as the distances ( to Hydrology, to
Roadways, to FirePoints). We therefore need to study the correlation
with the Cover Type to see which features we need to retain and which we
could discard.\

::: center
![Correlation matrix between numerical features](corr.png){width="90%"}
:::

In order to do so we performed one-hot encoding on the Cover Type to get
7 new binary variables. We then computed the correlations of the
features with each of the new 7 target variables. We present here the
correlations for the binary variable associated with Cover Type 1.
According to those results, \"Hillshade_9am\" and \"Aspect\" have to be
discarded because they have low correlations with the target. All other
features, including the distance-based ones, were retained because they
had strong correlations with the target variable.

::: center
![Correlation between features and target](photo2.jpg){width="50%"}
:::

#### Transformation of features using skewness

Additionally, we look at the data distribution of all numerical values
to study the skewness :\

::: center
![Density graph of the numerical features](Skewness.png){width="50%"}
:::

We apply the logarithmic function (log(x+1)) for right-skewed data and
the square function for left-skewed data.\

#### 

We then use the method above of looking at the correlations of each
binary variable associated with the Cover Type to find which features we
will retain: either the original or the skewness-transformed feature.\

## Data Preprocessing

#### Dropping Aspect and Hillshade_9am

As discussed in the analysis above, the aspect and hillshade_9am
features are not correlated with the cover type, thus we could drop them
them without losing any information.

#### Soil Type preprocessing

This preprocessing was applied in the case of random trees and gradient
boosting models. Since decision trees and random forests, are able to
handle categorical features natively and do not require them to be
encoded, one-hot encoding of the categorical features may not be
necessary and could even harm the performance of the model by adding
unnecessary complexity. To make things much simpler, we combine all the
one-hot-encoded soil types into a single categorical feature that can
take values from 1 to 40 for these models.\

#### Feature Engineering:

#### $\bullet$ Interaction features

\
A first type of features that came to mind are the sums and differences
of all pairs of distance based features. These features can help capture
interactions between different distances:\

-   Sum and absolute difference of horizontal distance to hydrology and
    horizontal distance to roadways

-   Sum and absolute difference of horizontal distance to hydrology and
    horizontal distance to firepoints

-   Sum and absolute difference of horizontal distance to roadways and
    horizontal distance to firepoints

-   Sum and absolute difference of elevation to horizontal and vertical
    hydrology

#### $\bullet$ Transformed features

\
We also take interest in transforming the original features based on
their skewness. We apply classical transformations : if a feature is
right skewed we apply the logarithm function to the feature, and if the
data is left skewed we apply the square function to the feature.\

#### Feature Selection

After creating all those features we proceed to select the ones that
have the highest correlation rate with the cover type as proceeded in
the analysis. Thus we created a list containing all the features we want
to take into consideration, and we dropped the rest.

::: center
![Correlation between the features and the
target](corr2.jpg){width="35%"}
:::

::: center
![List of the selected features](chosen.png){width="90%"}
:::

#### Scaling numerical features

This was applied in the case of linear models (SVM, logistic
regression), and neural networks that are sensitive to the scale of the
input features. Scaling the features helped improve the performance of
the models by making the optimization process more efficient and by
reducing the risk of numerical instability.

# Models

## Extra Trees Classifier

Extra trees classifier, also known as extremely randomized trees
classifier, is a type of ensemble model that is built using decision
trees. It is a variant of the random forest model, which is an ensemble
of decision trees that are trained using bootstrapped samples of the
training data and random subsets of the features.

Tree-based classifiers were particularly well-suited for this problem
for two main reasons:\

-   First, these algorithms can handle multi-class classification tasks
    directly, without the need for techniques like one-versus-one or
    one-versus-all. This makes the training process more efficient and
    straightforward. This in turn makes them efficient and accurate.

-   Second, tree-based classifiers are effective at handling categorical
    variables without the need for one-hot encoding, which can simplify
    the model and reduce the need for dummy variables. This was
    especially beneficial in this dataset, as the soil type feature was
    important in determining the output and had 40 different values.

\
The model produced good results by outperforming all the other models we
tried. Here are the performance metrics that we got from the
cross-validation approach:\

::: center
![ETC model performance metrics](tableau.png){width="35%"}
:::

As for the confusion matrix we got the following diagram:

::: center
![Confusion Matrix for the ETC model](Confusion matrix.png){width="70%"}
:::

In this confusion we can see that there is a high rate of confusion
between cover type 1 and cover type 2.

## Other models

There are many different machine learning models that you could consider
for a classification problem like predicting forest cover type. In our
case we tried random forests, gradient boosting, KNN, SVM, logistic
regression and neural networks. We got the following results (which you
could find at the end of the notebook) :

::: center
![Comparison between the different models we
tried](results.png){width="100%"}
:::

# Conclusion

In this project, we explored the Kaggle competition for predicting
forest cover type in an area of Colorado. We used a variety of machine
learning algorithms to build and compare models for this classification
task, and explored the different features in the dataset to understand
how they influence the prediction of the cover type.

#### 

Overall, we found that the extra trees classifier performed the best on
this dataset, with an accuracy of over 90% on the test set. A possible
extension to our work would be fine-tuning the hyperparameters of the
classifier using GridSearchCV for example. The grid search algorithm
will then train and evaluate a model for each combination of
hyperparameter values, and select the combination that performs the
best.

#### 

In conclusion, this project demonstrated the usefulness of machine
learning in predicting forest cover type and highlighted the importance
of carefully understanding the role of the different features.
