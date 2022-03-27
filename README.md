# Titanic-Classifier
![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)
![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)<br>
![](https://komarev.com/ghpvc/?username=Titanic-Classifier&color=green&style=for-the-badge&label=PAGE+VIEWS)

## Main Goal
A binary classifier to predict whether a person would have survived, or not, to the Titanic’s disaster.

## Dataset Description
Training set (710 samples), and testing set (177 samples).
<br>
Each dataset row represents a specific passenger’s information (predictors/features), such as:
ticket class; gender; age; number of siblings and spouses aboard; number of parents and children
aboard; passenger fare.
<br>
Finally, is also known whether the person survived or not (target variable).

## Prediction
Are you interested on knowing which would have had your probability of surviving? 
<br>
Change the `my_info` values into the [analysis.py](./scripts/analysis.py) file, then run the script.

## Overall Performances: 
* *Training*: 80.14% accuracy
* *Testing*: 78.53% accuracy

*** 
### Scatterplot showing the distribution of the two classes in the plane defined by the two most influential features
<img width="500" alt="Class&GenderVsSurvived" src="https://user-images.githubusercontent.com/80333091/160282366-2b97f037-6adc-4244-9fe3-9d38b51b43c9.png">
As shown in the upper scatterplot, the females are more likely to survive than males, while if the ticket class is low (1st class) the probability of surviving increases.
<br>
The feature which discriminates more the probability of surviving is the gender.
