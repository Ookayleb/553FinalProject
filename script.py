import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# pd.set_option("display.max_rows", None, "display.max_columns", None)

print("Classifiers in random forrest: ", end="")
numClass = int(input())
print("Test/train split: 0.", end="")
traintestSplit = float("0." + input())

print((traintestSplit))


# 1 - Negative
# 2 - Positive

df = pd.read_csv("Train.csv")

#Remove rows that have 0 or 3 in the Sentiment column, we are not interested in these
df = df[df['Sentiment'] > 0]
df = df[df['Sentiment'] < 3]

print(df)

#Split our dataframe into two, one containing negative sentiment and one containing positive.
negDF = df[df['Sentiment'] == 1]
posDF = df[df['Sentiment'] == 2]


sizePos	= len(posDF)
sizeNeg	= len(negDF)


labels = 'Positive', 'Negative'
sizes = [sizePos, sizeNeg]

fig1, ax1 = plt.subplots()
ax1.pie(sizes,  labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')

plt.savefig("graph.png", dpi=100)
#Graph our findings in a file called graph.png







#Create a list of all sentances
sentances = df["Product_Description"].values

#Create a TF-IDF model
vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
sentancesTFIDF = vectorizer.fit_transform(sentances).toarray()


classLabel = df.loc[:, "Sentiment"].values

#Split our data into test and training data sets
xTrain, xTest, yTrain, yTest = train_test_split(sentancesTFIDF, classLabel, test_size=traintestSplit, random_state=0)
text_classifier = RandomForestClassifier(n_estimators=numClass, random_state=0)
text_classifier.fit(xTrain, yTrain)

# print("debg")
# print(xTest)
# # print(yTrain)

# predictions = text_classifier.predict(xTest)

# print(predictions)

print(confusion_matrix(yTest,predictions))
print(classification_report(yTest,predictions))
print(accuracy_score(yTest, predictions))
