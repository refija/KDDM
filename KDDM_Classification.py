import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

data = pd.read_csv('Olympics.csv')
print(data.head())

missing_values = data.isnull().sum()
print(missing_values)

imputer = SimpleImputer(strategy='most_frequent')
data['Country'] = imputer.fit_transform(data[['Country']])

missing_values = data.isnull().sum()
print(missing_values)

# Step 5: Feature Engineering
# Create new features or extract relevant information from existing ones

# Number of medals won by each country in different years
medals_by_country_year = data.groupby(['Country', 'Year'])['Medal'].count()
print("Medals by Country and Year:")
#print(medals_by_country_year.to_string(index=True))
# Number of gold, silver, and bronze medals won by each country
medals_by_country = data.groupby('Country')['Medal'].value_counts().unstack(fill_value=0)

# Print the medals by country
print("Medals by Country:")
print(medals_by_country)

# Number of medals won by each country in different sports
medals_by_country_sport = data.groupby(['Country', 'Sport'])['Medal'].count()
print("Medals by Country and Sport:")
#print(medals_by_country_sport.to_string(index=True))   #################################################

#Countries won olumpic medals in total
total_medals_won = data.groupby('Country').size()
print("Total Number of Times Each Country won a medal the Olympic Games:")
#print(total_medals_won.to_string(index=True))          #################################################

# Number of times each country attended the Olympic Games
times_attended = data.groupby('Country')['Year'].nunique()
print("Number of Times Each Country Attended the Olympic Games:")
print(times_attended.to_string(index=True))

# Group the data by 'Country', 'Year', and 'Sport' columns and count the number of medals in each sport for each year
medals_by_country_year_sport = data.groupby(['Country', 'Year', 'Sport'])['Medal'].count()
print("Number of Medals Won by Each Country in Each Sport for Each Year:")
print(medals_by_country_year_sport.to_string(index=True))

# Group the data by Country, Sport, and gender columns
sports_by_type = data.groupby(['Country', 'Sport', 'Gender']).size()
print("Sports Grouped by Country, Sport, and Discipline:")
print(sports_by_type)

# Contingency table of gender and sport
gender_sport_table = pd.crosstab(data['Gender'], data['Sport'])
plt.figure(figsize=(20, 6))
sns.heatmap(gender_sport_table, cmap='YlGnBu', annot=True, fmt='d')
plt.xlabel('Sport')
plt.ylabel('Gender')
plt.title('Heatmap of Gender vs. Sport')
plt.show()

# Splitting the Dataset
features = [ 'Year', 'Sport', 'Discipline', 'Event', 'Gender', 'Country_Code', 'Country']
target = 'Medal'
X = data[features]
y = data[target]
#X = data.drop('Medal', axis=1)
#y = data['Medal']
# Encode categorical variables
encoder = LabelEncoder()
X_encoded = X.copy()
for feature in X.columns:
    if X[feature].dtype == 'object':
        X_encoded[feature] = encoder.fit_transform(X[feature].astype(str))

X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Model Selection and Training
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print('Validation Accuracy:', accuracy)

# y_test_pred = model.predict(X_val)
# accuracy_test = accuracy_score(y_val, y_test_pred)
# print('Test Accuracy:', accuracy_test)

# Create a Random Forest classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)


# Create an SVM classifier
model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)


# Convert string labels to numeric labels
label_mapping = {'Bronze': 0, 'Silver': 1, 'Gold': 2}
y_train_numeric = y_train.map(label_mapping)
y_test_numeric = y_val.map(label_mapping)

# Create the Gradient Boosting classifier
model = xgb.XGBClassifier()
model.fit(X_train, y_train_numeric)
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_test_numeric, y_pred)
print("Accuracy:", accuracy)
# Predictions and Interpretation
# Make predictions using the trained model on new data or interpret the model's predictions


# Create a new column representing the color of the medal
def classify_medal_color(medal):
    if medal == 'Gold':
        return 'Gold'
    elif medal == 'Silver':
        return 'Silver'
    elif medal == 'Bronze':
        return 'Bronze'
    else:
        return 'Unknown'

data['Medal Color'] = data['Medal'].apply(classify_medal_color)
medals_by_color = data.groupby('Medal Color').size()
print(medals_by_color)
