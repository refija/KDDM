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

unique_sports = data['Sport'].unique()
print("Unique sports:", unique_sports)
# Team sports: Basketball, Football, Handball, Hockey, Baseball, Softball, Volleyball
# Individual sports: Archery, Athletics, Weightlifting, Boxing,Canoe / Kayak, Cycling, Equestrian, Fencing, Gymnastics, Judo,
# Modern Pentathlon, Rowing, Sailing, Shooting, Wrestling,Table Tennis, Tennis, Badminton, Taekwondo, Triathlon

##################################################################################################################################

# Number of medals won by each country in different years
medals_by_country_year = data.groupby(['Country', 'Year'])['Medal'].count()
print("Medals by Country and Year:")
#print(medals_by_country_year.to_string(index=True))

# Number of gold, silver, and bronze medals won by each country
medals_by_country = data.groupby('Country')['Medal'].value_counts().unstack(fill_value=0)
print("Medals by Country:")
print(medals_by_country)

# Number of gold, silver, and bronze medals won by each country and sport
medals_by_country_sport = data.groupby(['Country', 'Sport'])['Medal'].value_counts().unstack(fill_value=0)
print("Medals by Country and Sport:")
print(medals_by_country_sport)

# Number of gold, silver, and bronze medals won by each country and sport and gender specific
medals_by_country_sport_gender = data.groupby(['Country', 'Sport', 'Gender'])['Medal'].value_counts().unstack(fill_value=0)
print("Medals by Country and Sport:")
print(medals_by_country_sport_gender)

# Create a function to classify the sports and create a new column 'Sport Type'
def classify_sport(sport):
    individual_sports = ['Archery', 'Athletics', 'Weightlifting', 'Boxing', 'Canoe / Kayak', 'Cycling',
                         'Equestrian', 'Fencing', 'Gymnastics', 'Judo', 'Modern Pentathlon', 'Rowing',
                         'Sailing', 'Shooting', 'Wrestling', 'Table Tennis', 'Tennis', 'Badminton',
                         'Taekwondo', 'Triathlon']
    if sport in individual_sports:
        return 'Individual sports'
    else:
        return 'Team sports'

data['Sport Type'] = data['Sport'].apply(classify_sport)
print(data)

# Group the data by country and sport type
grouped_data = data.groupby(['Country', 'Sport Type'])
medal_counts = grouped_data['Medal'].count()
print(medal_counts)


# Group the data by country, year, and athlete
grouped_data = data.groupby(['Country', 'Year', 'Athlete','Sport', 'Medal'])

# Get the count of medals for each country, year, and athlete
medal_counts = grouped_data.size()

# Print the medal counts by country, year, and athlete
print("Medal counts by country, year, and athlete:")
print(medal_counts)




# Group the data by athlete and country
grouped_data = data.groupby(['Country', 'Athlete'])

# Get the count of medals for each athlete and country
medal_counts = grouped_data['Medal'].count()

# Get the unique types of medals for each athlete and country
medal_types = grouped_data['Medal'].unique()

# Combine the medal counts and types into a new DataFrame
athlete_medals = pd.DataFrame({'Medal Count': medal_counts, 'Medal Types': medal_types})

# Print the DataFrame showing medal counts and types for each athlete and country
print(athlete_medals)
# Filter the athletes who have won more than 1 gold, 1 silver, and 1 bronze medal
filtered_data = data.groupby('Athlete').filter(lambda x: (x['Medal'] == 'Gold').sum() > 1 and
                                                        (x['Medal'] == 'Silver').sum() > 1 and
                                                        (x['Medal'] == 'Bronze').sum() > 1)

# Pivot the filtered data to create a matrix of athletes and medal counts
pivot_data = filtered_data.pivot_table(index='Athlete', columns='Medal', aggfunc='size', fill_value=0)

# Plot the heatmap
sns.heatmap(pivot_data, annot=True, cmap='YlGnBu')
plt.title('Medal Counts for Athletes with >1 Gold, >1 Silver, and >1 Bronze')
plt.xlabel('Medal')
plt.ylabel('Athlete')
plt.show()
# Filter the athletes who have won more than 1 gold, 1 silver, and 1 bronze medal
filtered_data = data.groupby(['Athlete', 'Country']).filter(lambda x: (x['Medal'] == 'Gold').sum() > 1 and
                                                                    (x['Medal'] == 'Silver').sum() > 1 and
                                                                    (x['Medal'] == 'Bronze').sum() > 1)

# Pivot the filtered data to create a matrix of athletes, countries, and medal counts
pivot_data = filtered_data.pivot_table(index='Athlete', columns='Country', values='Medal', aggfunc='count', fill_value=0)

# Plot the heatmap
sns.heatmap(pivot_data, annot=True, cmap='YlGnBu')
plt.title('Medal Counts for Athletes with >1 Gold, >1 Silver, and >1 Bronze')
plt.xlabel('Country')
plt.ylabel('Athlete')
plt.show()
# Filter the athletes who have won more than 1 gold, 1 silver, and 1 bronze medal
filtered_data = data.groupby(['Athlete', 'Country']).filter(lambda x: (x['Medal'] == 'Gold').sum() > 1 and
                                                                    (x['Medal'] == 'Silver').sum() > 1 and
                                                                    (x['Medal'] == 'Bronze').sum() > 1)

# Pivot the filtered data to create a matrix of athletes, sport, country and medal (total)
pivot_data = filtered_data.pivot_table(index=['Athlete', 'Sport'], columns='Country', values='Medal', aggfunc='count', fill_value=0)

# Plot the heatmap
sns.heatmap(pivot_data, annot=True, cmap='YlGnBu')
plt.title('Medal Counts for Athletes with >1 Gold, >1 Silver, and >1 Bronze')
plt.xlabel('Country')
plt.ylabel('Athlete / Sport')
plt.show()
















###############################################################################################################################
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
#print(times_attended.to_string(index=True))

# Group the data by 'Country', 'Year', and 'Sport' columns and count the number of medals in each sport for each year
medals_by_country_year_sport = data.groupby(['Country', 'Year', 'Sport'])['Medal'].count()
print("Number of Medals Won by Each Country in Each Sport for Each Year:")
#print(medals_by_country_year_sport.to_string(index=True))

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