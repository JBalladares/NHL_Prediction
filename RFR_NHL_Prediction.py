import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

#Load the data from the file
data = pd.read_csv("NHL_Goals.csv")
#See the first few rows
#print(data.head())
#See column names and data types
#print(data.info())
#check for any missing values
#print(data.isnull().sum())
#Convert columns with commas to integers
columns_to_fix = ["GP", "A", "P", "PIM", "EVP", "S"]
for col in columns_to_fix:
    data[col] = data[col].str.replace(",", "").astype(int)
#Convert "TOI/GP" from mm:ss format to float
def convert_toi(toi_str):
    if toi_str == "--":
        return np.nan
    minutes, seconds = map(int, toi_str.split(":"))
    return minutes + seconds / 60

data["TOI/GP"] = data["TOI/GP"].apply(convert_toi)

#New column to see Goals per Game
data["Goals_per_Game"] = data["G"] / data["GP"]
data["Shots"] = data["S"]
data["Shot_Percentage"] = data["S%"]

#Define features
features = ["GP", "TOI/GP", "Goals_per_Game", "Shot_Percentage"]

#Drop rows with missing or invalid data in key features
data_model = data.dropna(subset=features + ["G"])
#Define input features (X) and (y)
X = data_model[features]
y = data_model["G"]

#split data for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#train the random forest regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#predict on the test set
y_pred = model.predict(X_test)
#evaluate performance
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

#print("R2 score: ", r2)
#print("RMSE: ", rmse)

#plot actual vs. predicted goals
#plt.figure(figsize=(6, 6))
#plt.scatter(y_test, y_pred, alpha=0.7, edgecolors="k")
#plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
#plt.xlabel("Actual Career Goals")
#plt.ylabel("Predicted Career Goals")
#plt.title("Actual vs Predicted Career Goals")
#plt.grid(True)
#plt.tight_layout()
#plt.show()

#Plot feature importance

feat_df = pd.DataFrame({
    "Feature": list(X.columns),
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)
#print(feat_df)

#plt.figure(figsize=(8, 5))
#sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis")
#plt.title("Feature Importance (Random Forest)")
#plt.xlabel("Importance")
#plt.ylabel("Feature")
#plt.tight_layout()
#plt.show()

#Filter players with fewer than 500 goals
#only players who haven't broken the record
active_players = data_model[data_model["G"] < 897].copy()
#Remove retired players who are no longer active
retired_players = [
    "Patrick Marleau",
    "Marian Hossa",
    "Joe Pavelski",
    "Eric Staal"
]
active_players = active_players[~active_players["Player"].isin(retired_players)]

active_X = active_players[features] #Same features used during training
active_players["Predicted_G"] = model.predict(active_X)

#sort by highest predicted totals
top_predictions = active_players.sort_values(by="Predicted_G", ascending=False)
#print(top_predictions[["Player", "G", "Predicted_G"]].head(10))

#Final visualization
plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_predictions.head(10),
    x="Predicted_G",
    y="Player",
    color="skyblue"
)
plt.axvline(897, color='red', linestyle="--", label='Current Record')
plt.title("Top 10 Active Players by Predicted Career Goals")
plt.xlabel("Predicted Career Goal Total")
plt.ylabel("Player")
plt.legend()
plt.tight_layout()
#plt.show()

# Calculate when (in Years) it will take to break
#Need to calculate the remaining goals needed to beat current record
RECORD_GOALS = 897
active_players["Goals_Remaining"] = RECORD_GOALS - active_players["G"]

#Need goals per season calculated
active_players["Est_Goals_Per_Season"] = active_players["Goals_per_Game"] * 82

#Estimate Years Needed
active_players["Seasons_To_Break_Record"] = (active_players["Goals_Remaining"] / active_players["Est_Goals_Per_Season"]).astype(int)

#Estimate the Year they will break it
active_players["Est_Break_Year"] = 2025 + active_players["Seasons_To_Break_Record"].astype(int)

#Final Columns to display
top_predictions = active_players.sort_values(by="Predicted_G", ascending=False)
print(top_predictions.head(5)[["Player", "G", "Predicted_G", "Goals_Remaining", "Est_Goals_Per_Season", "Seasons_To_Break_Record", "Est_Break_Year"]].head(10))

#Okay but now, let's see who actually likely to break it based on age.
#Let's get the birth years of the top players
birth_years = {
"Sidney Crosby" : 1987,
"Steven Stamkos" : 1990,
"Evgeni Malkin" : 1986,
"John Tavares" : 1990,
"Patrick Kane" : 1988,
"Corey Perry" : 1985
}

#Now we need to add column to estimate age at time of breaking the record
def get_age_at_break(player, break_year):
    birth_year = birth_years.get(player)
    if birth_year:
        return break_year - birth_year
    else:
        return np.nan #skip if we can't get the year (for future players)
active_players["Age_At_Break"] = active_players.apply(
    lambda row: get_age_at_break(row["Player"], row["Est_Break_Year"]), axis=1
)

#Filter out player over 42 (close to retirement age)
realistic_candidates = active_players[(active_players["Age_At_Break"] <= 42) & (active_players["Predicted_G"] >= 800)]
#print(realistic_candidates[["Player", "G", "Predicted_G", "Goals_Remaining", "Est_Goals_Per_Season", "Seasons_To_Break_Record", "Est_Break_Year", "Age_At_Break"]])
