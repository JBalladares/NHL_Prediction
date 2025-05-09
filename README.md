# NHL_Prediction
Using machine learning to predict which active NHL players might break the all-time goal-scoring record.
---

## Project Summary

This project uses a **Random Forest Regression** model to predict the next NHL All Time Goal Scorer. With the current record being broken in 2025, it's interesting to take a closer look at the achievement and if there is anyone close on breaking it..again!

### Key Questions:
- Who are the most likely active players to challenge the record?
- How many seasons would they need to do it?
- What year would theybreak it?
---

## Methodology

- **Data Source**: Public NHL goal-scoring stats from 1980 to 2025
- **Model Used**: `RandomForestRegressor` from scikit-learn
- **Features Used**:
  - `Games Played (GP)`
  - `Time on Ice per Game (TOI/GP)`
  - `Goals per Game`
  - `Shooting Percentage`

- **Target**: Projected final career goals
---

## Key Findings
> TLDR:
- No currently active players in the Top 500 are on pace to break the 897 goal record.
- Players like Sidney Crosby and Steven Stamkos would need to play into their mid 40s at AND maintain their scoring pace to at least tie the record.
- The next realistic contender is likely to be a younger player but we may not see the record broken for close to 10 years.

There are a few players that can tie or even break the record but they would need to play for additional 7 season and/or into their mid 40s. With hockey being a fast and agressive sport, it's hard to see a player maintain their physicality and status well beyond 40. Not to say it's impossible as we know many things could happen but it's highly unlikely. So based on the current active players we may not see this record broken for at least 10 years...maybe.


---
