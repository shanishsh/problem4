import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


# Load the World Indicators data from "World Indicators.csv"


# Your code starts after this line

# Load the dataset
file_path = "hl.csv"
df = pd.read_csv(file_path)

# Preview the first few rows
print(df.head())
# Your code ends before this line
#----------------------------------------------------------------------------



# Create a linear model between year and population in the US

# Your code starts after this line
# Load dataset
df = pd.read_csv("hl.csv")

# Filter only United States rows
us = df[df["Country/Region"] == "United States"]

# Select predictor and target variable
X = us[["Year"]]
y = us["Population Total"]

# Create and train linear regression model
model = LinearRegression()
model.fit(X, y)

# Output coefficients
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
# Your code ends before this line
#----------------------------------------------------------------------------





# Predict the expected population in the US in 2015

# Your code starts after this line

# Load dataset
df = pd.read_csv("hl.csv")

# Filter rows for the United States
us = df[df["Country/Region"] == "United States"]

# Select features and target
X = us[["Year"]]
y = us["Population Total"]

# Train linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict population for 2015
prediction_2015 = model.predict([[2015]])[0]

print("Predicted US Population in 2015:", prediction_2015)
# Your code ends before this line
#-------------------------------------------------------------------------------






# For the data from Europe
# Create a linear model between Life Expectancy Female and the significant predictors among
#  Birth Rate
#  GDP
#  Health Exp % GDP
#  Infant Mortality Rate
#  Life Expectancy Male

# Summarize your model (only the final one)

# Hint: if you hit an issue with NaNs in the values consider using this: missing='drop'

# Your code starts after this line
# Load dataset
df = pd.read_csv("hl.csv")

# Filter only European countries
europe = df[df["Region"] == "Europe"]

# Select relevant columns
cols = [
    "Life Expectancy Female", 
    "Birth Rate", 
    "GDP", 
    "Health Exp % GDP",
    "Infant Mortality Rate",
    "Life Expectancy Male"
]

# Keep only rows without missing values
europe = europe[cols].dropna()

# Define predictor matrix (X) and dependent variable (y)
X = europe[[
    "Birth Rate", 
    "GDP", 
    "Health Exp % GDP", 
    "Infant Mortality Rate", 
    "Life Expectancy Male"
]]
y = europe["Life Expectancy Female"]

# Add constant term for regression
X = sm.add_constant(X)

# Build linear regression model
model = sm.OLS(y, X).fit()

# Print final model summary
print(model.summary())

# Your code ends before this line
#-----------------------------------------------------------------------------------------------







# Predict the Expected Life Expectancy Female of a country with this characteristics
#  Birth Rate = 3%
#  GDP = 1 billion
#  Health Exp % GDP = 4%
#  Infant Mortality Rate = 5%
#  Life Expectancy Male = 80
# Round the prediction to two decimal points

# Your code starts after this line
# Load dataset
df = pd.read_csv("/mnt/data/World Indicators.csv")

# Filter only European data
europe = df[df["Region"] == "Europe"]

# Select relevant model columns
cols = [
    "Life Expectancy Female", 
    "Birth Rate", 
    "GDP", 
    "Health Exp % GDP",
    "Infant Mortality Rate",
    "Life Expectancy Male"
]

# Drop rows with missing data
europe = europe[cols].dropna()

# Define X and y
X = europe[[
    "Birth Rate", 
    "GDP", 
    "Health Exp % GDP", 
    "Infant Mortality Rate", 
    "Life Expectancy Male"
]]
y = europe["Life Expectancy Female"]

# Add constant term
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Data for prediction
new_country = pd.DataFrame({
    "const": [1],
    "Birth Rate": [3],
    "GDP": [1_000_000_000],  # 1 billion
    "Health Exp % GDP": [4],
    "Infant Mortality Rate": [5],
    "Life Expectancy Male": [80]
})

# Make prediction
predicted_life_exp = model.predict(new_country)[0]

# Output rounded result
print("Predicted Life Expectancy Female:", round(predicted_life_exp, 2))

# Your code ends before this line

