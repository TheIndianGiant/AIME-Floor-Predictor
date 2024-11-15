# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Step 1: Create the Dataset

# Constructing the dataset based on the provided data
data = [
    # Year 2000
    {'Year': 2000, 'Test': 'AMC10A', 'AIME_Floor': 110},
    {'Year': 2000, 'Test': 'AMC10B', 'AIME_Floor': 110},
    {'Year': 2000, 'Test': 'AMC12A', 'AIME_Floor': 92},
    {'Year': 2000, 'Test': 'AMC12B', 'AIME_Floor': 92},
    
    # Year 2001
    {'Year': 2001, 'Test': 'AMC10A', 'AIME_Floor': 116},
    {'Year': 2001, 'Test': 'AMC10B', 'AIME_Floor': 116},
    {'Year': 2001, 'Test': 'AMC12A', 'AIME_Floor': 84},
    {'Year': 2001, 'Test': 'AMC12B', 'AIME_Floor': 84},
    
    # Year 2002
    {'Year': 2002, 'Test': 'AMC10A', 'AIME_Floor': 115},
    {'Year': 2002, 'Test': 'AMC10B', 'AIME_Floor': 118},
    {'Year': 2002, 'Test': 'AMC12A', 'AIME_Floor': 100},
    {'Year': 2002, 'Test': 'AMC12B', 'AIME_Floor': 100},
    
    # Year 2003
    {'Year': 2003, 'Test': 'AMC10A', 'AIME_Floor': 119},
    {'Year': 2003, 'Test': 'AMC10B', 'AIME_Floor': 121},
    {'Year': 2003, 'Test': 'AMC12A', 'AIME_Floor': 100},
    {'Year': 2003, 'Test': 'AMC12B', 'AIME_Floor': 100},
    
    # Year 2004
    {'Year': 2004, 'Test': 'AMC10A', 'AIME_Floor': 110},
    {'Year': 2004, 'Test': 'AMC10B', 'AIME_Floor': 115},
    {'Year': 2004, 'Test': 'AMC12A', 'AIME_Floor': 100},
    {'Year': 2004, 'Test': 'AMC12B', 'AIME_Floor': 100},
    
    # Year 2005
    {'Year': 2005, 'Test': 'AMC10A', 'AIME_Floor': 120},
    {'Year': 2005, 'Test': 'AMC10B', 'AIME_Floor': 120},
    {'Year': 2005, 'Test': 'AMC12A', 'AIME_Floor': 100},
    {'Year': 2005, 'Test': 'AMC12B', 'AIME_Floor': 100},
    
    # Year 2006
    {'Year': 2006, 'Test': 'AMC10A', 'AIME_Floor': 120},
    {'Year': 2006, 'Test': 'AMC10B', 'AIME_Floor': 120},
    {'Year': 2006, 'Test': 'AMC12A', 'AIME_Floor': 100},
    {'Year': 2006, 'Test': 'AMC12B', 'AIME_Floor': 100},
    
    # Year 2007
    {'Year': 2007, 'Test': 'AMC10A', 'AIME_Floor': 117},
    {'Year': 2007, 'Test': 'AMC10B', 'AIME_Floor': 115.5},
    {'Year': 2007, 'Test': 'AMC12A', 'AIME_Floor': 97.8},
    {'Year': 2007, 'Test': 'AMC12B', 'AIME_Floor': 100},
    
    # Year 2008
    {'Year': 2008, 'Test': 'AMC10A', 'AIME_Floor': 117},
    {'Year': 2008, 'Test': 'AMC10B', 'AIME_Floor': 120},
    {'Year': 2008, 'Test': 'AMC12A', 'AIME_Floor': 97.5},
    {'Year': 2008, 'Test': 'AMC12B', 'AIME_Floor': 97.5},
    
    # Year 2009
    {'Year': 2009, 'Test': 'AMC10A', 'AIME_Floor': 120},
    {'Year': 2009, 'Test': 'AMC10B', 'AIME_Floor': 120},
    {'Year': 2009, 'Test': 'AMC12A', 'AIME_Floor': 97.5},
    {'Year': 2009, 'Test': 'AMC12B', 'AIME_Floor': 100},
    
    # Year 2010
    {'Year': 2010, 'Test': 'AMC10A', 'AIME_Floor': 118.5},
    {'Year': 2010, 'Test': 'AMC10B', 'AIME_Floor': 118.5},
    {'Year': 2010, 'Test': 'AMC12A', 'AIME_Floor': 88.5},
    {'Year': 2010, 'Test': 'AMC12B', 'AIME_Floor': 88.5},
    
    # Year 2011
    {'Year': 2011, 'Test': 'AMC10A', 'AIME_Floor': 117},
    {'Year': 2011, 'Test': 'AMC10B', 'AIME_Floor': 117},
    {'Year': 2011, 'Test': 'AMC12A', 'AIME_Floor': 93},
    {'Year': 2011, 'Test': 'AMC12B', 'AIME_Floor': 97.5},
    
    # Year 2012
    {'Year': 2012, 'Test': 'AMC10A', 'AIME_Floor': 115.5},
    {'Year': 2012, 'Test': 'AMC10B', 'AIME_Floor': 120},
    {'Year': 2012, 'Test': 'AMC12A', 'AIME_Floor': 94.5},
    {'Year': 2012, 'Test': 'AMC12B', 'AIME_Floor': 99},
    
    # Year 2013
    {'Year': 2013, 'Test': 'AMC10A', 'AIME_Floor': 108},
    {'Year': 2013, 'Test': 'AMC10B', 'AIME_Floor': 120},
    {'Year': 2013, 'Test': 'AMC12A', 'AIME_Floor': 88.5},
    {'Year': 2013, 'Test': 'AMC12B', 'AIME_Floor': 93},
    
    # Year 2014
    {'Year': 2014, 'Test': 'AMC10A', 'AIME_Floor': 120},
    {'Year': 2014, 'Test': 'AMC10B', 'AIME_Floor': 120},
    {'Year': 2014, 'Test': 'AMC12A', 'AIME_Floor': 93},
    {'Year': 2014, 'Test': 'AMC12B', 'AIME_Floor': 100.5},
    
    # Year 2015
    {'Year': 2015, 'Test': 'AMC10A', 'AIME_Floor': 106.5},
    {'Year': 2015, 'Test': 'AMC10B', 'AIME_Floor': 120},
    {'Year': 2015, 'Test': 'AMC12A', 'AIME_Floor': 99},
    {'Year': 2015, 'Test': 'AMC12B', 'AIME_Floor': 100.5},
    
    # Year 2016
    {'Year': 2016, 'Test': 'AMC10A', 'AIME_Floor': 110},
    {'Year': 2016, 'Test': 'AMC10B', 'AIME_Floor': 110},
    {'Year': 2016, 'Test': 'AMC12A', 'AIME_Floor': 93},
    {'Year': 2016, 'Test': 'AMC12B', 'AIME_Floor': 100.5},
    
    # Year 2017
    {'Year': 2017, 'Test': 'AMC10A', 'AIME_Floor': 112.5},
    {'Year': 2017, 'Test': 'AMC10B', 'AIME_Floor': 120},
    {'Year': 2017, 'Test': 'AMC12A', 'AIME_Floor': 96},
    {'Year': 2017, 'Test': 'AMC12B', 'AIME_Floor': 100},
    
    # Year 2018
    {'Year': 2018, 'Test': 'AMC10A', 'AIME_Floor': 111},
    {'Year': 2018, 'Test': 'AMC10B', 'AIME_Floor': 108},
    {'Year': 2018, 'Test': 'AMC12A', 'AIME_Floor': 93},
    {'Year': 2018, 'Test': 'AMC12B', 'AIME_Floor': 99},
    
    # Year 2019
    {'Year': 2019, 'Test': 'AMC10A', 'AIME_Floor': 103.5},
    {'Year': 2019, 'Test': 'AMC10B', 'AIME_Floor': 108},
    {'Year': 2019, 'Test': 'AMC12A', 'AIME_Floor': 84},
    {'Year': 2019, 'Test': 'AMC12B', 'AIME_Floor': 94.5},
    
    # Year 2020
    {'Year': 2020, 'Test': 'AMC10A', 'AIME_Floor': 103.5},
    {'Year': 2020, 'Test': 'AMC10B', 'AIME_Floor': 102},
    {'Year': 2020, 'Test': 'AMC12A', 'AIME_Floor': 87},
    {'Year': 2020, 'Test': 'AMC12B', 'AIME_Floor': 87},
    
    # Year 2021 Spring
    {'Year': 2021, 'Test': 'AMC10A', 'AIME_Floor': 103.5},
    {'Year': 2021, 'Test': 'AMC10B', 'AIME_Floor': 102},
    {'Year': 2021, 'Test': 'AMC12A', 'AIME_Floor': 93},
    {'Year': 2021, 'Test': 'AMC12B', 'AIME_Floor': 91.5},
    
    # Year 2021 Fall
    {'Year': 2021, 'Test': 'AMC10A', 'AIME_Floor': 96},
    {'Year': 2021, 'Test': 'AMC10B', 'AIME_Floor': 96},
    {'Year': 2021, 'Test': 'AMC12A', 'AIME_Floor': 91.5},
    {'Year': 2021, 'Test': 'AMC12B', 'AIME_Floor': 84},
    
    # Year 2022
    {'Year': 2022, 'Test': 'AMC10A', 'AIME_Floor': 93},
    {'Year': 2022, 'Test': 'AMC10B', 'AIME_Floor': 94.5},
    {'Year': 2022, 'Test': 'AMC12A', 'AIME_Floor': 85.5},
    {'Year': 2022, 'Test': 'AMC12B', 'AIME_Floor': 81},
    
    # Year 2023
    {'Year': 2023, 'Test': 'AMC10A', 'AIME_Floor': 103.5},
    {'Year': 2023, 'Test': 'AMC10B', 'AIME_Floor': 105},
    {'Year': 2023, 'Test': 'AMC12A', 'AIME_Floor': 85.5},
    {'Year': 2023, 'Test': 'AMC12B', 'AIME_Floor': 88.5},
]

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)

# Data Cleaning: Handle multiple entries for the same Year and Test by averaging AIME_Floor
df = df.groupby(['Year', 'Test'], as_index=False).mean()

# Step 2: Data Preprocessing and Model Building

# Define features and target
X = df[['Year', 'Test']]
y = df['AIME_Floor']

# One-Hot Encoding for 'Test'
preprocessor = ColumnTransformer(
    transformers=[
        ('test', OneHotEncoder(), ['Test'])
    ],
    remainder='passthrough'  # Keep 'Year' as is
)

# Create a Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit the model
model_pipeline.fit(X, y)

# Step 3: Rounding Function

def round_to_nearest_1_5(x):
    """
    Rounds a number to the nearest multiple of 1.5.

    Parameters:
    - x (float): The number to round.

    Returns:
    - float: The number rounded to the nearest 1.5 multiple.
    """
    return round(x / 1.5) * 1.5

# Step 4: Prediction Function

def predict_aime_floor(year, test):
    """
    Predict the AIME floor based on the year and test type.

    Parameters:
    - year (int or float): The year for prediction.
    - test (str): The AMC test type (AMC10A, AMC10B, AMC12A, AMC12B).

    Returns:
    - float: The predicted AIME floor rounded to the nearest 1.5.
    """
    # Validate test input
    valid_tests = ['AMC10A', 'AMC10B', 'AMC12A', 'AMC12B']
    if test not in valid_tests:
        print(f"Invalid test name. Please choose from {', '.join(valid_tests)}.")
        return None
    
    # Create a DataFrame for the input
    input_data = pd.DataFrame({'Year': [year], 'Test': [test]})
    
    # Predict using the pipeline
    predicted_floor = model_pipeline.predict(input_data)[0]
    
    # Round to the nearest multiple of 1.5
    rounded_floor = round_to_nearest_1_5(predicted_floor)
    
    return rounded_floor

# Step 5: User Interface

def main():
    print("\n--- AIME Floor Predictor ---")
    while True:
        try:
            input_year = float(input("Enter the year you want to predict the AIME floor for (e.g., 2024): "))
        except ValueError:
            print("Invalid input. Please enter a numerical year.")
            continue

        tests = ['AMC10A', 'AMC10B', 'AMC12A', 'AMC12B']
        print("\nAvailable AMC Tests:")
        for i, test in enumerate(tests, 1):
            print(f"{i}. {test}")
        
        try:
            test_choice = int(input("Select the test by entering the corresponding number (1-4): "))
            if test_choice < 1 or test_choice > len(tests):
                print("Invalid choice. Please select a number between 1 and 4.")
                continue
            selected_test = tests[test_choice - 1]
        except ValueError:
            print("Invalid input. Please enter a number corresponding to the test.")
            continue
        
        prediction = predict_aime_floor(input_year, selected_test)
        if prediction is not None:
            print(f"\nPredicted AIME floor for {selected_test} in {input_year}: {prediction}")
        
        # Ask if the user wants to make another prediction
        another = input("\nDo you want to make another prediction? (y/n): ").strip().lower()
        if another != 'y':
            print("Thank you for using the AIME Floor Predictor!")
            break

if __name__ == "__main__":
    main()