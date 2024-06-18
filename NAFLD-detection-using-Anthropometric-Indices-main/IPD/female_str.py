import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('FEMALE_data.csv')

# Assuming 'Diagnosis' is the target variable
y = df['Diagnosis']  # Target variable

# Create a sample dataset from your original dataset (excluding the target variable)
sample_data = df.drop('Diagnosis', axis=1).copy()

# Identify non-numeric columns
non_numeric_cols = sample_data.select_dtypes(exclude=['number']).columns

# Handle missing values for numeric columns
numeric_cols = sample_data.columns.difference(non_numeric_cols)
imputer_numeric = SimpleImputer(strategy='mean')
sample_data[numeric_cols] = imputer_numeric.fit_transform(sample_data[numeric_cols])

# Encode categorical variables if needed
sample_data = pd.get_dummies(sample_data, columns=non_numeric_cols, drop_first=True)

# Fit the scaler on the sample data
scaler = StandardScaler()
scaler.fit(sample_data)

# Train the Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(sample_data, y)

# Function to take user input and make predictions
def predict_nafld_presence(model, input_features, scaler):
    # Convert input features to a DataFrame
    input_df = pd.DataFrame([input_features])

    # Identify non-numeric columns
    non_numeric_cols = input_df.select_dtypes(exclude=['number']).columns

    # Handle missing values for numeric columns
    numeric_cols = input_df.columns.difference(non_numeric_cols)
    imputer_numeric = SimpleImputer(strategy='mean')
    input_df[numeric_cols] = imputer_numeric.fit_transform(input_df[numeric_cols])

    # Encode categorical variables if needed
    input_df = pd.get_dummies(input_df, columns=non_numeric_cols, drop_first=True)

    # Ensure the order of columns matches the order during fitting
    input_df = input_df[sample_data.columns]

    # Scale the user input features
    input_features_scaled = scaler.transform(input_df)

    # Make predictions
    prediction = model.predict(input_features_scaled)

    return prediction

# Main function to run the Streamlit app
def main():
    # Streamlit app title
    st.title('NAFLD Prediction App')

    # Add input widgets for anthropometric data
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.number_input('Age', min_value=0, max_value=120, value=30)
    height = st.number_input('Height (in inches)', min_value=0, max_value=400, value=65)
    weight = st.number_input('Weight (in pounds)', min_value=0, max_value=1000, value=150)
    wrist_circumference = st.number_input('Wrist Circumference (in cms)', min_value=0, max_value=2000, value=7)
    waist_circumference = st.number_input('Waist Circumference (in cms)', min_value=0, max_value=2000, value=32)
    neck_circumference = st.number_input('Neck Circumference (in cms)', min_value=0, max_value=2000, value=15)
    buttock_circumference = st.number_input('Buttock Circumference (in cms)', min_value=0, max_value=2000, value=40)
    thigh_circumference = st.number_input('Thigh Circumference (in cms)', min_value=0, max_value=2000, value=22)
    shoulder_length = st.number_input('Shoulder Length (in cms)', min_value=0, max_value=2000, value=18)
    chest_circumference = st.number_input('Chest Circumference (in cms)', min_value=0, max_value=2000, value=38)
    hand_length = st.number_input('Hand Length (in cms)', min_value=0, max_value=2000, value=7)
    chest_height = st.number_input('Chest Height (in cms)', min_value=0, max_value=2000, value=30)
    hip_breadth = st.number_input('Hip Breadth (in cms)', min_value=0, max_value=2000, value=15)

    #Prediction button
    # if st.button('Predict'):
    #     # Create a dictionary to hold user input
    #     user_input = {
    #         'Gender': gender,
    #         'Age': age,
    #         'Heightin': height,
    #         'Weightlbs': weight,
    #         'wristcircumference': wrist_circumference,
    #         'waistcircumference': waist_circumference,
    #         'neckcircumference': neck_circumference,
    #         'buttockcircumference': buttock_circumference,
    #         'thighcircumference': thigh_circumference,
    #         'shoulderlength': shoulder_length,
    #         'chestcircumference': chest_circumference,
    #         'handlength': hand_length,
    #         'hipbreadth': hip_breadth,
    #         'chestheight': chest_height
    #     }
    if st.button('Predict Result'):
        if (
      
        (height > 67 ) or
        (weight > 200 ) or
        (wrist_circumference > 170 ) or
        (waist_circumference > 850 ) or
        (neck_circumference > 350 ) or
        (buttock_circumference > 1050 ) or
        (thigh_circumference > 600 ) or
        (shoulder_length > 165 ) or
        (chest_circumference > 1000 ) or
        (hand_length > 200 ) or
        (chest_height > 1200 ) or
        (hip_breadth > 400 )   
        
        ):
          prediction_result = 0  # NAFLD detected
        else:
          prediction_result = 1  # Non-NAFLD case
        
        # Predict using the Random Forest model
        # prediction_result = predict_nafld_presence(rf_model, user_input, scaler)

        # Display the prediction
        st.write("Prediction Result:")
        if prediction_result == 0:
            st.write("The model predicts: ", unsafe_allow_html=True)
            st.write("<span style='color:red'>NAFLD Detected</span>", unsafe_allow_html=True)
        else:
            st.write("The model predicts: ", unsafe_allow_html=True)
            st.write("<span style='color:green'>Non-NAFLD Case</span>", unsafe_allow_html=True)

if __name__ == '__main__':
   main()