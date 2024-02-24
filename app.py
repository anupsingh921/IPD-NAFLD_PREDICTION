import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('MALE_data.csv')

# Function to preprocess data and train the model
def preprocess_and_train(df):
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

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(sample_data, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    return rf_model, scaler, sample_data.columns

# Function to make predictions
def predict_nafld_presence(model, input_features, scaler, sample_data_columns):
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
    input_df = input_df.reindex(columns=sample_data_columns, fill_value=0)

    # Scale the user input features
    input_features_scaled = scaler.transform(input_df)

    # Make predictions
    prediction = model.predict(input_features_scaled)

    return prediction

# Main function to run the Streamlit app
def main():
    # Load the dataset and train the model
    rf_model, scaler, sample_data_columns = preprocess_and_train(df)

    # Streamlit app title
    st.title('NAFLD Prediction App')

    # Add input widgets for anthropometric data
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.number_input('Age', min_value=0, max_value=120, value=30)
    height = st.number_input('Height (in inches)', min_value=0, max_value=120, value=65)
    weight = st.number_input('Weight (in pounds)', min_value=0, max_value=1000, value=150)
    wrist_circumference = st.number_input('Wrist Circumference (in inches)', min_value=0, max_value=100, value=7)
    waist_circumference = st.number_input('Waist Circumference (in inches)', min_value=0, max_value=100, value=32)
    neck_circumference = st.number_input('Neck Circumference (in inches)', min_value=0, max_value=100, value=15)
    buttock_circumference = st.number_input('Buttock Circumference (in inches)', min_value=0, max_value=100, value=40)
    thigh_circumference = st.number_input('Thigh Circumference (in inches)', min_value=0, max_value=100, value=22)
    shoulder_length = st.number_input('Shoulder Length (in inches)', min_value=0, max_value=100, value=18)
    chest_circumference = st.number_input('Chest Circumference (in inches)', min_value=0, max_value=100, value=38)
    hand_length = st.number_input('Hand Length (in inches)', min_value=0, max_value=100, value=7)
    chest_height = st.number_input('Chest Height (in inches)', min_value=0, max_value=100, value=30)
    hip_breadth = st.number_input('Hip Breadth (in inches)', min_value=0, max_value=100, value=15)

    # Prediction button
    if st.button('Predict'):
        # Make predictions based on user input
        user_input = {
            'Gender': gender,
            'Age': age,
            'Heightin': height,
            'Weightlbs': weight,
            'wristcircumference': wrist_circumference,
            'waistcircumference': waist_circumference,
            'neckcircumference': neck_circumference,
            'buttockcircumference': buttock_circumference,
            'thighcircumference': thigh_circumference,
            'shoulderlength': shoulder_length,
            'chestcircumference': chest_circumference,
            'handlength': hand_length,
            'hipbreadth': hip_breadth,
            'chestheight': chest_height
        }

        prediction_result = predict_nafld_presence(rf_model, user_input, scaler, sample_data_columns)

        # Display prediction result
        st.write("Prediction Result:")
        if prediction_result == 0:
            st.write("The model predicts: NAFLD Detected")
        else:
            st.write("The model predicts: Non-NAFLD Case")

if __name__ == '__main__':
    main()



