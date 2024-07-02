from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

df = pd.read_csv(
    r'C:\Users\laksh\Desktop\Hotels_booking\fact_bookings.csv',
    usecols=[
        'booking_id', 'property_id', 'booking_date', 'check_in_date', 'checkout_date',
        'no_guests', 'room_category', 'booking_platform', 'ratings_given',
        'booking_status', 'revenue_generated', 'revenue_realized', 'date',
        'mmm yy', 'day_type'
    ],

    parse_dates=['booking_date', 'check_in_date', 'checkout_date', 'date'],
    dayfirst=True
)
ratings_median = df['ratings_given'].median()
df['ratings_given'].fillna(ratings_median, inplace=True)

X_train = df.drop('revenue_generated', axis=1)
y_train = (df['revenue_generated'] > 18000).astype(int)

categorical_cols = ['room_category', 'booking_platform', 'booking_status', 'day_type']
timestamp_columns = ['booking_id', 'booking_date', 'check_in_date', 'checkout_date', 'date', 'mmm yy']
numerical_cols = df.drop(['revenue_generated'] + categorical_cols + timestamp_columns, axis=1).columns.tolist()

# Preprocessing for numerical data: scaling
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data: one-hot encoding
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

# Preprocessor that applies transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='drop'  # This drops the columns not specified in numerical_cols or categorical_cols
)

# Create the preprocessing and modeling pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=0))
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

print(X_train.isnull().sum())
print(y_train.isnull().sum())

# Define the home page
@app.route('/')
def index():
    # Render the form for user input
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extracting form data
        form_data = request.form
        user_input = pd.DataFrame({
            'property_id': [int(form_data['property_id'])],
            'booking_date': [pd.to_datetime(form_data['booking_date'])],
            'check_in_date': [pd.to_datetime(form_data['check_in_date'])],
            'checkout_date': [pd.to_datetime(form_data['checkout_date'])],
            'no_guests': [int(form_data['no_guests'])],
            'room_category': [form_data['room_category']],
            'booking_platform': [form_data['booking_platform']],
            'ratings_given': [float(form_data['ratings_given']) if form_data['ratings_given'] else None],
            'booking_status': [form_data['booking_status']],
            'revenue_realized': [int(form_data['revenue_realized'])],
            'date': [pd.to_datetime(form_data['date'])],
            'day_type': [form_data['day_type']],
            # Add other fields as necessary
        })

        # Predicting using the pipeline
        predicted_proba = pipeline.predict_proba(user_input)[:, 1]  # Probability of class 1 (high revenue)

        # Calculate revenue based on the predicted probability
        predicted_revenue = int(predicted_proba * 45220)  # Adjust the scaling factor as needed

        # Prepare the result to be displayed
        result = f'Predicted Revenue: ${predicted_revenue}'

        # # Predicting using the pipeline
        # prediction = pipeline.predict(user_input)
        #
        # # Prepare the result to be displayed
        # result = 'High Revenue' if prediction[0] == 1 else 'Low Revenue'

        # Render the result page with the prediction
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)