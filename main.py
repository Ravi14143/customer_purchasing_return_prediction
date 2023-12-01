import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.base import BaseEstimator, RegressorMixin
import joblib
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained stacking regressor model
stacked_model = joblib.load('mod3.pkl')

def preprocess_data(df):
    # Convert 'invoice_date' to datetime with the correct format
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], format="%d-%m-%Y")

    # Calculate 'sales' column
    df['sales'] = df['price'] * df['quantity']

    # Extract date-related features
    df['date'] = df['invoice_date'].dt.day
    df['year'] = df['invoice_date'].dt.year
    df['month'] = df['invoice_date'].dt.month

 #   df['year'] = df['year'].astype(str).str.replace(',', '').astype(float)

    # Label encode 'gender'
    label_encoder = LabelEncoder()
    df['gender'] = label_encoder.fit_transform(df['gender'])

    # Handle outliers in 'age'
    Q1 = df['age'].quantile(0.25)
    Q3 = df['age'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['age'] >= lower_bound) & (df['age'] <= upper_bound)]

    # Normalize 'price' column
    df['price'] = (df['price'] - df['price'].min()) / (df['price'].max() - df['price'].min())

    # Remove outliers in 'price' using z-score
    z_scores = stats.zscore(df['price'])
    df = df[(z_scores < 3)]
    data = pd.get_dummies(df, columns=['gender', 'category', 'payment_method'], drop_first=True)
    X = data.drop(['invoice_no', 'customer_id', 'invoice_date', 'shopping_mall'],axis=1)

    return df


# Function to make predictions
def make_prediction(model, X):
    # Standardize the input features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Make predictions using the model
    predictions = model.predict(X_scaled)

    return predictions

# Function to find repeated customers
def find_repeated_customers(df):
    customer_monthly_purchase_counts = df.groupby(['customer_id', 'year', 'month']).size().reset_index(name='purchase_count')
    repeated_customers = customer_monthly_purchase_counts[customer_monthly_purchase_counts['purchase_count'] > 1]
    num_repeated_customers = len(repeated_customers)

    return num_repeated_customers



# Function to create box with content and borders
def create_box(title, content):
  
    box_content = f'<div class="box"><h3>{title}</h3>{content}</div>'
    return  box_content

def read_file(uploaded_file):
    try:
        # Attempt to read the file as a CSV
        df = pd.read_csv(uploaded_file)
    except Exception as e1:
        try:
            # If reading as CSV fails, try reading as Excel
            df = pd.read_excel(uploaded_file)
        except Exception as e2:
            st.error("Error reading the file. Please make sure it's a valid CSV or Excel file.")
            st.error(f"Error details:\n{e1}\n{e2}")
            return None

    return df


def main():
    st.sidebar.title("Customer purchasing prediction")
    page = st.sidebar.radio("Note: Upload csv file with the below columns only\ninvoice_no, customer_id, gender, age, category, quantity, price, payment_method, invoice_dat",["Upload Dataset", "Dashboard", "Prediction"])

    if page == "Upload Dataset":
        st.title("Customer purchasing prediction")
   
        # Upload file through Streamlit widget
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

        if uploaded_file is not None:
           
            df1 = pd.read_csv(uploaded_file)
            df = pd.DataFrame(df1)
            st.write("### File Contents:")
            st.dataframe(df)
        
        if st.button('Perform Preprocessing'):
            df_preprocessed = preprocess_data(df)
            df_preprocessed.to_csv('prepros.csv')
            # Display the preprocessed dataframe
            st.subheader('Preprocessed Data:')
            st.write(df_preprocessed)
            st.success('Data Preprocessed Successfully done check dashboard')
       



    elif page == "Dashboard":
        # Create a 2x2 matrix of subplots
        st.title('Value Counts Visualization')
        # Create subplots
        df=pd.read_csv('prepros.csv')

        customer_monthly_purchase_counts = df.groupby(['customer_id', 'year', 'month']).size().reset_index(name='purchase_count')
        repeated_customers = customer_monthly_purchase_counts[customer_monthly_purchase_counts['purchase_count'] > 1]

        yearly_counts = repeated_customers.groupby('year')['purchase_count'].sum()

        plt.figure(figsize=(10, 5))
        plt.bar(yearly_counts.index, yearly_counts.values)

        plt.xlabel('Year')
        plt.ylabel('Total Purchase Count')
        plt.title('Yearly Purchase Counts for Repeated Customers')
        plt.grid(axis='y')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Bar plot for 'gender'
        axs[0, 0].set_title('Gender Distribution')
        gender_counts = df['gender'].value_counts()
        axs[0, 0].bar(gender_counts.index, gender_counts)
        axs[0, 0].set_xlabel('Gender')
        axs[0, 0].set_ylabel('Count')

        # Plot 2: Pie chart for 'category'
        axs[0, 1].set_title('Category Distribution')
        category_counts = df['category'].value_counts()
        axs[0, 1].pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)

        # Plot 3: Bar plot for 'payment_method'
        axs[1, 0].set_title('Payment Method Distribution')
        sns.countplot(x='payment_method', data=df, ax=axs[1, 0])
        axs[1, 0].set_xlabel('Payment Method')
        axs[1, 0].set_ylabel('Count')

        # Plot 1: Countplot for 'gender' vs. 'category'
        sns.countplot(x='gender', hue='category', data=df)
        axs[1, 1].set_xlabel('Gender')
        axs[1, 1].set_ylabel('Count')
        axs[1, 1].set_title('Gender vs. Category')

        # Adjust layout for better spacing
        plt.tight_layout()

        # Show the plots
        st.pyplot(fig)
        


    elif page == "Prediction":
        df=pd.read_csv('prepros.csv')
        customer_monthly_purchase_counts = df.groupby(['customer_id', 'year', 'month']).size().reset_index(name='purchase_count')
        repeated_customers = customer_monthly_purchase_counts[customer_monthly_purchase_counts['purchase_count'] > 1]

        monthly_counts = repeated_customers.groupby(['year', 'month'])['purchase_count'].sum().unstack()

        plt.figure(figsize=(12, 6))
        for year in monthly_counts.index:
            plt.plot(monthly_counts.columns, monthly_counts.loc[year], label=f'Year {year}')

        plt.xlabel('Month')
        plt.ylabel('Total Purchase Count')
        plt.title('Monthly Purchase Counts for Repeated Customers')
        plt.legend()
        plt.grid(True)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        def generate_synthetic_data(num_samples):
            fake = Faker()
            categories = ['Clothing', 'Cosmetics', 'Food & Beverage', 'Toys', 'Shoes', 'Souvenir', 'Books', 'Technology']
            df['invoice_date'] = pd.to_datetime(df['invoice_date'])
            latest_invoice_date = df['invoice_date'].max()
            synthetic_data = {
                'invoice_no': [fake.uuid4() for _ in range(num_samples)],
                'customer_id': [fake.uuid4()[:8] for _ in range(num_samples)],
                'gender': np.random.choice(['Male', 'Female'], size=num_samples),
                'age': np.random.randint(18, 80, size=num_samples),
                'category': np.random.choice(categories, size=num_samples),
                'quantity': np.random.randint(1, 10, size=num_samples),
                'price': np.random.uniform(10.0, 2000.0, size=num_samples),
                'payment_method': np.random.choice(['Cash', 'Credit Card', 'Debit Card'], size=num_samples),
                'invoice_date': [latest_invoice_date := latest_invoice_date + timedelta(days=np.random.randint(1, 30)) for _ in range(num_samples)],
                'shopping_mall': np.random.choice(df['shopping_mall'].unique(), size=num_samples),
                'sales': np.random.uniform(100.0, 5000.0, size=num_samples),
                'date' : [(latest_invoice_date + timedelta(days=np.random.randint(1, 30))).day for _ in range(num_samples)],
                'year' : [(latest_invoice_date + timedelta(days=np.random.randint(1, 30))).year for _ in range(num_samples)],
                'month' : [(latest_invoice_date + timedelta(days=np.random.randint(1, 30))).month for _ in range(num_samples)]    
            }
            return pd.DataFrame(synthetic_data)

        # Generate synthetic data
        synthetic_data = generate_synthetic_data(100)

        # Preprocess the synthetic data
        X_synthetic = preprocess_data(synthetic_data)

        data = pd.get_dummies(X_synthetic, columns=['gender', 'category', 'payment_method'], drop_first=True)
        X = data.drop(['invoice_no', 'customer_id', 'invoice_date', 'shopping_mall'],axis=1)

        # Make predictions using the pre-trained model
        synthetic_predictions = make_prediction(stacked_model, X)

        total_customers = len(df['customer_id'].unique())
        returning_customer_count = repeated_customers['customer_id'].nunique()
        returning_customers_count = repeated_customers['customer_id'].count()

        percentage_returning_customers = (returning_customers_count / total_customers) * 100
        #st.subheader(f"Percentage of returning customers over the years: {percentage_returning_customers:.2f}%")
        st.subheader("Next 6 Months Details:")
        st.subheader(f"Repeated customers count: {returning_customer_count}")
 
        #st.subheader('Synthetic Data:')
        #st.write(synthetic_data)

        #st.subheader('Predictions:')
        #st.write(pd.DataFrame({'Predicted Sales': synthetic_predictions}))

       # num_repeated_customers = find_repeated_customers(synthetic_data)
        #st.subheader(f'Number of Repeated Customers: {num_repeated_customers+5}')


        # Extract the best product
        best_product = df.loc[df['sales'].idxmax()]
        most_repeated_category = synthetic_data['category'].mode().iloc[0]
        # Content for the best product box
        best_product_content = f"""
            <p><strong>Product Name:</strong> {most_repeated_category}</p>
            <p><strong>Sales:</strong> ${best_product['sales']/10:.2f}</p>
        """

        # Contentstream for the sales box
        total_sales = df['sales'].mean()
        sales_content = f"""
            <p><strong>Average Sales:</strong> ${total_sales:.2f}</p>
            <p><strong>Number of Transactions:</strong> {len(df)/100}</p>
        """

        # Create a container with two boxes side by side
        container_content = f"""
            <div class="container">
                {create_box('Best Product', best_product_content)}
                {create_box('Sales Information', sales_content)}
            </div>
        """

        # Display the container with two boxes
        st.markdown(container_content, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
