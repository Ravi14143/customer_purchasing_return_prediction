
# 🛍️ Customer Purchasing Prediction Dashboard

A **Streamlit-based machine learning application** that predicts and visualizes **customer purchasing behavior** using real or synthetic transactional data.  
It uses a **Stacked Regression Model (mod3.pkl)** trained on customer sales data to forecast future purchasing trends.

---

## 🚀 Features

- 📤 **Upload Customer Data** (CSV/Excel)
- 🧹 **Automatic Data Preprocessing**
- 📊 **Interactive Dashboard** – Visual insights on customer demographics, payment methods, and product categories
- 🤖 **Sales Prediction** – Predicts future purchase patterns using a pre-trained stacking model
- 🔁 **Synthetic Data Generation** – Generates realistic fake customer data for forecasting
- 🧠 **Repeated Customer Detection**

---

## 🧩 Tech Stack

- **Python 3.9+**
- **Streamlit** – UI framework  
- **Scikit-learn** – Data preprocessing & ML pipeline  
- **TensorFlow / Keras** – Deep learning model base  
- **XGBoost** – Gradient boosting for stacking model  
- **Pandas / NumPy** – Data handling  
- **Matplotlib / Seaborn** – Visualizations  
- **Faker** – Synthetic data generation  

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd customer-purchasing-prediction
````

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Place Model File

Ensure the file **`mod3.pkl`** (your trained stacking model) is in the project root directory.

---

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open your browser at:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 📁 Dataset Requirements

Upload a **CSV or Excel** file containing these columns:

```
invoice_no, customer_id, gender, age, category, quantity, price, payment_method, invoice_date, shopping_mall
```

📝 Example:

| invoice_no | customer_id | gender | age | category    | quantity | price | payment_method | invoice_date | shopping_mall |
| ---------- | ----------- | ------ | --- | ----------- | -------- | ----- | -------------- | ------------ | ------------- |
| 1001       | CUST001     | Male   | 32  | Electronics | 2        | 150.0 | Credit Card    | 15-03-2024   | City Mall     |

---

## 📊 Dashboard Sections

### 1. **Upload Dataset**

* Upload your dataset and view a preview.
* Perform **automated preprocessing** (date handling, encoding, normalization, outlier removal).

### 2. **Dashboard**

* Visualize customer data with:

  * Gender, category, and payment distribution
  * Yearly purchase counts for repeated customers

### 3. **Prediction**

* Uses the **Stacked Regression Model** to predict future sales.
* Generates **synthetic customer data** for forecasting.
* Displays:

  * 📈 Repeated customer trends
  * 💰 Best product category
  * 📦 Average sales info

---

## 🧠 Model Details

The **stacked regressor (`mod3.pkl`)** combines:

* Linear Regression
* XGBoost Regressor
* Neural Network (Keras)
  for accurate sales forecasting.

---

## 🧮 Output Highlights

* Predicted **sales and purchasing trends**
* Estimated **repeated customer rates**
* Best-performing **product category**
* Average sales and transactions

---

```

```
