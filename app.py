import streamlit as st
import pandas as pd
import joblib

model = joblib.load('customer_segmentation_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🧠 Customer Segmentation App")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    df_clean = df[['Order_ID', 'Transaction_Date', 'Amount']].copy()
    df_clean.columns = ['Order_ID', 'Transaction_Date', 'Amount']
    df_clean['Transaction_Date'] = pd.to_datetime(df_clean['Transaction_Date'].str.replace("'", ""))
    df_clean = df_clean[df_clean['Amount'].notna()]

    snapshot_date = df_clean['Transaction_Date'].max() + pd.Timedelta(days=1)
    rfm = df_clean.groupby('Order_ID').agg({
        'Transaction_Date': lambda x: (snapshot_date - x.max()).days,
        'Order_ID': 'count',
        'Amount': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm = rfm.reset_index()

    rfm_scaled = scaler.transform(rfm[['Recency', 'Frequency', 'Monetary']])
    rfm['Cluster'] = model.predict(rfm_scaled)

    st.success("✅ Segmentation Complete!")
    st.dataframe(rfm)

    csv = rfm.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results", data=csv, file_name='segmented_customers.csv')
