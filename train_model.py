import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# Load your dataset
df = pd.read_csv("C:/Users/rajku/Downloads/Paytm-FebData.csv") 

# Clean and prepare
df_clean = df[['Order_ID', 'Transaction_Date', 'Amount']].copy()
df_clean.columns = ['Order_ID', 'Transaction_Date', 'Amount']
df_clean['Order_ID'] = df_clean['Order_ID'].str.replace("'", "")
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

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(rfm_scaled)

joblib.dump(scaler, 'scaler.pkl')
joblib.dump(kmeans, 'customer_segmentation_model.pkl')
