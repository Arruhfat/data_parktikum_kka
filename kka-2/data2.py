import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data_praktikum_analisis_data.csv')

# Cleaning
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df = df.dropna(subset=['Total_Sales', 'CustomerID'])
df = df[df['Price_Per_Unit'] > 0]
df['Month'] = df['Order_Date'].dt.to_period('M').astype(str)
df['Profit_Margin'] = df['Total_Sales'] * 0.4

print("Data siap:", len(df), "baris")
print("="*50)
# TUGAS 1: TREN PENJUALAN BULANAN (Line Chart)
monthly_sales = df.groupby('Month')['Total_Sales'].sum()

plt.figure(figsize=(10,5))
plt.plot(monthly_sales.index, monthly_sales.values, marker='o', color='b')
plt.title('Tugas 1: Tren Penjualan Bulanan')
plt.xlabel('Bulan')
plt.ylabel('Total Penjualan')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# TUGAS 2: ANALISIS KORELASI (Heatmap)
corr = df[['Ad_Budget', 'Quantity', 'Price_Per_Unit', 'Total_Sales']].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Tugas 2: Korelasi Antar Variabel')
plt.tight_layout()
plt.show()
# TUGAS 3: IDENTIFIKASI PRODUK (Scatter Plot)
plt.figure(figsize=(10,6))
plt.scatter(df['Quantity'], df['Total_Sales'], alpha=0.5)
plt.axhline(y=df['Total_Sales'].median(), color='r', linestyle='--', label='Median Penjualan')
plt.axvline(x=df['Quantity'].median(), color='b', linestyle='--', label='Median Quantity')
plt.xlabel('Jumlah Terjual (Quantity)')
plt.ylabel('Total Penjualan')
plt.title('Tugas 3: Identifikasi Produk')
plt.legend()
plt.tight_layout()
plt.show()
# TUGAS 4: SEGMENTASI PELANGGAN (RFM Analysis)
snapshot = df['Order_Date'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'Order_Date': lambda x: (snapshot - x.max()).days,
    'Order_ID': 'count',
    'Total_Sales': 'sum'
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']

rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])

def get_segment(row):
    if row['R_Score'] >= 4 and row['F_Score'] >= 4:
        return 'Champion'
    if row['R_Score'] >= 4:
        return 'Loyal'
    if row['R_Score'] <= 2 and row['F_Score'] >= 3:
        return 'At Risk'
    if row['R_Score'] <= 2:
        return 'Lost'
    return 'Potential'

rfm['Segment'] = rfm.apply(get_segment, axis=1)

plt.figure(figsize=(10,6))
rfm['Segment'].value_counts().plot(kind='bar', color=['green','blue','orange','red','gray'])
plt.title('Tugas 4: Segmentasi Pelanggan (RFM)')
plt.xlabel('Segmen')
plt.ylabel('Jumlah Pelanggan')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# TUGAS 5: ANALISIS GEOGRAFIS (Horizontal Bar Chart)
region_col = 'Region' if 'Region' in df.columns else 'Product_Category'
profit_by_region = df.groupby(region_col)['Profit_Margin'].mean().sort_values()

plt.figure(figsize=(10,6))
profit_by_region.plot(kind='barh', color='salmon')
plt.title('Tugas 5: Profit Margin per Wilayah')
plt.xlabel('Rata-rata Profit Margin')
plt.ylabel(region_col)
plt.tight_layout()
plt.show()
# TUGAS 6: UJI HIPOTESIS DISKON DAN REGRESI LINEAR
print("="*50)
print("Tugas 6: Uji Hipotesis dan Regresi Linear")
print("="*50)
# 6a. Uji Hipotesis Diskon >20%
avg_price = df.groupby('Product_Category')['Price_Per_Unit'].transform('mean')
df['Discount_Flag'] = (df['Price_Per_Unit'] < 0.8 * avg_price).astype(int)

with_discount = df[df['Discount_Flag'] == 1]['Quantity']
without = df[df['Discount_Flag'] == 0]['Quantity']

t_stat, p_value = stats.ttest_ind(with_discount, without)

print("\nUji Hipotesis Diskon >20%")
print("-"*30)
print("Rata-rata quantity dengan diskon:", round(with_discount.mean(), 2))
print("Rata-rata quantity tanpa diskon:", round(without.mean(), 2))
print("P-value:", round(p_value, 4))

if p_value < 0.05:
    print("Kesimpulan: Diskon >20% SIGNIFIKAN meningkatkan volume penjualan")
else:
    print("Kesimpulan: Diskon >20% TIDAK SIGNIFIKAN meningkatkan volume penjualan")

# 6b. Regresi Linear (Iklan vs Penjualan)
if 'Ad_Budget' in df.columns:
    X = df[['Ad_Budget']]
    y = df['Total_Sales']
    
    model = LinearRegression()
    model.fit(X, y)
    
    print("\nRegresi Linear: Pengaruh Iklan terhadap Penjualan")
    print("-"*30)
    print("Koefisien iklan:", round(model.coef_[0], 4))
    print("Intercept:", round(model.intercept_, 2))
    print("R-Squared:", round(model.score(X, y), 4))
    
    plt.figure(figsize=(10,6))
    plt.scatter(X, y, alpha=0.5, label='Data Aktual')
    plt.plot(X, model.predict(X), color='red', linewidth=2, label='Garis Regresi')
    plt.xlabel('Biaya Iklan')
    plt.ylabel('Total Penjualan')
    plt.title('Tugas 6: Regresi Linear Iklan vs Penjualan')
    plt.legend()
    plt.tight_layout()
    plt.show()
# LAPORAN AKHIR
print("\n" + "="*50)
print("LAPORAN AKHIR PRAKTIKUM")
print("="*50)

print("\n1. Business Question")
print("-"*30)
print("Bagaimana mengoptimalkan strategi pemasaran untuk meningkatkan profit?")

print("\n2. Data Wrangling")
print("-"*30)
print("- Konversi Order_Date ke datetime")
print("- Hapus baris dengan Total_Sales atau CustomerID kosong")
print("- Filter Price_Per_Unit > 0")
print("- Buat kolom Month dan Profit_Margin")

print("\n3. Insights")
print("-"*30)
print("1. Tren penjualan:", "Meningkat" if monthly_sales.iloc[-1] > monthly_sales.iloc[0] else "Menurun")
print("2. Korelasi iklan vs penjualan:", round(df['Ad_Budget'].corr(df['Total_Sales']), 2))
print("3. Segmen pelanggan terbesar:", rfm['Segment'].value_counts().index[0])
print("4. Wilayah profit terendah:", profit_by_region.index[0])

print("\n4. Recommendations")
print("-"*30)
print("- Fokus promosi pada pelanggan Champion dan Loyal")
print("- Evaluasi strategi di wilayah dengan profit terendah")
if p_value < 0.05:
    print("- Pertahankan strategi diskon >20%")
else:
    print("- Gunakan strategi selain diskon (bundling atau gratis ongkir)")
if 'Ad_Budget' in df.columns and model.coef_[0] > 0.5:
    print("- Tingkatkan anggaran iklan karena ROI positif")