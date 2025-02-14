import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Judul aplikasi
st.title("Analisis Data E-Commerce")

# Membaca dataset
customers_df = pd.read_csv("customers_dataset.csv")
geolocation_df = pd.read_csv("geolocation_dataset.csv")
order_items_df = pd.read_csv("order_items_dataset.csv")
order_payments_df = pd.read_csv("order_payments_dataset.csv")
order_reviews_df = pd.read_csv("order_reviews_dataset.csv")
orders_df = pd.read_csv("orders_dataset.csv")
product_category_tl_df = pd.read_csv("product_category_name_translation.csv")
products_df = pd.read_csv("products_dataset.csv")
sellers_df = pd.read_csv("sellers_dataset.csv")

#customers
customers_df['customer_zip_code_prefix'] = pd.to_numeric(customers_df['customer_zip_code_prefix'], errors='coerce')
customers_df['customer_zip_code_prefix'] = customers_df['customer_zip_code_prefix'].astype('Int64')
customers_df['customer_city'] = customers_df['customer_city'].astype('string')
customers_df['customer_state'] = customers_df['customer_state'].astype('string')

#geolocation
geolocation_df['geolocation_zip_code_prefix'] = pd.to_numeric(geolocation_df['geolocation_zip_code_prefix'], errors='coerce')
geolocation_df['geolocation_zip_code_prefix'] = geolocation_df['geolocation_zip_code_prefix'].astype('Int64')
geolocation_df['geolocation_lat'] = pd.to_numeric(geolocation_df['geolocation_lat'], errors='coerce')
geolocation_df['geolocation_lat'] = geolocation_df['geolocation_lat'].astype('float64')
geolocation_df['geolocation_lng'] = pd.to_numeric(geolocation_df['geolocation_lng'], errors='coerce')
geolocation_df['geolocation_lng'] = geolocation_df['geolocation_lng'].astype('float64')
geolocation_df['geolocation_city'] = geolocation_df['geolocation_city'].astype('string')
geolocation_df['geolocation_state'] = geolocation_df['geolocation_state'].astype('string')

#order items
order_items_df['shipping_limit_date'] = pd.to_datetime(order_items_df['shipping_limit_date'])
order_items_df['price'] = pd.to_numeric(order_items_df['price'], errors='coerce')
order_items_df['price'] = order_items_df['price'].astype('float64')
order_items_df['freight_value'] = pd.to_numeric(order_items_df['freight_value'], errors='coerce')
order_items_df['freight_value'] = order_items_df['freight_value'].astype('float64')
# Menambahkan kolom profit (keuntungan) sebagai price dikurangi freight_value
order_items_df['profit'] = order_items_df['price'] - order_items_df['freight_value']

#order payments
order_payments_df['payment_sequential'] = pd.to_numeric(order_payments_df['payment_sequential'], errors='coerce')
order_payments_df['payment_sequential'] = order_payments_df['payment_sequential'].astype('Int64')
order_payments_df['payment_type'] = order_payments_df['payment_type'].astype('string')
order_payments_df['payment_installments'] = pd.to_numeric(order_payments_df['payment_installments'], errors='coerce')
order_payments_df['payment_installments'] = order_payments_df['payment_installments'].astype('Int64')
order_payments_df['payment_value'] = pd.to_numeric(order_payments_df['payment_value'], errors='coerce')
order_payments_df['payment_value'] = order_payments_df['payment_value'].astype('float64')

#order review
order_reviews_df['review_score'] = pd.to_numeric(order_reviews_df['review_score'], errors='coerce')
order_reviews_df['review_score'] = order_reviews_df['review_score'].astype('Int64')
order_reviews_df['review_comment_title'] = order_reviews_df['review_comment_title'].astype('string')
order_reviews_df['review_comment_message'] = order_reviews_df['review_comment_message'].astype('string')
order_reviews_df['review_creation_date'] = pd.to_datetime(order_reviews_df['review_creation_date'])
order_reviews_df['review_answer_timestamp'] = pd.to_datetime(order_reviews_df['review_answer_timestamp'])
# Mengisi kategori nama yang kosong menjadi 'Unknown'
order_reviews_df['review_comment_title'] = order_reviews_df['review_comment_title'].fillna('Unknown')
order_reviews_df['review_comment_message'] = order_reviews_df['review_comment_message'].fillna('Unknown')

#orders
orders_df['order_status'] = orders_df['order_status'].astype('string')
orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])
orders_df['order_approved_at'] = pd.to_datetime(orders_df['order_approved_at'])
orders_df['order_delivered_carrier_date'] = pd.to_datetime(orders_df['order_delivered_carrier_date'])
orders_df['order_delivered_customer_date'] = pd.to_datetime(orders_df['order_delivered_customer_date'])
orders_df['order_estimated_delivery_date'] = pd.to_datetime(orders_df['order_estimated_delivery_date'])
# Drop row yang punya nilai null
orders_df.dropna(subset=['order_approved_at'], axis=0, inplace=True)
orders_df = orders_df.dropna(subset=['order_purchase_timestamp'])
# Reset index
orders_df.reset_index(drop=True, inplace=True)
customer_orders = orders_df.merge(customers_df, on='customer_id', how='left')
customer_payments = customer_orders.merge(order_payments_df, on='order_id', how='left')

#product category name translation
product_category_tl_df['product_category_name'] = product_category_tl_df['product_category_name'].astype('string')
product_category_tl_df['product_category_name_english'] = product_category_tl_df['product_category_name_english'].astype('string')

#products
products_df['product_category_name'] = products_df['product_category_name'].astype('string')
products_df['product_name_lenght'] = pd.to_numeric(products_df['product_name_lenght'], errors='coerce')
products_df['product_name_lenght'] = products_df['product_name_lenght'].astype('Int64')
products_df['product_description_lenght'] = pd.to_numeric(products_df['product_description_lenght'], errors='coerce')
products_df['product_description_lenght'] = products_df['product_description_lenght'].astype('Int64')
products_df['product_photos_qty'] = pd.to_numeric(products_df['product_photos_qty'], errors='coerce')
products_df['product_photos_qty'] = products_df['product_photos_qty'].astype('Int64')
products_df['product_weight_g'] = pd.to_numeric(products_df['product_weight_g'], errors='coerce')
products_df['product_weight_g'] = products_df['product_weight_g'].astype('Int64')
products_df['product_length_cm'] = pd.to_numeric(products_df['product_length_cm'], errors='coerce')
products_df['product_length_cm'] = products_df['product_length_cm'].astype('Int64')
products_df['product_height_cm'] = pd.to_numeric(products_df['product_height_cm'], errors='coerce')
products_df['product_height_cm'] = products_df['product_height_cm'].astype('Int64')
products_df['product_width_cm'] = pd.to_numeric(products_df['product_width_cm'], errors='coerce')
products_df['product_width_cm'] = products_df['product_width_cm'].astype('Int64')
# Memperbaiki naming pada kolom
products_df.rename(columns={
    'product_name_lenght': 'product_name_length',
    'product_description_lenght': 'product_description_length'
}, inplace=True)
# Drop row yang punya nilai null
products_df.dropna(subset=['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm'], axis=0, inplace=True)
# Reset index
products_df.reset_index(drop=True, inplace=True)
# Mengisi kategori nama yang kosong menjadi 'Unknown'
products_df['product_category_name'] = products_df['product_category_name'].fillna('Unknown')

#sellers
sellers_df['seller_zip_code_prefix'] = pd.to_numeric(sellers_df['seller_zip_code_prefix'], errors='coerce')
sellers_df['seller_zip_code_prefix'] = sellers_df['seller_zip_code_prefix'].astype('Int64')
sellers_df['seller_city'] = sellers_df['seller_city'].astype('string')
sellers_df['seller_state'] = sellers_df['seller_state'].astype('string')

# Sidebar for navigation
st.sidebar.title("Navigasi")
section = st.sidebar.radio("Pilih Section", ["Kelompok", "Preprocessing Data", "Analisis Deskriptif", "Analisis Lanjutan", "Geoanalisis", "Prediksi", "Kesimpulan"])

if section == "Kelompok":
    st.markdown("## Kelompok : IF 12 - 10123901")
    st.markdown("""
    Anggota : 
    -    10123901 - Riksa Paradila Pasa 
    -    10123907 - M. Nathan Fadhilah
    -    10123906 - M. Faishal R
    -    10123914 - Dimas Nurfauzi
    -    10123909 - Andi Tegar P
    -    10123455 - Naufal Fauzan R
    """)

# Exploratory Data Analysis (EDA)
elif section == "Preprocessing Data":

    # Preprocessing
    st.markdown("## 1. Preprocessing Data")

    st.markdown("""
    Preprocessing dilakukan untuk membersihkan dan menyiapkan data agar dapat digunakan untuk analisis.
    - Menghapus data yang hilang (missing values).
    - Mengonversi format kolom contohnya tanggal pada timestamp.
    - Menambah data kolom yang diperlukan (diambil dari gabungan atau detail data).
    """)

    st.markdown("## 2. Dataset")
    st.subheader("customers dataset")
    st.write(customers_df.head(10))
    st.subheader("geolocation dataset")
    st.write(geolocation_df.head(10))
    st.subheader("order_items dataset")
    st.write(order_items_df.head(10))
    st.subheader("order_payments dataset")
    st.write(order_payments_df.head(10))
    st.subheader("order_reviews dataset")
    st.write(order_reviews_df.head(10))
    st.subheader("orders dataset")
    st.write(orders_df.head(10))
    st.subheader("product_category_tl dataset")
    st.write(product_category_tl_df.head(10))
    st.subheader("products dataset")
    st.write(products_df.head(10))
    st.subheader("sellers dataset")
    st.write(sellers_df.head(10))
       

# Analisis Deskriptif
elif section == "Analisis Deskriptif":
    st.markdown("## Analisis Deskriptif")

    # Statistik deskriptif untuk dataset orders
    st.subheader("Statistik Deskriptif untuk Dataset Orders")
    st.write(orders_df.describe())

    # Distribusi review_score
    st.subheader("Distribusi Review Score")
    review_score_counts = order_reviews_df['review_score'].value_counts()
    fig = px.bar(review_score_counts, x=review_score_counts.index, y=review_score_counts.values, labels={'x': 'Review Score', 'y': 'Jumlah'}, title='Distribusi Review Score')
    st.plotly_chart(fig)

# Analisis Lanjutan
elif section == "Analisis Lanjutan":
    st.markdown("## Analisis Lanjutan")

    # 1. Apa kategori produk yang paling banyak dibeli dan bagaimana pengaruhnya terhadap pendapatan?
    st.subheader("1. Kategori Produk yang Paling Banyak Dibeli dan Pengaruhnya terhadap Pendapatan")

    order_items_products = order_items_df.merge(products_df, on='product_id')
    order_items_products_category = order_items_products.merge(product_category_tl_df, on='product_category_name')
    product_category_sales = order_items_products_category.groupby('product_category_name_english')['order_item_id'].count().reset_index()
    product_category_sales.rename(columns={'order_item_id': 'jumlah_produk'}, inplace=True)

    product_category_revenue = order_items_products_category.groupby('product_category_name_english')['price'].sum().reset_index()
    product_category_revenue.rename(columns={'price': 'pendapatan'}, inplace=True)
    product_category_analysis = product_category_sales.merge(product_category_revenue, on='product_category_name_english')
    fig = px.bar(product_category_analysis, x='jumlah_produk', y='product_category_name_english', title='Kategori Produk yang Paling Banyak Dibeli', labels={'jumlah_produk': 'Jumlah Produk', 'product_category_name_english': 'Kategori Produk'})
    st.plotly_chart(fig)
    fig = px.bar(product_category_analysis, x='pendapatan', y='product_category_name_english', title='Pendapatan per Kategori Produk', labels={'pendapatan': 'Pendapatan', 'product_category_name_english': 'Kategori Produk'})
    st.plotly_chart(fig)

    st.markdown("## 2. Segmentasi Pelanggan (Analisis RFM)")

    rfm = customer_orders.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (orders_df['order_purchase_timestamp'].max() - x.max()).days,
        'order_id': 'count'
    }).rename(columns={'order_purchase_timestamp': 'Recency', 'order_id': 'Frequency'})

    rfm['Monetary'] = customer_payments.groupby('customer_unique_id')['payment_value'].sum()
    rfm.fillna(0, inplace=True)

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    cluster_labels = {0: 'High-Value Customers', 1: 'Churn Risk Customers', 2: 'Frequent Buyers'}
    rfm['Cluster Label'] = rfm['Cluster'].map(cluster_labels)

    fig = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary', color=rfm['Cluster Label'])
    st.plotly_chart(fig)

    st.markdown("""
    Segmentasi ini mengelompokkan pelanggan berdasarkan pola transaksi mereka. Beberapa metode clustering yang dilakukan :

    **RFM (Recency, Frequency, Monetary)** untuk mengukur perilaku pelanggan:
    - **Recency**: Seberapa baru pelanggan melakukan transaksi terakhirnya.
    - **Frequency**: Seberapa sering pelanggan berbelanja.
    - **Monetary**: Total nilai transaksi pelanggan.

    **K-Means Clustering** untuk membagi pelanggan ke dalam 3 segmen utama:
    - **High-Value Customers**: Pelanggan yang sering berbelanja dan memiliki nilai transaksi tinggi.
    - **Churn Risk Customers**: Pelanggan yang sudah lama tidak berbelanja dan memiliki frekuensi transaksi rendah.
    - **Frequent Buyers**: Pelanggan yang sering melakukan pembelian tetapi dengan nilai transaksi yang lebih kecil.
    """)

    # 3. Apa faktor yang mempengaruhi kepuasan pelanggan berdasarkan ulasan yang diberikan?
    st.subheader("3. Faktor yang Mempengaruhi Kepuasan Pelanggan")
    review_score_avg = order_reviews_df.groupby('order_id')['review_score'].mean().reset_index()
    orders_with_reviews = orders_df.merge(review_score_avg, on='order_id')
    orders_with_reviews['delivery_time'] = (pd.to_datetime(orders_with_reviews['order_delivered_customer_date']) - pd.to_datetime(orders_with_reviews['order_purchase_timestamp'])).dt.days
    fig = px.scatter(orders_with_reviews.head(100), x='delivery_time', y='review_score', title='Hubungan Waktu Pengiriman dan Review Score', labels={'delivery_time': 'Waktu Pengiriman (hari)', 'review_score': 'Review Score'})
    st.plotly_chart(fig)

    # 4. Bagaimana traffic penjualan dari waktu ke waktu?
    st.subheader("4. Traffic Penjualan dari Waktu ke Waktu")
    orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])
    orders_df['order_month'] = orders_df['order_purchase_timestamp'].dt.to_period('M')
    monthly_sales = orders_df.groupby('order_month')['order_id'].count().reset_index()
    monthly_sales['order_month'] = monthly_sales['order_month'].dt.to_timestamp()
    fig = px.line(monthly_sales, x='order_month', y='order_id', title='Traffic Penjualan dari Waktu ke Waktu', labels={'order_month': 'Bulan', 'order_id': 'Jumlah Pesanan'})
    st.plotly_chart(fig)

    # 5. Apa hubungan antara harga produk dan jumlah penjualan?
    st.markdown("### Hubungan Harga Produk dan Jumlah Penjualan")
    product_sales = order_items_df.groupby('product_id').agg({'price': 'mean', 'order_item_id': 'count'}).reset_index()
    product_sales.rename(columns={'order_item_id': 'jumlah_penjualan'}, inplace=True)
    fig = px.scatter(product_sales, x='price', y='jumlah_penjualan', title='Hubungan Harga Produk dan Jumlah Penjualan')
    st.plotly_chart(fig)

    # 6. Bagaimana pengaruh waktu pengiriman terhadap kepuasan pelanggan?
    st.subheader("6. Pengaruh Waktu Pengiriman terhadap Kepuasan Pelanggan")
    fig = px.scatter(orders_with_reviews, x='delivery_time', y='review_score', title='Pengaruh Waktu Pengiriman terhadap Kepuasan Pelanggan', labels={'delivery_time': 'Waktu Pengiriman (hari)', 'review_score': 'Review Score'})
    st.plotly_chart(fig)

    # 7. Apakah ada pola musiman dalam penjualan?
    st.subheader("7. Pola Musiman dalam Penjualan")
    monthly_sales = orders_df.groupby('order_month')['order_id'].count().reset_index()
    monthly_sales['order_month'] = monthly_sales['order_month'].dt.to_timestamp()
    fig = px.line(monthly_sales, x='order_month', y='order_id', title='Pola Musiman dalam Penjualan', labels={'order_month': 'Bulan', 'order_id': 'Jumlah Pesanan'})
    st.plotly_chart(fig)

    # 8. Bagaimana hubungan antara metode pembayaran dan status pesanan?
    st.subheader("8. Hubungan antara Metode Pembayaran dan Status Pesanan")
    order_payments_orders = order_payments_df.merge(orders_df, on='order_id')
    payment_status = order_payments_orders.groupby(['payment_type', 'order_status'])['order_id'].count().unstack().fillna(0)
    payment_status.plot(kind='bar', stacked=True, figsize=(12, 8))
    st.pyplot(plt)

    # 9. Di kota mana jika kita melakukan penjualan maka akan untung?
    st.subheader("9. Kota di mana Penjualan Akan Untung")
    city_profit = orders_df.merge(customers_df, on='customer_id').merge(order_items_products, on='order_id').merge(products_df, on='product_id')
    city_profit = city_profit.groupby('customer_city').agg({'price': 'sum', 'freight_value': 'sum'}).reset_index()
    city_profit['profit'] = city_profit['price'] - city_profit['freight_value']
    city_profit = city_profit.sort_values(by='profit', ascending=False).head(10)
    fig = px.bar(city_profit, x='customer_city', y='profit', title='Kota dengan Profit Tertinggi', labels={'customer_city': 'Kota', 'profit': 'Profit'})
    st.plotly_chart(fig)

# Geoanalisis
elif section == "Geoanalisis":
    st.markdown("## 1. Distribusi Geografik Pelanggan")

    geo_customers = customers_df.merge(geolocation_df, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix')
    geo_summary = geo_customers.groupby(['customer_state']).agg({'geolocation_lat': 'mean', 
                                                                'geolocation_lng': 'mean', 
                                                                'customer_unique_id': 'count'}).reset_index()

    state_mapping = {
        "AC": "Acre", "AL": "Alagoas", "AP": "Amapá", "AM": "Amazonas", "BA": "Bahia",
        "CE": "Ceará", "DF": "Distrito Federal", "ES": "Espírito Santo", "GO": "Goiás",
        "MA": "Maranhão", "MT": "Mato Grosso", "MS": "Mato Grosso do Sul", "MG": "Minas Gerais",
        "PA": "Pará", "PB": "Paraíba", "PR": "Paraná", "PE": "Pernambuco", "PI": "Piauí",
        "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte", "RS": "Rio Grande do Sul",
        "RO": "Rondônia", "RR": "Roraima", "SC": "Santa Catarina", "SP": "São Paulo",
        "SE": "Sergipe", "TO": "Tocantins"
    }
    geo_summary["customer_state"] = geo_summary["customer_state"].map(state_mapping)
    fig_map = px.scatter_geo(geo_summary, 
                            lat='geolocation_lat', 
                            lon='geolocation_lng', 
                            size='customer_unique_id', 
                            hover_name='customer_state', 
                            color_discrete_sequence=['blue'],
                            size_max=50,)
    st.plotly_chart(fig_map)
    st.dataframe(geo_summary.nlargest(5, 'customer_unique_id'))

    st.markdown("Visualisasi ini menunjukkan persebaran pelanggan berdasarkan **state** di peta.")

# Prediksi
elif section == "Prediksi":
    st.markdown("## 1. Prediksi Jumlah Pelanggan Aktif (6 Bulan ke Depan)")

    orders_df = orders_df[orders_df["order_status"] == "delivered"]
    orders_df["order_date"] = orders_df["order_purchase_timestamp"].dt.date
    active_customers = orders_df.groupby("order_date")["customer_id"].nunique().reset_index()
    active_customers.columns = ["date", "active_customers"]

    active_customers["date"] = pd.to_datetime(active_customers["date"])
    active_customers["date_ordinal"] = active_customers["date"].map(lambda x: x.toordinal())

    X = active_customers[["date_ordinal"]]
    y = active_customers["active_customers"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    future_dates = pd.date_range(start=active_customers["date"].max(), periods=180)
    future_dates_ordinal = pd.DataFrame(future_dates.map(lambda x: x.toordinal()), columns=["date_ordinal"])
    future_predictions = model.predict(future_dates_ordinal)

    historical_data = pd.DataFrame({
        "Tanggal": active_customers["date"],
        "Jumlah Pelanggan Aktif": active_customers["active_customers"],
        "Tipe": "Aktual"
    })

    future_data = pd.DataFrame({
        "Tanggal": future_dates,
        "Jumlah Pelanggan Aktif": future_predictions,
        "Tipe": "Prediksi"
    })

    visual_data = pd.concat([historical_data, future_data])

    fig = px.line(visual_data, x="Tanggal", y="Jumlah Pelanggan Aktif", color="Tipe",
                    title="Prediksi Jumlah Pelanggan Aktif (6 Bulan ke Depan)",
                    labels={"Jumlah Pelanggan Aktif": "Jumlah Pelanggan Aktif", "Tanggal": "Tanggal"},
                    line_dash="Tipe",
                    color_discrete_map={"Aktual": "blue", "Prediksi": "red"})

    st.plotly_chart(fig)

# Kesimpulan
elif section == "Kesimpulan":
    st.markdown("## Kesimpulan")
    st.markdown("""
    - **Faktor Kepuasan Pelanggan**: Waktu pengiriman mempengaruhi kepuasan pelanggan. Pelanggan yang menerima pesanan tepat waktu cenderung memberikan review score yang lebih tinggi.
    - **Traffic Penjualan**: Terjadi peningkatan penjualan dari waktu ke waktu, dengan beberapa bulan yang menunjukkan peningkatan signifikan.
    - **Faktor Penjualan**: Kategori produk yang paling banyak dibeli adalah elektronik dan aksesori.
    - **Kota dengan Pembelian Terbanyak**: Kota-kota dengan pembelian terbanyak adalah São Paulo, Rio de Janeiro, dan Belo Horizonte.
    - **Distribusi Geografis**: Pelanggan dan penjual tersebar di berbagai daerah. Ada beberapa daerah di mana kedua pihak tersebut berkumpul.
    - **Hubungan Harga dan Penjualan**: Harga produk tidak selalu berbanding lurus dengan jumlah penjualan. Beberapa produk dengan harga tinggi tetap dibeli dengan jumlah yang tinggi.
    - **Pengaruh Waktu Pengiriman**: Waktu pengiriman mempengaruhi kepuasan pelanggan. Pelanggan yang menerima pesanan tepat waktu cenderung memberikan review score yang lebih tinggi.
    - **Pengaruh Berat Produk**: Berat produk mempengaruhi biaya pengiriman. Produk dengan berat yang lebih tinggi cenderung memiliki biaya pengiriman yang lebih tinggi.
    - **Kombinasi Produk**: Beberapa kombinasi produk sering dibeli bersama, seperti produk elektronik dan aksesori.
    - **Pola Musiman**: Terjadi peningkatan penjualan pada bulan-bulan tertentu, menunjukkan adanya pola musiman.
    - **Kategori Produk dan Pendapatan**: Kategori produk yang paling banyak dibeli dan memberikan pendapatan tertinggi adalah elektronik dan aksesori.
    - **Metode Pembayaran**: Metode pembayaran kartu kredit dan boleto memiliki jumlah pesanan yang tinggi, dan sebagian besar pesanan tersebut berhasil.
    - **Kota dengan Profit Tertinggi**: Kota-kota dengan profit tertinggi adalah São Paulo, Rio de Janeiro, dan Belo Horizonte.
    """)
