import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier

# Function -------------------------------------------------------------------------
def loadData():
    DATA_URL = ('data/KU-HAR_time_domain_subsamples_20750x300.csv')
    data = data = pd.read_csv(DATA_URL, header=None)
    return data

def loadDataInfo():
    data = {'Aktivitas': ['Stand','Sit','Talk-sit','Talk-stand','Stand-sit','Lay','Lay-stand','Pick','Jump','Push-up','Sit-up','Walk','Walk-backward','Walk-circle','Run','Stair-up','Stair-down','Table-tennis'],
        'Keterangan' : ["Berdiri diam (1 min)","Duduk diam (1 min)","Berbicara dengan gerakan tangan sambil duduk (1 min)","Berbicara dengan gerakan tangan sambil berdiri atau berjalan (1 min)","Berulang kali berdiri dan duduk (5 times)","Berbaring diam (1 min)","Berulang kali berdiri dan berbaring (5 times)","Mengambil benda dari lantai (10 times)","Melompat berulang kali (10 times)","Melakukan push-up penuh (5 times)","Melakukan sit-up (5 times)","Berjalan sejauh 20 meter (≈12 s)","Berjalan mundur sejauh 20 meters (≈20 s)","Berjalan di sepanjang jalan melingkar (≈ 20 s)","Berlari sejauh 20 meter (≈7 s)","Naik pada satu set tangga (≈1 min)","Turun dari satu set tanggas (≈50 s)","Bermain tenis meja (1 min)"],
        'Label': ['%d' % i for i in range(18)]}
    return data

def home():
    # Dataset
    st.subheader('Dataset')
    st.write("Dataset yang digunakan adalah KU-HAR: Human Activity Recognition Dataset (v 1.0) milik Niloy Sikder dari kaggle.com")
    st.write('Link dataset : https://www.kaggle.com/niloy333/kuhar')

    # Informasi
    st.subheader('Informasi Dataset')
    st.write("Human Activity Recognition (HAR) mengacu pada kapasitas mesin untuk melihat tindakan manusia. Dataset ini berisi informasi tentang 18 aktivitas berbeda yang dikumpulkan dari 90 peserta (75 pria dan 15 wanita) menggunakan sensor smartphone (Accelerometer dan Gyroscope). Dataset ini memiliki 1945 sampel aktivitas mentah yang dikumpulkan langsung dari para peserta, dan 20750 subsampel diambil dari mereka. Berikut tabel dari aktivitas tersebut")
    data = loadDataInfo()
    df = pd.DataFrame(data)
    st.table(df)

def eda():
    data = loadData()
    st.subheader('Exploratory Data & Analysis')
    # Tabel Dataset
    st.subheader('Data Human Activity Recognition')
    number = st.number_input('Masukkan index data', step=1, min_value=0, max_value=20749, key=1)
    st.write('Menampilkan 100 data dari index ke-', number)
    st.write(data[:][number:(number+100)])
    
    # Split Signals n Labels
    dff = data.values
    signals = dff[:, 0: 1800]  
    signals = np.array(signals, dtype=np.float32)
    labels = dff[:, 1800]

    # Load Data Info
    data_info = loadDataInfo()

    # Grafik
    st.subheader('Grafik Aktivitas')
    number2 = st.number_input('Masukkan index data', step=1, min_value=0, max_value=20749, key=2)
    st.write('Menampilkan grafik data kelas ', data_info['Aktivitas'][number2])
    acc_x = pd.DataFrame(signals[number2, 0: 300])
    acc_y = pd.DataFrame(signals[number2, 300: 600])
    acc_z = pd.DataFrame(signals[number2, 600: 900])
    gyr_x = pd.DataFrame(signals[number2, 900: 1200])
    gyr_y = pd.DataFrame(signals[number2, 1200: 1500])
    gyr_z = pd.DataFrame(signals[number2, 1500: 1800])
    with st.expander("Accelerometer X"):
        st.line_chart(acc_x)
    with st.expander("Accelerometer Y"):
        st.line_chart(acc_y)
    with st.expander("Accelerometer Z"):
        st.line_chart(acc_z)
    with st.expander("Gyroscope X"):
        st.line_chart(gyr_x)
    with st.expander("Gyroscope Y"):
        st.line_chart(gyr_y)
    with st.expander("Gyroscope Z"):
        st.line_chart(gyr_y)
            
def model():
    data = loadData()
    dff = data.values
    signals = dff[:, 0: 1800]  
    signals = np.array(signals, dtype=np.float32)
    labels = dff[:, 1800]

    # dft untuk merubah data signal ke distkrit
    fft = np.zeros(signals.shape, dtype=np.float32)
    for i in range(0,len(signals)):
        for j in range(0, 6):
            tmp = np.fft.fft(signals[i, j*300:(j+1)*300])
            fft[i, j*300:(j+1)*300] = abs(tmp)

    # split data, 80% data training 20% data testing
    x_train, x_test, y_train, y_test = train_test_split(fft,labels, test_size=0.2, random_state=9, stratify=labels)
   
   # algoritma KNN
    st.subheader('Algoritma KNN')
    knn = KNeighborsClassifier(n_neighbors=17)
    knn.fit(x_train,y_train)
    akurasi = knn.score(x_test,y_test)
    st.write('Hasil akurasi : ',akurasi)

    # algoritma random forest
    st.subheader('Algoritma Random Forest')
    lr_r = RandomForestClassifier(n_estimators=300, max_features='sqrt')
    lr_r.fit(x_train,  y_train)
    akurasi2 = lr_r.score(x_test, y_test)
    st.write('Hasil akurasi : ',akurasi2)

def credit():
    st.subheader('Kelompok 13')
    data = {'Nama': ["Dimas Agung Gumelar","Amin Fadilah","Yoshua A. Sitorus"],
        'Nim' : ["195150201111032","195150201111035","175150200111025"],
        'Nomor Absen': [26,27,2]}
    df = pd.DataFrame(data)
    st.table(df)
    
# Web ------------------------------------------------------------------------------
st.title('Project Pengantar Sains Data')

# Sidebar
page_names = ['Home', 'Exploratory Data & Analysis', 'Model', 'Credit']
page = st.sidebar.radio('', page_names)
if page == 'Home':
    home()
elif page == 'Exploratory Data & Analysis':
    eda()
elif page == 'Model':
    model()
else :
    credit()