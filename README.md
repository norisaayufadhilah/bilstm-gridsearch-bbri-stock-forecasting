# bilstm-gridsearch-bbri-stock-forecasting
Forecasting BBRI stock prices using BiLSTM with Grid Search hyperparameter optimization and model performance evaluation.
# ====== 1. IMPORT LIBRARY ======
# Mengimpor semua library yang dibutuhkan untuk pemrosesan data,
visualisasi, modelling, dan evaluasi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from datetime import timedelta
# Mengakses file di Google Drive
from google.colab import drive
drive.mount('/content/drive')

# ====== 2. LOAD DATA ======
# Membaca dataset saham dari Google Drive
# parse_dates digunakan agar kolom 'Date' dikenali sebagai tipe tanggal
# header=0 menandakan baris pertama sebagai nama kolom
df = pd.read_csv('/content/drive/My Drive/BBRI4.csv', parse_dates=['Date'],
header=0)
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
import numpy as np
import tensorflow as tf
import random, os
SEED = 1234
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(1234)

# ====== 3. STATISTIK DESKRIPTIF ======
print(" Statistik Deskriptif Kolom 'Close':")
print(df['Close'].describe())

# Plotting grafik time series untuk harga saham penutupan
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Close'], color='blue', label='Harga Saham Penutupan')
plt.title('Time Series Harga Saham Penutupan Harian PT. Bank Rakyat
Indonesia')
plt.xlabel('Tanggal')
plt.ylabel('Harga Saham (IDR)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ====== 4. DATA CLEANING YANG LENGKAP UNTUK TIME
SERIES ======
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# Mengubah kolom 'Date' menjadi datetime
df['Date'] = pd.to_datetime(df['Date'])
# 1. MEMBUAT RENTANG TANGGAL LENGKAP
# Ubah baris ini untuk secara eksplisit memulai dari tanggal 1
full_date_range = pd.date_range(start='2023-01-01', end=df['Date'].max(),
freq='D')
df_full = pd.DataFrame(full_date_range, columns=['Date'])
# 2. MENGGABUNGKAN DATA UNTUK MENGIDENTIFIKASI
MISSING VALUE
df_merged = pd.merge(df_full, df, on='Date', how='left')
# 4. MENAMPILKAN HASILNYA
pd.set_option('display.max_rows', None)
print("\n Data Setelah Diisi dengan Forward Fill dan Backward Fill:")
print(df_merged) # Contoh output
# ====== 6. SIMPAN FILE BERSIH KE CSV ======
df_merged[['Date', 'Close']].to_csv('data_sebelum_cleaned_close.csv',
index=False)
# ====== MENGISI MISSING VALUE DAN MEMASTIKAN TANGGAL
BERURUT ======
# Membuat rentang tanggal yang lengkap dari tanggal 1 Januari 2023
full_date_range = pd.date_range(start='2023-01-01', end=df['Date'].max(),
freq='D')

df_full = pd.DataFrame(full_date_range, columns=['Date'])
# Menggabungkan data asli dengan rentang tanggal yang lengkap
df_merged = pd.merge(df_full, df, on='Date', how='left')
# Mengisi missing values dengan harga dari hari sebelumnya (forward fill)
# dan mengisi jika ada NaN di awal dengan bfill
df_merged['Close'] = df_merged['Close'].ffill().bfill()
# Cek kembali missing values
print("\n Data Setelah Mengisi Missing Value dengan Forward Fill:")
print(df_merged['Close'].isna().sum())
# Menampilkan SELURUH data untuk verifikasi
print("\n Data Setelah Pengisian Missing Value:")
print(df_merged[['Date', 'Close']])
# ====== 7. SPLIT DATA SETELAH SCALING ======
train_size = int(len(df_merged) * 0.8)
data_train_scaled = df_merged[:train_size]
data_test_scaled = df_merged[train_size:]

# ====== 5. NORMALISASI DATA ======
# Inisialisasi scaler untuk normalisasi
scaler = MinMaxScaler()
# Normalisasi data training (fit and transform)
scaler.fit(data_train_scaled[['Close']])
data_train_scaled_values = scaler.transform(data_train_scaled[['Close']])
data_test_scaled_values = scaler.transform(data_test_scaled[['Close']])
df_normalized_train = pd.DataFrame({
'Date': data_train_scaled['Date'],
'Data Aktual': data_train_scaled['Close'],
'Normalized Data': data_train_scaled_values.flatten()
})

print("\n Tabel Data Training yang sudah Dinormalisasi:")
print(df_normalized_train)
print("\n Tabel Data Testing yang sudah Dinormalisasi:")
print(df_normalized_test)
# Menyimpan hasil normalisasi ke file CSV
df_normalized_train.to_csv('normalized_train.csv', index=False)
df_normalized_test.to_csv('normalized_test.csv', index=False)
# ====== 6. FUNGSI WINDOWING UNTUK TIME SERIES ======
def create_dataset(dataset, time_step=30):
X, y = [], []
for i in range(len(dataset) - time_step):
X.append(dataset[i:(i + time_step), 0])
y.append(dataset[i + time_step, 0])
return np.array(X), np.array(y)
# ====== 7. FUNCTION MEMBANGUN MODEL BiLSTM ======
def build_model(units, dropout, input_shape):
model = Sequential()
model.add(Bidirectional(LSTM(units, return_sequences=True),
input_shape=input_shape))
model.add(Dropout(dropout))
model.add(LSTM(units))
model.add(Dropout(dropout))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
return model

# ====== 9. GRID SEARCH DENGAN TIME SERIES CROSS
VALIDATION ======
neurons = [5, 10, 15, 20]
batch_sizes = [4, 16, 32]
epochs = [50, 100, 150]
dropouts = [0.1, 0.2]
time_steps = [30] # time_steps adalah list yang berisi nilai tunggal

results = []
best_mape = float('inf')
best_mse_scaled = float('inf')
best_params = {}
# Pisahkan data testing dan data training untuk normalisasi
data_scaled_2d_train = df_normalized_train['Normalized
Data'].values.reshape(-1, 1)
print(f" Total data: {len(data_scaled_2d_train)}")
print(" Memulai proses Grid Search...")
# ====== LOOP GRID SEARCH ======
for time_step in time_steps: # `time_step` akan menjadi nilai tunggal dari list
`time_steps`
X, y = create_dataset(data_scaled_2d_train, time_step) # Menggunakan
`time_step` yang bernilai tunggal
X = X.reshape(X.shape[0], X.shape[1], 1)
tscv = TimeSeriesSplit(n_splits=3)
for neuron in neurons:
for dropout in dropouts:
for batch in batch_sizes:
for epoch in epochs:
mape_scores = []
mse_denorm_scores = []
mse_scaled_scores = []
for train_idx, val_idx in tscv.split(X):

X_train_cv, X_val_cv = X[train_idx], X[val_idx]
y_train_cv, y_val_cv = y[train_idx], y[val_idx]
model = build_model(neuron, dropout, (time_step, 1))
model.fit(X_train_cv, y_train_cv, epochs=epoch,

batch_size=batch, verbose=0)

y_pred = model.predict(X_val_cv)
# Denormalisasi untuk evaluasi
y_val_denorm = scaler.inverse_transform(y_val_cv.reshape(-1,

1))

y_pred_denorm = scaler.inverse_transform(y_pred)
mape = mean_absolute_percentage_error(y_val_denorm,

y_pred_denorm)

mape_scores.append(mape)

mse_denorm = np.mean((y_val_denorm - y_pred_denorm) ** 2)
mse_denorm_scores.append(mse_denorm)
mse_scaled = np.mean((y_val_cv - y_pred.flatten()) ** 2)
mse_scaled_scores.append(mse_scaled)
# Hitung rata-rata per kombinasi
avg_mape = np.mean(mape_scores)
avg_mse_denorm = np.mean(mse_denorm_scores)
avg_mse_scaled = np.mean(mse_scaled_scores)
print(f"\n Kombinasi: TimeStep={time_step},

Neuron={neuron}, Dropout={dropout}, Batch={batch}, Epoch={epoch}")

print(f" Avg MAPE: {avg_mape:.4f}")
print(f" Avg MSE (Denormalized): {avg_mse_denorm:.2f}")
print(f" Avg MSE (Scaled): {avg_mse_scaled:.6f}")
results.append({
"time_step": time_step, "neuron": neuron, "dropout": dropout,
"batch": batch, "epoch": epoch, "MAPE": avg_mape,
"MSE_Denorm": avg_mse_denorm, "MSE_Scaled":

avg_mse_scaled
})
if avg_mse_scaled < best_mse_scaled:
best_mse_scaled = avg_mse_scaled
best_params = {
"time_step": time_step, "neuron": neuron, "dropout":

dropout,

"batch": batch, "epoch": epoch
}

print("\n Grid Search selesai.")
print(f" Kombinasi terbaik: {best_params}")
print(f" MSE Scaled Terbaik: {best_mse_scaled:.6f}")
# ====== 8. SIMPAN HASIL GRID SEARCH KE CSV DAN DRIVE
======
results_df = pd.DataFrame(results)
local_path = '/content/hasil_grid_search_bilstmm.csv'
drive_path = '/content/drive/MyDrive/hasil_grid_search_bilstmm.csv'
# Simpan ke lokal dan Google Drive
results_df.to_csv(local_path, index=False)
results_df.to_csv(drive_path, index=False)

# ====== 8. MENAMPILKAN HASIL GRID SEARCH ======
# Mengurutkan berdasarkan MAPE (primary) dan kemudian MSE_Scaled
(secondary)
df_results = pd.DataFrame(results).sort_values(by=["MAPE",
"MSE_Scaled"])
# Hapus kolom 'MSE' setelah DataFrame dibuat
if 'MSE' in df_results.columns:
df_results = df_results.drop(columns=['MSE'])
# Menampilkan seluruh hasil grid search
print("\n Hasil Grid Search Terurut Berdasarkan MAPE dan MSE
Scaled:")
print(df_results)
# Simpan hasil grid search ke file CSV di direktori kerja saat ini
df_results.to_csv('hasil_grid_search_terurut.csv', index=False)
# ====== 9. VISUALISASI PENGARUH HYPERPARAMETER ======
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

fig.suptitle('Pengaruh Hyperparameter Terhadap Kinerja Model (MAPE Rata-
Rata)', fontsize=16)

# a) Pengaruh Jumlah Neuron
sns.lineplot(x='neuron', y='MAPE', data=df_results, ax=axs[0, 0], marker='o')
axs[0, 0].set_title('a) Pengaruh Jumlah Neuron terhadap MAPE')
axs[0, 0].set_xlabel('Jumlah Neuron')
axs[0, 0].set_ylabel('MAPE Rata-Rata')
axs[0, 0].grid(True)
# b) Pengaruh Ukuran Batch
sns.lineplot(x='batch', y='MAPE', data=df_results, ax=axs[0, 1], marker='o')
axs[0, 1].set_title('b) Pengaruh Ukuran Batch terhadap MAPE')
axs[0, 1].set_xlabel('Ukuran Batch')
axs[0, 1].set_ylabel('MAPE Rata-Rata')
axs[0, 1].grid(True)
# c) Pengaruh Epoch
sns.lineplot(x='epoch', y='MAPE', data=df_results, ax=axs[1, 0], marker='o')
axs[1, 0].set_title('c) Pengaruh Epoch terhadap MAPE')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('MAPE Rata-Rata')
axs[1, 0].grid(True)

# d) Pengaruh Dropout
sns.lineplot(x='dropout', y='MAPE', data=df_results, ax=axs[1, 1], marker='o')
axs[1, 1].set_title('d) Pengaruh Dropout terhadap MAPE')
axs[1, 1].set_xlabel('Tingkat Dropout')
axs[1, 1].set_ylabel('MAPE Rata-Rata')
axs[1, 1].grid(True)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
# ====== 10. PELATIHAN MODEL TERBAIK ======
best_ts = int(best_params['time_step']) # Ganti 'time_steps' menjadi
'time_step'
best_neuron = int(best_params['neuron'])
best_dropout = float(best_params['dropout'])
best_batch = int(best_params['batch'])
best_epoch = int(best_params['epoch'])
# Dataset training window
X_train, y_train = create_dataset(data_scaled_2d_train, best_ts)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
model = build_model(best_neuron, best_dropout, (best_ts, 1))
early_stop = EarlyStopping(monitor='val_loss', patience=5,
restore_best_weights=True)
history = model.fit(
X_train, y_train,
epochs=best_epoch, batch_size=best_batch,
validation_split=0.1,
callbacks=[early_stop], verbose=1
)
print("\n Model BiLSTM berhasil dilatih dengan parameter terbaik dari
hasil Grid Search!")
# ====== 11. EVALUASI TRAINING ======
y_train_pred = model.predict(X_train)
y_train_pred_denorm = scaler.inverse_transform(y_train_pred)
y_train_denorm = scaler.inverse_transform(y_train.reshape(-1, 1))
mape_train = mean_absolute_percentage_error(y_train_denorm,
y_train_pred_denorm)
print(f"\n Evaluasi Training Set:")
print(f" MAPE (denorm): {mape_train:.4f}")

plt.figure(figsize=(12, 5))
plt.plot(y_train_denorm, label='Actual - Train')
plt.plot(y_train_pred_denorm, label='Predicted - Train')
plt.title('Prediksi vs Aktual Harga Saham (Training)')
plt.xlabel('Waktu')
plt.ylabel('Harga Saham (IDR)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
# ====== 12. EVALUASI TESTING ======
# Gabungkan train + test untuk membangun window yang mulus
data_scaled_2d_full = np.concatenate([
df_normalized_train['Normalized Data'].values,
df_normalized_test['Normalized Data'].values
]).reshape(-1, 1)
X_full, y_full = create_dataset(data_scaled_2d_full, best_ts)
train_len = len(df_normalized_train)
idx_test = np.where(np.arange(len(y_full)) + best_ts >= train_len)[0]
X_test = X_full[idx_test].reshape(-1, best_ts, 1)
y_test = y_full[idx_test]
# Prediksi test
y_test_pred = model.predict(X_test)
y_test_pred_denorm = scaler.inverse_transform(y_test_pred)
y_test_denorm = scaler.inverse_transform(y_test.reshape(-1, 1))
mape_test = mean_absolute_percentage_error(y_test_denorm,
y_test_pred_denorm)
print(f"\n Evaluasi Testing Set:")
print(f" MAPE (denorm): {mape_test:.4f}")
plt.figure(figsize=(12, 5))
plt.plot(y_test_denorm, label='Actual - Test')
plt.plot(y_test_pred_denorm, label='Predicted - Test')
plt.title('Prediksi vs Aktual Harga Saham (Testing)')
plt.xlabel('Waktu')
plt.ylabel('Harga Saham (IDR)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ====== 13. TABEL PERBANDINGAN DENGAN TANGGAL DAN APE
======
ape = np.abs((y_test_denorm.flatten() - y_test_pred_denorm.flatten()) /
y_test_denorm.flatten()) * 100
ape = np.where(y_test_denorm.flatten() == 0, np.nan, ape)
# Pastikan tanggal sesuai panjang data testing
test_dates = df['Date'].iloc[-len(y_test_denorm):].values # Perhatikan
pengambilan tanggal ini
df_perbandingan = pd.DataFrame({
'Tanggal': test_dates,
'Data Aktual': y_test_denorm.flatten().round(2),
'Prediksi BiLSTM': y_test_pred_denorm.flatten().round(2),
'APE (%)': ape.round(3)
})
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print("\n Perbandingan Data Aktual, Prediksi, dan APE:")
print(df_perbandingan)
# Simpan hasil ke Google Drive
output_path =
'/content/drive/MyDrive/hasil_perbandingan_aktual_vs_prediksi_ape.csv'
df_perbandingan.to_csv(output_path, index=False)
print(f"\n File hasil perbandingan berhasil disimpan ke: {output_path}")
# ====== 14. PERAMALAN 90 HARI KE DEPAN ======
print("\n PERAMALAN 90 HARI KE DEPAN")
forecast_horizon = 90
last_input = data_scaled_2d_full[-best_ts:].reshape(1, best_ts, 1)
future_preds_scaled = []
for _ in range(forecast_horizon):
next_pred_scaled = model.predict(last_input, verbose=0)[0][0]
future_preds_scaled.append(next_pred_scaled)
new_input = np.array(next_pred_scaled).reshape(1, 1, 1)
last_input = np.concatenate((last_input[:, 1:, :], new_input), axis=1)
future_preds =
scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1))

last_date = df['Date'].max()
future_dates = pd.date_range(start=last_date + timedelta(days=1),
periods=forecast_horizon)
df_forecast_90 = pd.DataFrame({'Date': future_dates, 'Predicted Close':
future_preds.flatten()})
df_forecast_90.to_csv('/content/drive/MyDrive/hasil_peramalan_90_hari.csv',
index=False)
print("\n Hasil Peramalan 90 Hari (10 baris pertama):")
print(df_forecast_90.head(10))
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label='Data Historis', color='blue')
plt.plot(df_forecast_90['Date'], df_forecast_90['Predicted Close'],
label='Prediksi 90 Hari ke Depan', color='red', linestyle='--')
plt.title('Prediksi Harga Saham 90 Hari ke Depan (Model BiLSTM)')
plt.xlabel('Tanggal')
plt.ylabel('Harga Saham (IDR)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
