#########################################################
# Proje: Seer Breast Cancer Data Yapay Zeka Projesi
# Ders: Yapay Zekaya Giriş
# Dersin Kodu: MBM6-453
# Dersin Öğretmeni: Doç. Dr. Mete YAĞANOĞLU
# Bölüm: Bilgisayar Mühendisliği(İÖ)
# Ad: Muhammed Emin
# Soyad: Yelaldı
# Okul Numarası: 210757029
#########################################################

# Kullandığım gerekli kütüphaneler
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.utils import shuffle
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score, roc_curve, auc)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imblearn.over_sampling import SMOTE


# Veri Çoğaltma: Görüntü tabanlı model için ImageDataGenerator kullanımı.
datagen = ImageDataGenerator(rescale=1.0/255.0)

# Rastgelelik kontrolü için sabit bir SEED belirliyoruz, böylece sonuçlar tekrar üretilebilir.
# NOT: Random state değerini tek tek değiştirmemek için böyle bir değişken atadım.
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Veri Setini Yükleme ve Temizleme
dosya_yolu = "data.csv"
veri = pd.read_csv(dosya_yolu)

# Gereksiz sütunları kaldırıyoruz (tamamen boş olanlar)
veri = veri.dropna(axis=1, how='all')

# Kısmen boş olan sütunlar için %80 doluluk kriterini uyguluyoruz.
veri = veri.dropna(thresh=int(0.8 * len(veri)), axis=1)

# Bu arada sütun isimlerini temizleyelim. Sık rastlanan boşluk hataları var.
veri.columns = veri.columns.str.strip()

# Yeni bir öznitelik oluşturuyoruz: Pozitif_Dugum_Orani
# Bu oran, Reginol Node Positive değerinin Regional Node Examined'e bölünmesiyle hesaplanır.
# Eksik değerler varsa 0 olarak doldurulur.
veri['Pozitif_Dugum_Orani'] = (veri['Reginol Node Positive'] / veri['Regional Node Examined']).fillna(0)

# Veri Keşfi (EDA)
print("Veri Seti Özet Bilgisi:")
print(veri.info())   # Veri setinin genel özet bilgisi
print("\nVeri Seti Özet İstatistikleri:")
print(veri.describe())  # Sayısal sütunlar için özet istatistikler

# EDA Korelasyon Matrisi: Sayısal sütunlar arasındaki ilişkileri analiz ediyoruz.
sayisal_veri = veri.select_dtypes(include=[np.number])  # Sadece sayısal sütunları seçiyoruz

# Korelasyon grafiğini çiziyoruz. `annot` False çünkü görsel kirliliği önlemek istiyoruz.
plt.figure(figsize=(12, 10))
sns.heatmap(sayisal_veri.corr(), annot=False, cmap='coolwarm')
plt.title("Korelasyon Matrisi (Sayısal Veriler)")
plt.show()

# EDA Aykırı Değer Analizi: 

# Aykırı Değerlerin İşlenmesi (IQR Yöntemi)
Q1 = veri['Pozitif_Dugum_Orani'].quantile(0.25)  # İlk çeyrek
Q3 = veri['Pozitif_Dugum_Orani'].quantile(0.75)  # Üçüncü çeyrek
IQR = Q3 - Q1  # Çeyrekler arası fark
lower_bound = Q1 - 1.5 * IQR  # Alt sınır
upper_bound = Q3 + 1.5 * IQR  # Üst sınır
# Aykırı değerleri alt veya üst sınırlarla değiştiriyoruz.
veri['Pozitif_Dugum_Orani'] = np.where(veri['Pozitif_Dugum_Orani'] > upper_bound, upper_bound,
    np.where(veri['Pozitif_Dugum_Orani'] < lower_bound, lower_bound,veri['Pozitif_Dugum_Orani']))

sns.boxplot(data=veri[['Pozitif_Dugum_Orani']])
plt.title("Aykırı Değerler (Pozitif Düğüm Oranı)")
plt.show()

# Kategorik Sütunların Kodlanması (Label Encoding):
# Not: Kategorik sütunları modelin anlayabileceği sayısal formatlara dönüştürüyoruz.
# Bunun için LabelEncoder yöntemi kullandım
label_encoder = LabelEncoder()

# Kodlama yapılacak kategorik sütunlar
kategorik_sutunlar = ['Race', 'Marital Status', 'T Stage', 'N Stage', '6th Stage', 'Grade',
                       'A Stage', 'Estrogen Status', 'Progesterone Status']

# Her bir kategorik sütun için LabelEncoder yöntemiyle uyguluyoruz
for sutun in kategorik_sutunlar:
    # Her sütunu ayrı ayrı encode ediyoruz
    veri[sutun] = label_encoder.fit_transform(veri[sutun])


# Özellik ve Hedef Ayrımı:

# Özellikler (X) ve hedef değişken (y) ayrımı yapıyoruz.
X = veri.drop(columns=['Status'])  # 'Durum' yerine 'Status' kullanıyoruz
y = veri['Status'].apply(lambda x: 1 if x == 'Alive' else 0)  # 'Hayatta' yerine 'Alive' kullanıyoruz

# Not: Veriyi karıştırmak, eğitim ve test setlerini daha dengeli hale getirebilir.
X, y = shuffle(X, y, random_state=SEED)

# Veriyi eğitim ve test setlerine ayırıyoruz (%80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# PCA ile Boyut Azaltma:

# Veriyi ölçeklendirme uyguluyoruz
scaler = StandardScaler()

# Ölçeklendirme işlemi
X_train_scaled = scaler.fit_transform(X_train)  # Sadece eğitim setiyle fit ediyoruz
X_test_scaled = scaler.transform(X_test)  # Test setine aynı ölçeklemeyi uyguluyoruz

# İlk olarak PCA'yı tüm bileşenlerle deneyerek varyans oranını analiz ediyoruz.
pca = PCA()  
X_train_pca_full = pca.fit_transform(X_train_scaled)

# Kümülatif varyans oranını hesaplıyoruz
kumulatif_varyans = np.cumsum(pca.explained_variance_ratio_)

# %95 varyansı açıklayan minimum bileşen sayısı
optimal_bilesen_sayisi = np.argmax(kumulatif_varyans >= 0.95) + 1
print(f"%95 varyansı açıklayan minimum bileşen sayısı: {optimal_bilesen_sayisi}")

# PCA ile boyut azaltma işlemini tamamlıyoruz.
pca = PCA(n_components=optimal_bilesen_sayisi)
X_train_pca = pca.fit_transform(X_train_scaled)  # PCA dönüşümü
X_test_pca = pca.transform(X_test_scaled)

# PCA açıklanan varyans grafiği
plt.figure(figsize=(8, 5))
plt.bar(range(1, optimal_bilesen_sayisi + 1), pca.explained_variance_ratio_)
plt.title('PCA Açıklanan Varyans Oranları')
plt.ylabel('Varyans Oranı')
plt.xlabel('Ana Bileşenler')
plt.show()

# LGBM Modeli Eğitimi kullandım
lgb_model = lgb.LGBMClassifier(random_state=SEED)
lgb_model.fit(X_train, y_train)

# Öznitelik önem analizi
oznitelik_onem = pd.DataFrame({
    'Oznitelik': X.columns,
    'Onem': lgb_model.feature_importances_
}).sort_values(by='Onem', ascending=False)

# İlk 10 önemli özniteliğin görselleştirilmesi
en_onemli_oznitelikler = oznitelik_onem.head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=en_onemli_oznitelikler, x='Onem', y='Oznitelik', palette="coolwarm")
plt.title('Öznitelik Önem Sıralaması (LightGBM)')
plt.xlabel('Önem Skoru')
plt.ylabel('Öznitelikler')
plt.show()

# Makine Öğrenmesi Modelleri:
# Çeşitli modelleri tanımlıyoruz.
# Not: Öncelikle çeşitli modelleri tanımlıyoruz. Daha sonra bunları eğitim ve değerlendirme için kullanacağız.

# SMOTE uygulaması
smote = SMOTE(random_state=SEED)

# Eğitim verisini SMOTE ile dengeleme
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("SMOTE Sonrası Eğitim Verisi Dağılımı:")
print(y_train_smote.value_counts())

# Tüm modelleri yeniden tanımlama
modeller = {
    'Lojistik Regresyon': LogisticRegression(max_iter=1000),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=SEED),
    'CatBoost': CatBoostClassifier(verbose=0, random_state=SEED),
    'LightGBM': LGBMClassifier(random_state=SEED),
    'Random Forest': RandomForestClassifier(random_state=SEED),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
}

# SMOTE ile tüm modellerin eğitimi ve değerlendirmesi
sonuclar_smote = []  # Performans sonuçları
roc_egrileri_smote = {}  # ROC eğrileri

for name, model in modeller.items():
    model.fit(X_train_smote, y_train_smote)  # SMOTE ile dengelenmiş veri setiyle eğitim
    y_pred_smote = model.predict(X_test)  # Test seti tahminleri
    y_pred_proba_smote = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    fpr_smote, tpr_smote, _ = roc_curve(y_test, y_pred_proba_smote) if y_pred_proba_smote is not None else (None, None, None)
    
    # Performans metrikleri
    metrikler = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred_smote),
        'F1-Score': f1_score(y_test, y_pred_smote),
        'Precision': precision_score(y_test, y_pred_smote),
        'Recall': recall_score(y_test, y_pred_smote),
        'AUC': auc(fpr_smote, tpr_smote) if fpr_smote is not None else None,
        'Kappa': cohen_kappa_score(y_test, y_pred_smote),
        'ROC_Curve': (fpr_smote, tpr_smote)
    }
    
    sonuclar_smote.append(metrikler)
    if fpr_smote is not None:
        roc_egrileri_smote[name] = (fpr_smote, tpr_smote)

# Performans sonuçlarını görselleştirme
sonuclar_smote_df = pd.DataFrame(sonuclar_smote)

print("\nSMOTE ile Dengelenmiş Modellerin Performansı:")
print(sonuclar_smote_df)

# ROC eğrilerinin çizimi
plt.figure(figsize=(12, 8))
for name, (fpr_smote, tpr_smote) in roc_egrileri_smote.items():
    plt.plot(fpr_smote, tpr_smote, label=f'{name}')
plt.plot([0, 1], [0, 1], 'k--')  # Rastgele tahmin çizgisi
plt.title('ROC Eğrileri (SMOTE)')
plt.xlabel('Yanlış Pozitif Oranı')
plt.ylabel('Doğru Pozitif Oranı')
plt.legend()
plt.show()

# EfficientNet için Giriş Verisinin Hazırlanması:
# EfficientNet için giriş boyutunu 64x64 olarak ayarlıyoruz çünkü epoch etme süresi çok uzun sürüyordu
def reshape_for_efficientnet(X, target_size=(64, 64)):
    
    # EfficientNet giriş formatına uygun hale getirme fonksiyonu
    # PCA çıktısını 64x64 RGB formatına dönüştürmek için kullanıyoruz
    reshaped_data = np.zeros((X.shape[0], target_size[0], target_size[1], 3))  # Hedef boyut
    for i in range(X.shape[0]):
        # Tek boyutlu PCA verisini yeniden şekillendirir
        single_channel = X[i].reshape(-1, 1)  # Tek boyutlu veri
        # Eksik boyutları sıfırlarla doldurur
        single_channel = np.pad(single_channel, ((0, target_size[0] * target_size[1] - len(single_channel)), (0, 0)),mode='constant', constant_values=0)
        single_channel = single_channel[:target_size[0] * target_size[1]].reshape(target_size[0], target_size[1])
        # Genişletilmiş veriyi RGB formatına dönüştürür
        reshaped_data[i] = np.stack([single_channel] * 3, axis=-1)
    return reshaped_data

# Orijinal veriyi EfficientNet için uygun forma getirme
X_train_efficient = reshape_for_efficientnet(X_train.values)
X_test_efficient = reshape_for_efficientnet(X_test.values)

# Veri artırma işlemi (Data Augmentation):
# Not: Eğitim verisinin çeşitliliğini artırmak için çeşitli augmentasyonlar kullanıyoruz.
datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Tüm pikselleri 0-1 aralığına normalize ediyoruz
    rotation_range=20,  # Rastgele dönüş
    width_shift_range=0.2,  # Genişlik kaydırma
    height_shift_range=0.2,  # Yükseklik kaydırma
    shear_range=0.2,  # Kesme
    zoom_range=0.2,  # Yakınlaştırma
    horizontal_flip=True  # Yatay çevirme
)

# EfficientNet modelinin tanımlanması
efficientnet_base = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(64, 64, 3), pooling='avg')
efficientnet_base.trainable = False  # Önceden eğitilmiş ağırlıkları donduruyoruz

# Model mimarisi
efficientnet_model = Sequential([
      efficientnet_base,  # EfficientNet tabanı
    Dense(128, activation='relu'),  # İlk ek katman
    Dropout(0.3),  # Aşırı öğrenmeyi önlemek için
    Dense(64, activation='relu'),  # Daha küçük bir gizli katman
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary sınıflandırma çıktısı
])

# Model derleme
efficientnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Modelin Eğitimi
efficientnet_history = efficientnet_model.fit(
    datagen.flow(X_train_efficient, y_train, batch_size=32),  # Augmentation ile eğitim
    validation_data=datagen.flow(X_test_efficient, y_test, batch_size=32),  # Validation için augmentasyon
    epochs=10,
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],  # Erken durdurma ile eğitim
    verbose=1
)

# Performans Değerlendirme

# Test seti üzerindeki tahminler
efficientnet_y_pred_proba = efficientnet_model.predict(X_test_efficient/ 255.0).ravel()
efficientnet_y_pred = (efficientnet_y_pred_proba > 0.5).astype(int)

# Değerlendirme metrikleri
efficientnet_metrikler = {
    'Model': 'EfficientNetB0 (64x64)',
    'Accuracy': accuracy_score(y_test, efficientnet_y_pred),
    'F1-Score': f1_score(y_test, efficientnet_y_pred),
    'Precision': precision_score(y_test, efficientnet_y_pred),
    'Recall': recall_score(y_test, efficientnet_y_pred),
    'AUC': roc_auc_score(y_test, efficientnet_y_pred_proba),
    'Kappa': cohen_kappa_score(y_test, efficientnet_y_pred)
}

# EfficientNetB0 Performansı
# Sonuçları yazdırma bölümü
print("\nEfficientNetB0 Performansı (64x64):")
print(efficientnet_metrikler)

# Derin Öğrenme Modelleri (MLP ve CNN ve LSTM):
# Farklı mimarileri deniyoruz ve eğitim performanslarını değerlendiriyoruz.
mlp_model = Sequential([
    Input(shape=(X_train_pca.shape[1],)),  # PCA'dan gelen özellik sayısı giriş boyutu oluyor
    Dense(128, activation='relu'),  # İlk katman, klasik relu aktivasyonu
    Dropout(0.3),  # Aşırı öğrenmeyi önlemek için Dropout
    Dense(64, activation='relu'),  # Daha küçük boyutlu bir gizli katman
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Çıkış katmanı (binary classification için sigmoid aktivasyonu)
])

# Modeli derleme bölümü
mlp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
mlp_history = mlp_model.fit(X_train_pca, y_train, validation_data=(X_test_pca, y_test),epochs=20, batch_size=64, callbacks=[EarlyStopping(patience=3)])

# Eğitim sonrası değerlendirme bölümü
mlp_y_pred_proba = mlp_model.predict(X_test_pca).ravel()
mlp_metrics = {
    'Model': 'MLP',
    'Accuracy': mlp_history.history['val_accuracy'][-1],
    'F1-Score': f1_score(y_test, (mlp_y_pred_proba > 0.5).astype(int)),
    'Precision': precision_score(y_test, (mlp_y_pred_proba > 0.5).astype(int)),
    'Recall': recall_score(y_test, (mlp_y_pred_proba > 0.5).astype(int)),
    'AUC': roc_auc_score(y_test, mlp_y_pred_proba),
    'Kappa': cohen_kappa_score(y_test, (mlp_y_pred_proba > 0.5).astype(int))
}

# CNN Modelinin Tanımlanması:
# Veriyi CNN için yeniden şekillendiriyoruz
X_train_cnn = X_train_pca.reshape(-1, X_train_pca.shape[1], 1, 1)
X_test_cnn = X_test_pca.reshape(-1, X_test_pca.shape[1], 1, 1)

cnn_model = Sequential([
     Input(shape=(X_train_pca.shape[1], 1, 1)),  # CNN girişi
    Conv2D(32, kernel_size=(2, 1), activation='relu'),  # İlk konvolüsyon katmanı
    MaxPooling2D(pool_size=(2, 1)),  # Havuzlama katmanı
    Flatten(),  # Düzleştirme (Fully Connected Layers için hazırlık)
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary sınıflandırma çıkışı
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# CNN Model Eğitim
cnn_history = cnn_model.fit(X_train_cnn, y_train, validation_data=(X_test_cnn, y_test),epochs=20, batch_size=64, callbacks=[EarlyStopping(patience=3)])

# CNN Tahminleri
cnn_y_pred_proba = cnn_model.predict(X_test_cnn).ravel()
cnn_metrics = {
    'Model': 'CNN',
    'Accuracy': cnn_history.history['val_accuracy'][-1],
    'F1-Score': f1_score(y_test, (cnn_y_pred_proba > 0.5).astype(int)),
    'Precision': precision_score(y_test, (cnn_y_pred_proba > 0.5).astype(int)),
    'Recall': recall_score(y_test, (cnn_y_pred_proba > 0.5).astype(int)),
    'AUC': roc_auc_score(y_test, cnn_y_pred_proba),
    'Kappa': cohen_kappa_score(y_test, (cnn_y_pred_proba > 0.5).astype(int))
}

# LSTM Modelinin Tanımlanması:
lstm_model = Sequential([
    Input(shape=(X_train_pca.shape[1], 1)),  # LSTM giriş boyutları (örnek sayısı, özellik sayısı)
    LSTM(128, activation='tanh', return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary sınıflandırma için
])

# Modelin derlenmesi
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Verinin Şekillendirilmesi
X_train_lstm = X_train_pca.reshape(-1, X_train_pca.shape[1], 1)
X_test_lstm = X_test_pca.reshape(-1, X_test_pca.shape[1], 1)

# Modelin Eğitilmesi
lstm_history = lstm_model.fit(X_train_lstm, y_train, validation_data=(X_test_lstm, y_test),epochs=20, batch_size=64, callbacks=[EarlyStopping(patience=3)])

# LSTM Tahminleri
lstm_y_pred_proba = lstm_model.predict(X_test_lstm).ravel()
lstm_metrics = {
    'Model': 'LSTM',
    'Accuracy': lstm_history.history['val_accuracy'][-1],
    'F1-Score': f1_score(y_test, (lstm_y_pred_proba > 0.5).astype(int)),
    'Precision': precision_score(y_test, (lstm_y_pred_proba > 0.5).astype(int)),
    'Recall': recall_score(y_test, (lstm_y_pred_proba > 0.5).astype(int)),
    'AUC': roc_auc_score(y_test, lstm_y_pred_proba),
    'Kappa': cohen_kappa_score(y_test, (lstm_y_pred_proba > 0.5).astype(int))
}

# Transfer Öğrenme (EfficientNetB0)
efficientnet_model = EfficientNetB0(weights=None, input_shape=(224, 224, 3), classes=1)
efficientnet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# MLP, CNN , LSTM ve EfficientNetB0 için Loss ve Accuracy Grafikleri
def plot_loss_accuracy(history, model_name):
    
    # Verilen modelin eğitim ve doğrulama kayıplarını ve doğruluklarını çizer.
    # param history: Keras history objesi (model eğitim geçmişi)
    # param model_name: Modelin ismi
    plt.figure(figsize=(12, 5))
    
    # Doğruluk Grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.title(f'{model_name} Doğruluk Grafiği')
    plt.xlabel('Epochs')
    plt.ylabel('Doğruluk')
    plt.legend()

    # Kayıp Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.title(f'{model_name} Kayıp Grafiği')
    plt.xlabel('Epochs')
    plt.ylabel('Kayıp')
    plt.legend()

    plt.tight_layout()
    plt.show()

# MLP için grafikler
plot_loss_accuracy(mlp_history, "MLP")

# CNN için grafikler
plot_loss_accuracy(cnn_history, "CNN")

# LSTM için grafikler
plot_loss_accuracy(lstm_history, "LSTM")

# EfficientNetB0 için grafikler
plot_loss_accuracy(efficientnet_history, "EfficientNetB0 (64x64)")

# EfficientNetB0 sonuçlarını diğer modellerle birleştirme işlemi
# EfficientNet modelinin SMOTE sonrası değerlendirme sonuçlarını mevcut sonuçlara ekliyoruz.
sonuclar_smote.append(efficientnet_metrikler)
sonuclar_smote_df = pd.DataFrame(sonuclar_smote)  # Tüm modellerin sonuçlarını DataFrame formatında birleştiriyoruz.

# En İyi Modeli Kaydetme
# En iyi modelin adını Accuracy skoruna göre belirliyoruz.
best_model_name = max(sonuclar_smote, key=lambda x: x['Accuracy'])['Model']

# Seçilen modeli diske kaydediyoruz, böylece ileride yüklenebilir.
joblib.dump(modeller[best_model_name], 'en_iyi_model.pkl')

# Yeni Örnek Tahmini
def yeni_veri_tahmin(new_data):
    # Yeni gelen örnekleri tahmin etmek için en iyi modeli kullanır.
    # Veriyi standartlaştırır, PCA ile boyutunu azaltır ve tahmin eder.
    best_model = joblib.load('best_model.pkl')  # SMOTE sonrası en iyi modeli yükleme
    new_data_scaled = scaler.transform(new_data)  # Yeni veriyi ölçeklendirme
    new_data_pca = pca.transform(new_data_scaled)  # Yeni veriyi PCA ile dönüştürme
    return best_model.predict(new_data_pca)  # Tahmin sonucu döndürme

# Sonuçları Birleştirme ve Görselleştirme
# Daha önceki tüm sonuçları (MLP, CNN ve LSTM dahil) SMOTE sonrası sonuç DataFrame'ine ekliyoruz.
sonuclar_smote.extend([mlp_metrics, cnn_metrics, lstm_metrics])
sonuclar_smote_df = pd.DataFrame(sonuclar_smote)  # Yeni bir DataFrame oluşturuyoruz.

# SMOTE sonrası tüm sonuçların DataFrame formatındaki tablosunu yazdırıyoruz.
print(sonuclar_smote_df)

# En iyi modeli seçmek için bir fonksiyon yazıyoruz
def get_best_model(sonuclar_df, metric='Accuracy'):
    
    # Belirli bir metriğe göre en iyi modeli seçer.
    # Varsayılan olarak 'Accuracy' metriği kullanılır.
    
    best_model = sonuclar_df.loc[sonuclar_df[metric].idxmax()]
    return best_model

# En iyi modelin belirlenmesi (Default olarak Accuracy kullanılır)
best_model = get_best_model(sonuclar_smote_df)  # En iyi modeli SMOTE sonrası sonuçlardan belirliyoruz.

# En iyi modelin performansı
# Seçilen modelin performans sonuçlarını yazdırıyoruz.
print("\nEn İyi Model Performans Sonuçları (SMOTE):")
print(best_model)

# En iyi modelin görselleştirilmesi
# Modellerin Accuracy ve F1-Score performanslarını barplot olarak görselleştiriyoruz.
plt.figure(figsize=(10, 6))

# Accuracy için barplot
sns.barplot(
    data=sonuclar_smote_df, 
    x='Model', 
    y='Accuracy', 
    order=sonuclar_smote_df.sort_values(by='Accuracy', ascending=False)['Model'], 
    color='blue', 
    alpha=0.6, 
    label='Accuracy'
)

# F1-Score için ikinci barplot
sns.barplot(
    data=sonuclar_smote_df, 
    x='Model', 
    y='F1-Score', 
    order=sonuclar_smote_df.sort_values(by='Accuracy', ascending=False)['Model'], 
    color='orange', 
    alpha=0.6, 
    label='F1-Score'
)

# En iyi model çizgisini belirliyoruz
plt.axhline(y=best_model['Accuracy'], color='r', linestyle='--', label=f"En İyi Model Accuracy: {best_model['Model']}")
plt.axhline(y=best_model['F1-Score'], color='g', linestyle='--', label=f"En İyi Model F1-Score: {best_model['Model']}")

# Grafik başlık ve etiketleri
plt.title('Modellerin Accuracy ve F1-Score Karşılaştırması (SMOTE)')  
plt.ylabel('Değer')  # Y ekseni etiketi
plt.xlabel('Model')  # X ekseni etiketi
plt.xticks(rotation=45)  # X eksenindeki yazıların açısını ayarlama
plt.legend()  
plt.tight_layout()  # Grafik düzenlemesi
plt.show()