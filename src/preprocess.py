import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_prepare_dataset(file_path):
    data = pd.read_csv(file_path)

    # Gereksiz sütunları çıkar
    data = data.drop(columns=['Unnamed: 0', 'Title', 'Cast'])

    # Tarih sütunundan yıl bilgisi çıkar
    data['Release Year'] = data['Release Date'].str.extract(r'(\d{4})').astype(float)
    data = data.drop(columns=['Release Date'])

    # Süre sütununu dakika cinsine çevir
    data['Runtime'] = data['Runtime'].str.extract(r'(\d+)').astype(float)

    # Eksik değerleri doldur
    data['Writer'] = data['Writer'].fillna('Unknown')
    data['Country of Origin'] = data['Country of Origin'].fillna('Unknown')
    data['Languages'] = data['Languages'].fillna('Unknown')

    return data

def encode_features(data):
    label_encoder = LabelEncoder()
    data['Director'] = label_encoder.fit_transform(data['Director'])
    data['Writer'] = label_encoder.fit_transform(data['Writer'])
    data['Country of Origin'] = label_encoder.fit_transform(data['Country of Origin'])
    data['Languages'] = label_encoder.fit_transform(data['Languages'])
    return data

def normalize_features(data):
    scaler = StandardScaler()
    numerical_columns = ['Average Rating', 'Runtime', 'Release Year']
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    return data

def categorize_target(data):
    bins = [0, 4, 7, 10]
    labels = ['Low', 'Medium', 'High']
    data['Average Rating Class'] = pd.cut(data['Average Rating'], bins=bins, labels=labels)
    return data