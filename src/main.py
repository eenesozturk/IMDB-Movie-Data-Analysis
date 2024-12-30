
from preprocess import load_and_prepare_dataset, encode_features, normalize_features, categorize_target
from knn_model import knn_model
from mlp_model import mlp_model
from naive_bayes_model import naive_bayes_model
import pandas as pd

if __name__ == "__main__":
    # Veri seti yolu
    file_path = 'dataset/IMDB_Movies_Dataset.csv'

    # Veri setini yükle ve işle
    data = load_and_prepare_dataset(file_path)

    # Kategorik öznitelikleri dönüştür
    encoded_data = encode_features(data.copy())
    encoded_data.to_csv('dataset/encoded_dataset.csv', index=False)

    # Normalize edilmiş veri kümesini oluştur
    normalized_data = normalize_features(encoded_data.copy())
    normalized_data.to_csv('dataset/normalized_dataset.csv', index=False)

    # Hedef değişkeni kategorik hale getir
    data = categorize_target(normalized_data.copy())

    # Özellikler ve hedef değişken
    X = data.drop(columns=['Average Rating', 'Average Rating Class'])
    y = data['Average Rating Class']

    # Eksik değerleri kontrol etme
    print("Eksik değer kontrolü:")
    print("X_train eksik değer sayısı:", X.isnull().sum().sum())
    print("y_train eksik değer sayısı:", y.isnull().sum())

    # Eksik değerleri doldurma
    X = X.fillna(0)
    y = y.cat.add_categories(['Unknown']).fillna('Unknown') if hasattr(y, "cat") else y.fillna('Unknown')

    # Eksik değerlerin tekrar kontrolü
    print("Eksik değer kontrolü (düzeltmeden sonra):")
    print("X_train eksik değer sayısı:", X.isnull().sum().sum())
    print("y_train eksik değer sayısı:", y.isnull().sum())

    # KNN modeli sonuçları
    knn_results = knn_model(X, y)
    knn_results_df = pd.DataFrame(knn_results)
    knn_results_df['Yapılandırma'] = ['k=3', 'k=7', 'k=11']
    knn_results_df['Algorithm'] = 'KNN'

    # MLP modeli sonuçları
    mlp_results = mlp_model(X, y)
    mlp_results_df = pd.DataFrame(mlp_results, columns=['Yapılandırma', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
    mlp_results_df['Algorithm'] = 'MLP'

    # Naive Bayes modeli sonuçları
    nb_results = naive_bayes_model(X, y)
    nb_results_df = pd.DataFrame(nb_results, columns=['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
    nb_results_df['Yapılandırma'] = 'Varsayılan'

    # Tüm sonuçları birleştir
    all_results = pd.concat([knn_results_df, mlp_results_df, nb_results_df])
    all_results = all_results[['Algorithm', 'Yapılandırma', 'Accuracy', 'Precision', 'Recall', 'F1-Score']]
    all_results.to_csv('results/algorithm_results.csv', index=False)

    # Sonuçları göster
    print("Algoritma Sonuçları:")
    print(all_results)