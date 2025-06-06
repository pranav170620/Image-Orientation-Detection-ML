import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def train_and_save_model(data_file, model_file, pca_model_file, n_components):
    data = joblib.load(data_file)
    sizes = [int(key) for key in data.keys()]  # Extract sub-image sizes

    for size in sizes:
        images, labels = data[str(size)]['images'], data[str(size)]['labels']

        # Reduce dimensionality using Principal Component Analysis (PCA)
        pca = PCA(n_components=n_components)
        images_pca = pca.fit_transform(images)

        # Standardize the data (optional, but can help with model size)
        scaler = StandardScaler()
        images_scaled = scaler.fit_transform(images_pca)

        # Train the model
        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(images_scaled, labels)

        # Save the model
        joblib.dump(model, f'model.{size}.joblib', compress=3)

        # Save the PCA model
        joblib.dump(pca, f'pca_model_{size}.joblib', compress=3)

if __name__ == "__main__":
    data_file = 'prepared_data.joblib'  # Update with your prepared data file
    n_components = 30  # You can adjust the number of PCA components
    train_and_save_model(data_file, f'model.{n_components}.joblib', f'pca_model_{n_components}.joblib', n_components)