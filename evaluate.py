from argparse import ArgumentParser
from joblib import load
import os

def evaluate(model_file, n_pixels):
    """Evaluate a model on the eval1 dataset.

    Args:
        model_file (str): Path to the model file.
        n_pixels (int): Number of pixels used to train the model.
    """

    print(f"Evaluating {model_file} with {n_pixels} pixels")

    # Check that model file is no larger than 20 MB
    if os.path.getsize(model_file) > 20 * 1024 * 1024:
        print("ERROR: Model file is larger than the allowed 20 MB limit.")
        return

    model = load(model_file)

    eval_data = load(open("eval1.joblib", "rb"))[n_pixels]
    x_test = eval_data["x_test"]
    y_test = eval_data["y_test"]

    pca = load(open(f"pca_model_{n_pixels}.joblib", "rb"))  # Load the PCA model with the correct number of components

    # Apply PCA to the test data
    x_test_pca = pca.transform(x_test)

    score = model.score(x_test_pca, y_test)
    print("Score:", score * 100, "%")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_file", type=str)
    parser.add_argument("n_pixels", type=int)
    args = parser.parse_args()
    evaluate(args.model_file, args.n_pixels)