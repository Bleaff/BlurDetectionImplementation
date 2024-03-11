import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def laplacian_variance(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def gradient_magnitude(image):
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return np.mean(gradient_magnitude)

def frequency_based(image):
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_transform_shifted)
    return np.mean(magnitude_spectrum)

def edge_based(image):
    edges = cv2.Canny(image, 50, 150)
    return np.sum(edges) / float(image.size)

def is_blurred(image, threshold=100, method='laplacian_variance'):
    if method == 'laplacian_variance':
        metric = laplacian_variance(image)
    elif method == 'gradient_magnitude':
        metric = gradient_magnitude(image)
    elif method == 'frequency_based':
        metric = frequency_based(image)
    elif method == 'edge_based':
        metric = edge_based(image)
    else:
        raise ValueError("Invalid method. Choose from 'laplacian_variance', 'gradient_magnitude', 'frequency_based', 'edge_based'.")

    return metric < threshold, metric

def evaluate_accuracy_precision_recall(true_labels, predictions):
    accuracy = (np.array(predictions) == np.array(true_labels)).mean() * 100
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    return accuracy, precision, recall, f1

def test_all_methods(blurred_folder, not_blurred_folder, output_csv_path):
    thresholds = [200]
    methods = ['laplacian_variance', 'gradient_magnitude', 'frequency_based', 'edge_based']

    results = []

    for method in methods:
        for threshold in thresholds:

            blurred = [is_blurred(cv2.imread(os.path.join(blurred_folder, img), cv2.IMREAD_GRAYSCALE), threshold, method)
                           for img in os.listdir(blurred_folder)]

            not_blurred = [is_blurred(cv2.imread(os.path.join(not_blurred_folder, img), cv2.IMREAD_GRAYSCALE), threshold, method)
                           for img in os.listdir(not_blurred_folder)]

            true_blurred = [1 for i in range(len(blurred))]
            true_not_blurred = [0 for i in range(len(not_blurred))]
            mean_bl = [el[1] for el in blurred]
            mean_nbl = [el[1] for el in not_blurred]
            blurred = [el[0] for el in blurred]
            not_blurred = [el[0] for el in not_blurred]
            predictions = blurred
            predictions.extend(not_blurred)
            true_blurred.extend(true_not_blurred)
            true_labels = true_blurred
            print(len(true_labels), len(predictions))
            accuracy, precision, recall, f1 = evaluate_accuracy_precision_recall(true_labels, predictions)

            results.append({
                'Method': method,
                'Threshold': threshold,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'Mean_blurred': sum(mean_bl) / len(mean_bl),
                'Mean_not_blurred': sum(mean_nbl)/len(mean_nbl)
            })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)

    # Plot results
    plot_results(results_df)

def plot_results(results_df):
    methods = results_df['Method'].unique()

    for method in methods:
        method_data = results_df[results_df['Method'] == method]
        plt.plot(method_data['Threshold'], method_data['Accuracy'], label=f'{method} Accuracy')

    plt.xlabel('Threshold')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy for Different Methods and Thresholds')
    plt.show()

# Example usage
blurred_folder = "/Users/bleaf/Desktop/Job/Blur/fin_dataset_last/blur/train"
not_blurred_folder = "/Users/bleaf/Desktop/Job/Blur/fin_dataset_last/clear/train"
output_csv_path = "results.csv"

test_all_methods(blurred_folder, not_blurred_folder, output_csv_path)

