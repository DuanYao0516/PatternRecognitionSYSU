import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
from skimage import io
def pca(X, num_components):
    """
    Perform PCA (Principal Component Analysis) on the dataset X.

    Parameters:
    X (numpy.ndarray): The input data matrix where each row represents a sample and each column represents a feature.
    num_components (int): The number of principal components to return.

    Returns:
    X_reduced (numpy.ndarray): The dataset projected onto the top 'num_components' principal components.
    components (numpy.ndarray): The principal components (eigenvectors).
    explained_variance (numpy.ndarray): The amount of variance explained by each of the selected components.
    X_restored (numpy.ndarray): The dataset restored to the original dimension from the reduced dimension.
    """
    # Step 1: Mean centering the data
    X_meaned = X - np.mean(X, axis=0)
    mean = np.mean(X, axis=0)

    # Step 2: Compute the covariance matrix of the data
    covariance_matrix = np.cov(X_meaned, rowvar=False)

    # Step 3: Compute the SVD of the covariance matrix
    U, S, Vt = np.linalg.svd(covariance_matrix)

    # Step 4: Select the top 'num_components' principal components
    components = Vt[:num_components]

    # Step 5: Project the data onto the principal components
    X_reduced = np.dot(X_meaned, components.T)

    # Step 6: Compute explained variance
    explained_variance = S[:num_components] / np.sum(S)

    # Step 7: Restore the data from the reduced dimension
    X_restored = np.dot(X_reduced, components) + mean

    return X_reduced, components, explained_variance, X_restored


def plot_faces(images, num_rows, num_cols, output_path):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            face = images[i].reshape(32, 32).T
            ax.imshow(face, cmap='gray')
        ax.axis('off')
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()


def plot_eigen_faces(components, output_path):
    # 绘制并保存前49个主成分（eigenfaces）的图像
    fig, axes = plt.subplots(7, 7, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < components.shape[0]:
            eigen_face = components[i].reshape(32, 32)
            ax.imshow(eigen_face, cmap='gray')
        ax.axis('off')
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()


# def plot_recovered_faces(X_original, X_restored, num_faces, output_path):
#     # 绘制并保存原始脸与重建脸（不同降维情况下）的对比图像
#     fig, axes = plt.subplots(2, num_faces, figsize=(15, 6))
#     for i in range(num_faces):
#         original_face = X_original[i].reshape(32, 32).T
#         restored_face = X_restored[i].reshape(32, 32).T
#         axes[0, i].imshow(original_face, cmap='gray')
#         axes[0, i].axis('off')
#         axes[1, i].imshow(restored_face, cmap='gray')
#         axes[1, i].axis('off')
#     plt.axis('off')
#     plt.savefig(output_path)
#     plt.close()


def task_1_2():
    # Ensure the directories exist
    if not os.path.exists('../results/PCA'):
        os.makedirs('../results/PCA')

    # Load data
    mat_data = sio.loadmat('../data/faces.mat')
    X = mat_data['X']

    # Only take the first 1024 columns for processing
    X = X[:, :1024]

    # Save the original 100 faces (reshape to 32x32 for visualization)
    original_faces = X[:100].reshape(-1, 32, 32)
    plot_faces(original_faces, num_rows=10, num_cols=10, output_path='../results/PCA/original_faces.jpg')

    # Perform PCA on all 5000 faces
    dimensions = [10, 50, 100, 150]

    for dim in dimensions:
        _, _, _, X_restored = pca(X, num_components=dim)
        # Extract the restored first 100 faces and reshape for visualization
        X_restored_faces = X_restored[:100].reshape(-1, 32, 32)
        plot_faces(X_restored_faces, num_rows=10, num_cols=10,
                   output_path=f'../results/PCA/recovered_faces_top_{dim}.jpg')

    print("Processing complete. Check the results in the '../results/PCA' directory.")


def task_3():
    from PIL import Image

    # Load the colored image
    image_path = '../data/scenery.jpg'
    image = Image.open(image_path)
    image_data = np.array(image)

    # Get image dimensions
    height, width, channels = image_data.shape

    # Perform PCA on each channel separately
    dimensions = [10, 50, 100, 150]
    for dim in dimensions:
        restored_channels = []

        for channel in range(channels):
            # Extract the channel data
            channel_data = image_data[:, :, channel]
            reshaped_data = channel_data.reshape(height, -1)

            # Perform PCA on the channel data
            X_reduced, components, _, X_restored = pca(reshaped_data, num_components=dim)

            # Reshape the restored channel data
            restored_channel = X_restored.reshape(height, width)
            restored_channels.append(restored_channel)

        # Stack the restored channels back into an image
        restored_image_data = np.stack(restored_channels, axis=2)
        restored_image_data = np.clip(restored_image_data, 0, 255).astype(np.uint8)  # Ensure values are within valid range

        # Save the restored image
        output_path = f'../results/PCA/recovered_scenery_top_{dim}.jpg'
        restored_image = Image.fromarray(restored_image_data)
        restored_image.save(output_path)

    print("Scenery image compression and restoration complete. Check the results in the '../results/PCA' directory.")



if __name__ == "__main__":
    # 绘制49主成分，并绘制重建人脸
    task_1_2()
    # 重建彩色图像
    task_3()
