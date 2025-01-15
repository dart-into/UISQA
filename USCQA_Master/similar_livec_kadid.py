import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torchvision import models, transforms
import torch
from PIL import Image
import os
import cv2
import pandas as pd
#import piq
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances_argmin_min

image_folder1 = '/home/user/data/LIVEwild/Images/'  
image_folder2 = '/home/user/data/Kadid/kadid10k/Images/'  
num_clusters = 30  

def extract_color_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([image], [0, 1, 2], None, [9, 9, 9], [0, 256, 0, 256, 0, 256])
    hist = hist.flatten()
    hist = hist / hist.sum()
    return hist

def process_dataset(image_folder):
    image_paths = []
    features = []
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('png', 'bmp', 'jpg', 'jpeg', 'JPG')):
            image_path = os.path.join(image_folder, filename)
            features.append(extract_color_features(image_path))
            image_paths.append(filename)
    return image_paths, np.array(features)

image_paths1, features_file1 = process_dataset(image_folder1)
image_paths2, features_file2 = process_dataset(image_folder2)

kmeans_file1 = KMeans(n_clusters=10, random_state=42)
kmeans_file2 = KMeans(n_clusters=50, random_state=42)

kmeans_file1.fit(features_file1)
kmeans_file2.fit(features_file2)

labels1 = kmeans_file1.labels_
labels2 = kmeans_file2.labels_

centroids1 = kmeans_file1.cluster_centers_
centroids2 = kmeans_file2.cluster_centers_

centroids_df1 = pd.DataFrame(centroids1, columns=[f"Feature_{i}" for i in range(centroids1.shape[1])])

centroids_df1.to_csv("his1.csv", index=False)

centroids_df2 = pd.DataFrame(centroids2, columns=[f"Feature_{i}" for i in range(centroids2.shape[1])])

centroids_df2.to_csv("his2.csv", index=False)


file1_labels = kmeans_file1.labels_
file2_labels = kmeans_file2.labels_

def niqe(image1, image2):
    i_mean_1 = np.mean(image1, axis=0)
    i_mean_2 = np.mean(image2, axis=0)
    
    diff_mean = i_mean_1 - i_mean_2
    diff_mean = np.reshape(diff_mean, (27, 1))
    diff_t = np.transpose(diff_mean)
    #diff_t = np.reshape(diff_t, (1, 27))
    
    i_cov_1 = np.cov(image1, rowvar=False)
    i_cov_2 = np.cov(image2, rowvar=False)
    cov = (i_cov_1 + i_cov_2)/2
    cov = np.linalg.inv(cov)
    
    NIQE = diff_t@cov@diff_mean
    NIQE = pow(NIQE, 0.5)
    
    return NIQE

def compute_distances(features, centers):
    distances = np.linalg.norm(features - centers[:, np.newaxis], axis=2)
    return distances

distances_file2 = compute_distances(features_file2, kmeans_file2.cluster_centers_)

alpha=2
filtered_file2_indices = []
for cluster_id in range(kmeans_file2.n_clusters):
    
    cluster_samples = features_file2[clusters == cluster_id]

    cluster_mean = np.mean(cluster_samples, axis=0)
    cluster_std = np.std(cluster_samples, axis=0)

    threshold = cluster_mean + alpha * cluster_std
    
    cluster_indices = np.where(file2_labels == cluster_id)[0]
    cluster_distances = distances_file2[cluster_id, cluster_indices]

    close_indices = cluster_indices[cluster_distances < threshold]
    filtered_file2_indices.extend(close_indices)

#filtered_file2_features = image_paths2[filtered_file2_indices]
filtered_file2_labels = file2_labels[filtered_file2_indices]
#print(filtered_file2_labels.shape)

df1 = pd.DataFrame({'image_name': image_paths1, 'cluster': file1_labels})
df2 = pd.DataFrame({'image_name': image_paths2, 'cluster': filtered_file2_labels})
#df3 = pd.DataFrame({'image_name': image_paths2, 'cluster': file2_labels})

df1.to_csv('clustered_images_dataset1.csv', index=False)
df2.to_csv('clustered_images_dataset2.csv', index=False)
#df3.to_csv('clustered_images_dataset3.csv', index=False)

similarity_matrix = cosine_similarity(kmeans_file1.cluster_centers_, kmeans_file2.cluster_centers_)

#print(similarity_matrix)

best_match_file2 = np.argmax(similarity_matrix, axis=1)

for i, match in enumerate(best_match_file2):
    print(f"file1  cluster {i} close to file2  cluster {match} similarity: {similarity_matrix[i, match]}")



