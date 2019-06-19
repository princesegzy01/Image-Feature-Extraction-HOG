import csv
import cv2
import sys
import numpy as np
from sklearn.decomposition import PCA
import os
from os import path
import sys
import numpy as np

hog = cv2.HOGDescriptor()


# DECLARE PARAMETER
NUM_FEATURES = 10  # this is the number of features to be extracted to a the csv file


# Declare empty container to hold extracted category
category = []

# Declare empty container to hold extracted features
hogArray = []

# Loop through the dataset folder to fetch category folder
for folder in os.listdir("dataset/"):

    if(folder == ".DS_Store"):
        continue

    # Loop through each category
    for filename in os.listdir(path.join("dataset", folder)):

        # Select images which are png and jpg only
        if (filename[-3:] == "png" or filename[-3:] == "jpg"):

            # Get full image by joining
            # all the path to the image
            image = path.join("dataset", folder, filename)

            # Use open cv to read the image
            img = cv2.imread(image)

            # Resize the image to (64, 128)
            # Default for hog
            resized = cv2.resize(img, (64, 128), interpolation=cv2.INTER_AREA)

            # Compute the Hog Features
            h = hog.compute(resized)

            # Transpose the result from a vector to an array
            hogImage = h.T

            # append the category of the
            # image to a category container
            category.append(folder)

            # append the extracted features of
            # the image to a category container
            hogArray.append(hogImage)


# convert the extracted features
# from array to vector
hogArray_np = np.array(hogArray)

# Reshaped the Features to the acurrate size
reshaped_hog_Array = np.reshape(
    hogArray_np, (hogArray_np.shape[0], hogArray_np.shape[2]))


# setup PCA for dimensionality reduction
pca = PCA(n_components=NUM_FEATURES)
reduced_features = pca.fit_transform(reshaped_hog_Array)
features = reduced_features.tolist()

# Create a container to hold data to be saved into csv
csvData = []
for id, line in enumerate(features):
    newImg = line

    # Prepend the category of each image to
    # the begining of the features
    newImg.insert(0, category[id])
    csvData.append(newImg)


# Save the csv file
with open('result_data.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)

csvFile.close()

print("Done Extracting Features")
