import pickle

import cv2
import numpy as np

image_path = r'C:\PycharmProjects\ExpertInformedDL\source_attention\only green RLS_036_OS_TC.png'
out_path = r'C:\PycharmProjects\ExpertInformedDL\source_attention\only green RLS_036_OS_TC.pickle'

# load the image and convert it to grayscale and save as pickle
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# convert ndarray
image = np.asarray(image)

# normalize and convert to float type
image = image.astype(np.float32)
image = image / 255.0

# save as pickle
pickle.dump(image, open(out_path, 'wb'))