import cv2
import numpy as np
M1=cv2.imread(r'C:\Users\atchaya\OneDrive\Desktop\Atchaya_passportsizephoto.jpg')
M2=cv2.imread(r'C:\Users\atchaya\OneDrive\Desktop\image.jpg')
M1=cv2.resize(M1,(512,512))
M2=cv2.resize(M2,(512,512))

Out=cv2.absdiff(M1,M2)

cv2.imshow("M1 Image",M1)
cv2.imshow("M2 Image",M2)
cv2.imshow("Absolute difference Image",Out)


cv2.waitKey(0)
cv2.destroyAllWindows()
------------------------------------------------------------------------------


import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt

# --- Step 1: Preprocess (resize to 64x128) ---
img = cv2.imread(r'C:\Users\atchaya\OneDrive\Desktop\Atchaya_passportsizephoto.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (64, 128))

# --- Step 2: Calculate Gradients (x and y) ---
Gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
Gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

# --- Step 3: Magnitude and Orientation ---
magnitude = np.sqrt(Gx**2 + Gy**2)
orientation = np.arctan2(Gy, Gx) * (180 / np.pi) % 180  # [0,180)

# --- Step 4 & 5: HOG using skimage (internally does cell/block division & normalization) ---
features, hog_image = hog(
    img,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    visualize=True
)

# --- Step 6: Display ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Input Image')
plt.imshow(img, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('HOG Features Visualization')
plt.imshow(hog_image, cmap='gray')
plt.show()

print("HOG Feature Vector Length:", len(features))



---------------------------------------------------------

import matplotlib.pyplot as plt


img = cv2.imread(r'C:\Users\atchaya\OneDrive\Desktop\Atchaya_passportsizephoto.jpg', cv2.IMREAD_GRAYSCALE)

# Contrast Stretching
r_min, r_max = np.min(img), np.max(img)
stretched = ((img - r_min) / (r_max - r_min) * 255).astype(np.uint8)

# Linear Filtering (3x3 average blur)
kernel = np.ones((3, 3), np.float32) / 9
filtered = cv2.filter2D(stretched, -1, kernel)

titles = ['Original', 'Contrast Stretched', 'Linear Filtered']
images = [img, stretched, filtered]

plt.figure(figsize=(10, 6))
for i in range(3):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], cmap='gray'), plt.title(titles[i]), plt.axis('off')
    plt.subplot(2, 3, i+4), plt.hist(images[i].ravel(), 256, [0,256]), plt.title('Histogram')
plt.tight_layout(), plt.show()


----------------------------------------------------------------------------------------

img=cv2.imread(r'C:\Users\atchaya\OneDrive\Desktop\Atchaya_passportsizephoto.jpg')
img=cv2.resize(img,(512,512))


scaled_img=cv2.resize(img,None,fx=1.5,fy=1.5,interpolation=cv2.INTER_LINEAR)


rows,cols=img.shape[:2]
rotation_matrix=cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
rotated_img=cv2.warpAffine(img,rotation_matrix,(cols,rows))

shear_factor=0.3
shear_matrix=np.float32([[1,shear_factor,0],[0,1,0]])
sheared_img=cv2.warpAffine(img,shear_matrix,(int(cols+shear_factor*rows),rows))


cv2.imshow("Original", img)
cv2.imshow("Scaled", scaled_img)
cv2.imshow("Rotated", rotated_img)
cv2.imshow("Sheared", sheared_img)


cv2.waitKey(0)
cv2.destroyAllWindows()

--------------------------------------------------------------
img = cv2.imread(r'C:\Users\atchaya\OneDrive\Desktop\Atchaya_passportsizephoto.jpg',cv2.IMREAD_GRAYSCALE)

sift=cv2.SIFT_create()

keypoints,descriptors=sift.detectAndCompute(img,None)

sift_img=cv2.drawKeypoints(
    img,keypoints,None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)


cv2.imshow('Original Image', img)
cv2.imshow('SIFT Features', sift_img)
cv2.waitKey(0)
cv2.destroyAllWindows()





----------------------------------------------------------------------------

import cv2
import os

# Path to video
video_path = 'video.mp4'

# Folder to save extracted frames
output_folder = 'frames'
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # stop when video ends

    # Save frame as image
    frame_name = os.path.join(output_folder, f'frame_{count}.jpg')
    cv2.imwrite(frame_name, frame)
    count += 1

cap.release()
print(f"Extracted {count} frames and saved to '{output_folder}'")
--------------------------------------------------------------


import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Load grayscale image ---
img = cv2.imread(r'C:\Users\atchaya\OneDrive\Desktop\Atchaya_passportsizephoto.jpg', cv2.IMREAD_GRAYSCALE)

# --- Step 2: Smoothing (Gaussian Blur) ---
blur = cv2.GaussianBlur(img, (5, 5), 1.4)

# --- Step 3: Gradient Calculation (Sobel operator) ---
Gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
Gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

# Compute gradient magnitude and direction
magnitude = np.sqrt(Gx**2 + Gy**2)
magnitude = magnitude / magnitude.max() * 255
angle = np.arctan2(Gy, Gx) * 180 / np.pi

# --- Step 4: Non-Maximum Suppression (Canny internally handles this) ---
edges = cv2.Canny(blur, 100, 200)

# --- Step 5: Display results ---
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(2, 2, 2), plt.imshow(blur, cmap='gray'), plt.title('Smoothed')
plt.subplot(2, 2, 3), plt.imshow(magnitude, cmap='gray'), plt.title('Gradient Magnitude')
plt.subplot(2, 2, 4), plt.imshow(edges, cmap='gray'), plt.title('Canny Edges')
plt.show()

-----------------------------------------------------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Load grayscale image ---
img = cv2.imread(r'C:\Users\atchaya\OneDrive\Desktop\image.jpg', cv2.IMREAD_GRAYSCALE)

# --- Step 2: Choose a seed point (you can click or fix manually) ---
seed_point = (100, 100)  # (x, y) coordinates

# --- Step 3: Set threshold for similarity ---
threshold = 10

# --- Step 4: Initialize output mask ---
segmented = np.zeros_like(img)
height, width = img.shape

# --- Step 5: Region Growing Algorithm ---
def region_growing(image, seed, threshold):
    region = []
    seed_value = int(image[seed[1], seed[0]])
    mask = np.zeros_like(image, dtype=np.uint8)
    stack = [seed]

    while stack:
        x, y = stack.pop()
        if mask[y, x] == 0:
            mask[y, x] = 255
            region.append((x, y))
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] == 0:
                        if abs(int(image[ny, nx]) - seed_value) < threshold:
                            stack.append((nx, ny))
    return mask

# --- Step 6: Apply region growing ---
segmented = region_growing(img, seed_point, threshold)

# --- Step 7: Display ---
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.subplot(1,2,2)
plt.imshow(segmented, cmap='gray')
plt.title('Region Grown Segmentation')
plt.show()
