import cv2
import numpy as np
# M1=cv2.imread(r'C:\Users\atchaya\OneDrive\Desktop\Atchaya_passportsizephoto.jpg')
# M2=cv2.imread(r'C:\Users\atchaya\OneDrive\Desktop\image.jpg')
# M1=cv2.resize(M1,(512,512))
# M2=cv2.resize(M2,(512,512))

# Out=cv2.absdiff(M1,M2)

# cv2.imshow("M1 Image",M1)
# cv2.imshow("M2 Image",M2)
# cv2.imshow("Absolute difference Image",Out)


# cv2.waitKey(0)
# cv2.destroyAllWindows()



# img=cv2.imread(r'C:\Users\atchaya\OneDrive\Desktop\Atchaya_passportsizephoto.jpg')
# img=cv2.resize(img,(512,512))


# scaled_img=cv2.resize(img,None,fx=1.5,fy=1.5,interpolation=cv2.INTER_LINEAR)


# rows,cols=img.shape[:2]
# rotation_matrix=cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
# rotated_img=cv2.warpAffine(img,rotation_matrix,(cols,rows))

# shear_factor=0.3
# shear_matrix=np.float32([[1,shear_factor,0],[0,1,0]])
# sheared_img=cv2.warpAffine(img,shear_matrix,(int(cols+shear_factor*rows),rows))


# cv2.imshow("Original", img)
# cv2.imshow("Scaled", scaled_img)
# cv2.imshow("Rotated", rotated_img)
# cv2.imshow("Sheared", sheared_img)


# cv2.waitKey(0)
# cv2.destroyAllWindows()


# img = cv2.imread(r'C:\Users\atchaya\OneDrive\Desktop\Atchaya_passportsizephoto.jpg',cv2.IMREAD_GRAYSCALE)

# sift=cv2.SIFT_create()

# keypoints,descriptors=sift.detectAndCompute(img,None)

# sift_img=cv2.drawKeypoints(
#     img,keypoints,None,
#     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# )


# cv2.imshow('Original Image', img)
# cv2.imshow('SIFT Features', sift_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




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



# import cv2
# import os

# # Path to video
# video_path = 'video.mp4'

# # Folder to save extracted frames
# output_folder = 'frames'
# os.makedirs(output_folder, exist_ok=True)

# # Open the video file
# cap = cv2.VideoCapture(video_path)
# count = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break  # stop when video ends

#     # Save frame as image
#     frame_name = os.path.join(output_folder, f'frame_{count}.jpg')
#     cv2.imwrite(frame_name, frame)
#     count += 1

# cap.release()
# print(f"Extracted {count} frames and saved to '{output_folder}'")
