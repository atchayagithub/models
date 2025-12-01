import cv2
import numpy as np

# Load image
img = cv2.imread("image1.jpg", cv2.IMREAD_GRAYSCALE)

# Create SIFT object
sift = cv2.SIFT_create()

# 1) Constructing Scale Space + 2) Keypoint Localization
# -------------------------------------------------------
# detect keypoints in different scales (octaves)
keypoints = sift.detect(img, None)

# 3) Orientation Assignment + 4) Keypoint Descriptors
# -------------------------------------------------------
# compute orientation for each keypoint and build feature descriptor
keypoints, descriptors = sift.compute(img, keypoints)

# Draw keypoints on image
out = cv2.drawKeypoints(
    img, keypoints, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

print("Number of keypoints detected:", len(keypoints))
print("Descriptor shape:", descriptors.shape)

# Display Output
cv2.imshow("SIFT Keypoints", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
