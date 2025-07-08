import cv2
import numpy as np
import matplotlib.pyplot as plt

def enhance_gray_detail(img_gray, x0=125, k=0.05):
    # img_gray: uint8 grayscale image
    x = np.arange(0, 256, dtype=np.float32)
    s_curve = 255 / (1 + np.exp(-k * (x - x0)))
    s_curve = s_curve.astype(np.uint8)

    img_enhanced = cv2.LUT(img_gray, s_curve)
    return img_enhanced

# Example usage
img = cv2.imread('../assets/1.pgm', cv2.IMREAD_GRAYSCALE)

# Larger k sharpens detail around 125 but may cause ringing.
# Smaller k softens the enhancement.
enhanced_img = enhance_gray_detail(img, x0=125, k=0.08)  # Adjust k for more/less contrast

# Display
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.subplot(1,2,2)
plt.title("Enhanced")
plt.imshow(enhanced_img, cmap='gray')
plt.show()

# Optional: Save result
# cv2.imwrite('enhanced_gray_image.png', enhanced_img)
