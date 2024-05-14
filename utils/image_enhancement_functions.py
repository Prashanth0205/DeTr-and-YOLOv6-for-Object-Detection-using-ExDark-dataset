import cv2
import numpy as np 

# Histogram Equalization
def histogram_equalization(image):
    # Convert the image to Lab color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Split the image into L, a, and b channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply histogram equalization to the L channel
    l_equalized = cv2.equalizeHist(l_channel)

    # Merge the equalized L channel back with the original a and b channels
    equalized_lab_image = cv2.merge((l_equalized, a_channel, b_channel))

    # Convert the enhanced Lab image back to BGR color space
    return cv2.cvtColor(equalized_lab_image, cv2.COLOR_Lab2BGR)

#------------------------------------------------------------------------------------

# Contrast Limited Adaptive Histogram Equalization (CLAHE)
def clahe(image):
    # Convert the image to Lab color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Split the Lab image into L, a, and b channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel
    # cliplimit is 1 for LHE 
    clahe = cv2.createCLAHE(clipLimit = 2.5, tileGridSize=(8, 8))
    clipped_l_channel = clahe.apply(l_channel)

    # Merge the enhanced L channel back with the original a and b channels
    enhanced_lab_image = cv2.merge((clipped_l_channel, a_channel, b_channel))

    # Convert the enhanced Lab image back to BGR color space
    return cv2.cvtColor(enhanced_lab_image, cv2.COLOR_Lab2BGR)

#------------------------------------------------------------------------------------

# Color Balance Adjustment
def color_balance_adjustment(image):
    # Convert the image to float to perform calculations
    image_float = image.astype(np.float32)

    # Calculate the scaling factors for each channel
    r_mean = np.mean(image_float[:, :, 0])
    g_mean = np.mean(image_float[:, :, 1])
    b_mean = np.mean(image_float[:, :, 2])

    k = (r_mean + g_mean + b_mean) / 3.0

    # Adjust each channel by multiplying it with the scaling factor
    r_channel = (image_float[:, :, 0] * k / (r_mean + 1e-10))
    g_channel = (image_float[:, :, 1] * k / (g_mean + 1e-10))
    b_channel = (image_float[:, :, 2] * k / (b_mean + 1e-10))

    # Merge the adjusted channels back into an RGB image
    adjusted_image = np.stack((r_channel, g_channel, b_channel), axis=-1)

    # Clip the values to be in the valid 0-255 range
    adjusted_image = np.clip(adjusted_image, 0, 255)

    return np.uint8(adjusted_image)

#------------------------------------------------------------------------------------

# Min Max Contrast Enhancement
def min_max_contrast_enhancement(image):
    # Convert the image to a floating-point representation
    img_float = image.astype(np.float32)

    # Calculate the minimum and maximum intensity values for the entire image
    min_val = img_float.min()
    max_val = img_float.max()

    # Apply the Min-Max contrast enhancement to the entire image
    enhanced_image = 255 * (img_float - min_val) / (max_val - min_val)

    # Clip the values to the [0, 255] range
    enhanced_image = np.clip(enhanced_image, 0, 255)

    # Convert the image back to uint8 format
    enhanced_image = enhanced_image.astype(np.uint8)

    return enhanced_image
