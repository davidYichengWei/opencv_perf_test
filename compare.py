import sys
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def compare_images(image_name):
    # Construct file paths
    output_dir = "output/"
    cpu_path = output_dir + "cpu_segmented_" + image_name + ".jpeg"
    gpu_path = output_dir + "gpu_segmented_" + image_name + ".jpeg"
    
    # Read images
    cpu_img = cv2.imread(cpu_path)
    gpu_img = cv2.imread(gpu_path)
    
    if cpu_img is None or gpu_img is None:
        print("Error: Could not read one or both images")
        return
    
    # Ensure images have same dimensions
    if cpu_img.shape != gpu_img.shape:
        print("Error: Images have different dimensions")
        return
    
    # Calculate PSNR
    psnr_value = psnr(cpu_img, gpu_img)
    
    # Calculate SSIM (convert to grayscale for SSIM)
    cpu_gray = cv2.cvtColor(cpu_img, cv2.COLOR_BGR2GRAY)
    gpu_gray = cv2.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
    ssim_value = ssim(cpu_gray, gpu_gray)
    
    print(f"Comparison Results for {image_name}:")
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compare_segmentation.py <image_name>")
        sys.exit(1)
        
    image_name = sys.argv[1]
    compare_images(image_name)