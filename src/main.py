import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def add_gaussian_noise(image, var):
    row, col, ch = image.shape
    mean = 0
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_gaussian_filter(image, kernel_size, sigma):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def calculate_metrics(original, processed):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    psnr_value = psnr(original_gray, processed_gray)
    ssim_value = ssim(original_gray, processed_gray)
    return psnr_value, ssim_value

def process_image(image_path, noise_var, kernel_size, sigma):
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Görüntü yüklenemedi: {image_path}")
    
    # Görüntüyü en-boy oranını koruyarak yeniden boyutlandır
    max_dim = 3840  # 4K çözünürlük için maksimum boyut
    height, width = original_image.shape[:2]
    print(f"Orijinal görüntü boyutları: {width}x{height}")
    
    if width > height:
        scale = max_dim / width
    else:
        scale = max_dim / height
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    resized_original_image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    noisy_image = add_gaussian_noise(resized_original_image, var=noise_var)
    
    # Gaussian filtresini birden fazla kez uygula
    filtered_image = noisy_image
    for _ in range(4):  # 4 kez uygula
        filtered_image = apply_gaussian_filter(filtered_image, kernel_size=kernel_size, sigma=sigma)
    
    noisy_psnr, noisy_ssim = calculate_metrics(resized_original_image, noisy_image)
    filtered_psnr, filtered_ssim = calculate_metrics(resized_original_image, filtered_image)
    
    return resized_original_image, noisy_image, filtered_image, noisy_psnr, noisy_ssim, filtered_psnr, filtered_ssim

def visualize_results(original, noisy, filtered, noisy_psnr, noisy_ssim, filtered_psnr, filtered_ssim):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Orijinal Görüntü')
    axs[0].axis('off')

    axs[1].imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
    axs[1].set_title(f'Gürültülü Görüntü\nPSNR: {noisy_psnr:.2f}, SSIM: {noisy_ssim:.2f}')
    axs[1].axis('off')

    axs[2].imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
    axs[2].set_title(f'Filtrelenmiş Görüntü\nPSNR: {filtered_psnr:.2f}, SSIM: {filtered_ssim:.2f}')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

def save_image(filename, image):
    cv2.imwrite(filename, image)

def main():
    parser = argparse.ArgumentParser(description="Gaussian Gürültü Azaltma")
    parser.add_argument("--image", type=str, default="images/sample_image.jpg", help="Giriş görüntüsünün yolu")
    parser.add_argument("--noise_var", type=float, default=0.003, help="Gaussian gürültünün varyansı")
    parser.add_argument("--kernel_size", type=int, default=9, help="Gaussian çekirdeğinin boyutu")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian çekirdeğinin standart sapması")
    args = parser.parse_args()

    try:
        results = process_image(args.image, args.noise_var, args.kernel_size, args.sigma)
    except Exception as e:
        print(f"İşlem başarısız oldu: {e}")
        sys.exit(1)

    original, noisy, filtered, noisy_psnr, noisy_ssim, filtered_psnr, filtered_ssim = results

    visualize_results(original, noisy, filtered, noisy_psnr, noisy_ssim, filtered_psnr, filtered_ssim)

    # Sonuçları kaydet
    os.makedirs("results", exist_ok=True)
    save_image("results/noisy_image.jpg", noisy)
    save_image("results/filtered_image.jpg", filtered)

    print("İşlem tamamlandı. Sonuçlar 'results' klasörüne kaydedildi.")

if __name__ == "__main__":
    main()
