import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image

def load_image(path):
    """Load an image from the given path."""
    try:
        # PIL kullanarak resmi yükle
        image = Image.open(path)
        # Gri tonlamaya dönüştür
        image = image.convert('L')
        # NumPy dizisine dönüştür
        image_array = np.array(image)
        # [0, 1] aralığına normalize et
        return image_array / 255.0
    except Exception as e:
        raise ValueError(f"Unable to read the image at {path}: {str(e)}")

def save_image(path, image):
    """Save the image to the given path."""
    # [0, 255] aralığına dönüştür ve uint8 tipine çevir
    image_to_save = (image * 255).astype(np.uint8)
    # PIL Image nesnesine dönüştür
    pil_image = Image.fromarray(image_to_save)
    # Kaydet
    pil_image.save(path)

def calculate_metrics(original, processed):
    """Calculate PSNR and SSIM between original and processed images."""
    psnr = peak_signal_noise_ratio(original, processed)
    ssim = structural_similarity(original, processed, data_range=1.0)
    return psnr, ssim

def get_image_dimensions(image):
    """Get the dimensions of the image."""
    return image.shape[1], image.shape[0]  # width, height
