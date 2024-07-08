import numpy as np
from skimage import io
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_gaussian

# Tek bir görüntü yükle
image = io.imread('sample_image.jpg')

# Görüntüyü float tipine dönüştür ve normalize et (0-1 aralığı)
image = image.astype(np.float32) / 255.0

# Gürültü azaltma işlemi uygula
denoised_image = denoise_gaussian(image)

# Orijinal ve gürültüsü azaltılmış görüntüler arasındaki benzerliği hesapla
data_range = 1.0  # Görüntüler 0-1 aralığında normalize edildiği için
similarity = ssim(image, denoised_image, data_range=data_range, channel_axis=-1)

print(f"Orijinal ve gürültüsü azaltılmış görüntü arasındaki yapısal benzerlik: {similarity}")
