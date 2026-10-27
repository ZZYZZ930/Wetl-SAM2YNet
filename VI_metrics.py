import numpy as np
import tifffile

def calculate_vi(image_path, vi_type='ndvi', L=0.5):
    """
    Calculate vegetation indices from a remote sensing image.
    The image is assumed to have 4 bands in the order: B, G, R, NIR.

    :param image_path: Path to the .tif image file.
    :param vi_type: Vegetation index type to compute.
                    Supported: 'ndvi', 'dvi', 'rvi', 'savi', 'ndwi', 'grvi', 'vari'.
    :param L: Soil adjustment factor for SAVI, default 0.5.
    :return: Computed vegetation index as a 2D numpy array.
    """
    try:
        # Read image using tifffile
        img = tifffile.imread(image_path)
    except FileNotFoundError:
        print(f"Error: File not found {image_path}")
        return None

    # Ensure the image has at least 4 channels
    if img.ndim != 3 or img.shape[2] < 4:
        print(f"Error: Image {image_path} requires at least 4 bands (B, G, R, NIR).")
        return None

    # Extract bands based on common order (B:0, G:1, R:2, NIR:3)
    # Convert to float32 for computation
    blue = img[:, :, 0].astype(np.float32)
    green = img[:, :, 1].astype(np.float32)
    red = img[:, :, 2].astype(np.float32)
    nir = img[:, :, 3].astype(np.float32)

    # Small epsilon to avoid division by zero
    epsilon = 1e-8

    vi_map = {
        'ndvi': lambda: (nir - red) / (nir + red + epsilon),
        'dvi': lambda: nir - red,
        'rvi': lambda: nir / (red + epsilon),
        'savi': lambda: ((nir - red) / (nir + red + L + epsilon)) * (1 + L),
        'ndwi': lambda: (green - nir) / (green + nir + epsilon),
        'grvi': lambda: (green - red) / (green + red + epsilon),
        'vari': lambda: (green - red) / (green + red - blue + epsilon)
    }

    if vi_type.lower() in vi_map:
        return vi_map[vi_type.lower()]()
    else:
        print(f"Error: Unsupported vegetation index type '{vi_type}'.")
        return None

if __name__ == '__main__':
    # Example vegetation-index calculation for a four-band training image.
    train_image_path = 'img_datas/train/image/2_image.tif'

    # Compute all supported vegetation indices
    supported_vis = ['ndvi', 'dvi', 'rvi', 'savi', 'ndwi', 'grvi', 'vari']
    for vi in supported_vis:
        print(f"Computing {vi.upper()}...")
        vi_result = calculate_vi(train_image_path, vi_type=vi)
        if vi_result is not None:
            print(f"{vi.upper()} done. Result shape: {vi_result.shape}, dtype: {vi_result.dtype}")
        print("-" * 20)
