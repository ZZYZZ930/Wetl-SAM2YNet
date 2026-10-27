import os
import torch
import numpy as np
from osgeo import gdal
from tqdm import tqdm
from networks.WetlSAM2YNet import WetlSAM2YNet
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_objects
import tifffile

# --- Configuration ---
# Model path
MODEL_PATH = 'output_pth/WetlSAM2YNet-100.pth'
# Input image directory
INPUT_DIR = 'map_img/'
# Output image directory
OUTPUT_DIR = os.path.join(INPUT_DIR, 'results')
# Model input size
INPUT_SIZE = 512
# Sliding-window overlap for spatially smooth probability aggregation.
OVERLAP = INPUT_SIZE * 3 // 4 
# Batch size during inference
BATCH_SIZE = 16
# Number of classes
NUM_CLASSES = 21
# Image normalization factor
NORMALIZATION_FACTOR = 3600.0
# NoData value
NO_DATA_VALUE = 0
# Post-processing: remove isolated regions smaller than this number of pixels
MIN_OBJECT_SIZE = 100 

# --- Helper functions ---

def read_gdal_info(path):
    """Read GDAL metadata only"""
    dataset = gdal.Open(path)
    if dataset is None:
        raise FileNotFoundError(f"Unable to open file: {path}")
    proj = dataset.GetProjection()
    geotrans = dataset.GetGeoTransform()
    nodata = dataset.GetRasterBand(1).GetNoDataValue()
    del dataset
    return proj, geotrans, nodata

def write_gdal(path, data, proj, geotrans):
    """Write image using GDAL"""
    datatype = gdal.GDT_Byte
    height, width = data.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, width, height, 1, datatype)
    dataset.SetProjection(proj)
    dataset.SetGeoTransform(geotrans)
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    band.SetNoDataValue(NO_DATA_VALUE)
    band.FlushCache()
    del dataset

def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
    """Generate Gaussian weight map"""
    tmp = np.zeros((patch_size, patch_size))
    center_coords = [i // 2 for i in [patch_size, patch_size]]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigma=patch_size * sigma_scale, order=0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(gaussian_importance_map[gaussian_importance_map > 0])
    return gaussian_importance_map

def calculate_vi_from_array(image_array, vi_type='ndvi', L=0.5):
    """
    Calculate vegetation index from a loaded 4-band NumPy array.
    Assumed band order: B, G, R, NIR (0, 1, 2, 3)
    """
    blue, green, red, nir = [image_array[:, :, i].astype(np.float32) for i in range(4)]
    
    np.seterr(divide='ignore', invalid='ignore')
    if vi_type == 'rvi': vi = nir / red
    elif vi_type == 'dvi': vi = nir - red
    elif vi_type == 'savi': vi = ((nir - red) / (nir + red + L)) * (1 + L)
    elif vi_type == 'ndwi': vi = (green - nir) / (green + nir)
    else: raise ValueError(f"Unsupported vegetation index type: {vi_type}")
    vi[~np.isfinite(vi)] = 0
    return vi

# --- Main ---

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = WetlSAM2YNet(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    stride = INPUT_SIZE - OVERLAP
    gaussian_weights = torch.from_numpy(_get_gaussian(INPUT_SIZE)).to(device)

    image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.tif', '.tiff'))]

    for image_name in image_files:
        image_path = os.path.join(INPUT_DIR, image_name)
        output_path = os.path.join(OUTPUT_DIR, image_name.replace('.tif', '_pred.tif'))
        print(f"\n--- Processing image: {image_name} ---")

        try:
            print("Loading source image...")
            full_image_raw = tifffile.imread(image_path)
            proj, geotrans, nodata_val = read_gdal_info(image_path)
            
            # Use zero as background when the raster metadata has no NoData value.
            if nodata_val is None:
                nodata_val = 0
                print("Warning: NoData value not found in metadata. Assuming 0 as background.")
            
            print("Precomputing vegetation indices...")
            vi_layers = {}
            for vi_name in ['rvi', 'dvi', 'savi', 'ndwi']:
                vi_layers[vi_name] = calculate_vi_from_array(full_image_raw, vi_type=vi_name)
            
            full_image_normalized = full_image_raw / NORMALIZATION_FACTOR

        except Exception as e:
            print(f"Image preprocessing failed: {e}, skipping this file.")
            continue
        
        height, width, _ = full_image_raw.shape
        
        probs_aggregator = np.zeros((height, width, NUM_CLASSES), dtype=np.float32)
        weights_aggregator = np.zeros((height, width), dtype=np.float32)

        patch_coords = []
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                y_start, x_start = y, x
                if y + INPUT_SIZE > height: y_start = height - INPUT_SIZE
                if x + INPUT_SIZE > width: x_start = width - INPUT_SIZE
                patch_coords.append((y_start, x_start))
        
        patch_coords = sorted(list(set(patch_coords)))

        batch_main, batch_aux, batch_info = [], [], []

        with torch.no_grad():
            pbar = tqdm(total=len(patch_coords), desc=f"Smooth inference {image_name[:10]}...")
            for i, (y, x) in enumerate(patch_coords):
                raw_patch = full_image_raw[y:y+INPUT_SIZE, x:x+INPUT_SIZE, :]
                if nodata_val is not None and np.all(raw_patch == nodata_val):
                    pbar.update(1)
                    continue

                main_patch_normalized = full_image_normalized[y:y+INPUT_SIZE, x:x+INPUT_SIZE, :]
                main_input = main_patch_normalized[:, :, [3, 2, 1]] # NIR, R, G
                blue_channel = main_patch_normalized[:, :, 0]
                vi_channels_patch = [vi_layers[vi_name][y:y+INPUT_SIZE, x:x+INPUT_SIZE] for vi_name in ['rvi', 'dvi', 'savi', 'ndwi']]
                aux_input = np.stack([blue_channel] + vi_channels_patch, axis=-1)
                
                main_tensor = torch.from_numpy(main_input.astype(np.float32)).permute(2, 0, 1)
                aux_tensor = torch.from_numpy(aux_input.astype(np.float32)).permute(2, 0, 1)

                batch_main.append(main_tensor)
                batch_aux.append(aux_tensor)
                batch_info.append({'y': y, 'x': x})

                if len(batch_main) == BATCH_SIZE or i == len(patch_coords) - 1:
                    if not batch_main: continue
                    main_batch_tensor = torch.stack(batch_main).to(device)
                    aux_batch_tensor = torch.stack(batch_aux).to(device)

                    logits = model(main_batch_tensor, aux_batch_tensor)
                    probs = F.softmax(logits, dim=1)
                    weighted_probs = probs * gaussian_weights

                    for j, info in enumerate(batch_info):
                        y_start, x_start = info['y'], info['x']
                        
                        # Construct a NoData mask for the current patch.
                        original_patch_raw = full_image_raw[y_start:y_start+INPUT_SIZE, x_start:x_start+INPUT_SIZE, :]
                        nodata_mask_patch = np.all(original_patch_raw == nodata_val, axis=2)

                        # Get weighted probabilities and weights
                        patch_probs_np = weighted_probs[j].cpu().numpy().transpose(1, 2, 0)
                        patch_weights_np = gaussian_weights.cpu().numpy()
                        
                        # Zero out probabilities and weights for NoData regions
                        patch_probs_np[nodata_mask_patch] = 0
                        patch_weights_np[nodata_mask_patch] = 0

                        # Aggregate
                        probs_aggregator[y_start:y_start+INPUT_SIZE, x_start:x_start+INPUT_SIZE] += patch_probs_np
                        weights_aggregator[y_start:y_start+INPUT_SIZE, x_start:x_start+INPUT_SIZE] += patch_weights_np

                    batch_main, batch_aux, batch_info = [], [], []
                    pbar.update(len(probs))
            pbar.close()

        # --- Post-processing ---
        # Identify background regions never covered by valid pixels
        background_mask = (weights_aggregator == 0)
        
        # Avoid division by zero while keeping background information
        weights_aggregator[background_mask] = 1
        
        # Compute average probability
        avg_probs = probs_aggregator / weights_aggregator[..., np.newaxis]
        
        # Predicted class indices (0-20)
        final_pred = np.argmax(avg_probs, axis=2).astype(np.uint8)
        
        # Convert model outputs from indices 0-20 to label values 1-21
        final_pred += 1
        
        # Assign the configured NoData value to pixels without valid observations.
        final_pred[background_mask] = NO_DATA_VALUE
        
        print("Applying morphological post-processing to remove small noise...")
        # Ensure background values do not participate in morphological operations
        pred_for_cleaning = final_pred.copy()
        pred_for_cleaning[pred_for_cleaning == NO_DATA_VALUE] = 0  # The morphology routine interprets zero as background.
        cleaned_pred = remove_small_objects(pred_for_cleaning, min_size=MIN_OBJECT_SIZE, connectivity=1)
        # Restore the true NoData value
        cleaned_pred[pred_for_cleaning == 0] = NO_DATA_VALUE
        
        write_gdal(output_path, cleaned_pred.astype(np.uint8), proj, geotrans)
        print(f"Result saved to: {output_path}")

    print("\nAll images processed.")

if __name__ == '__main__':
    main()
