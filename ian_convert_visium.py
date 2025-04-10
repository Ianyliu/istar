import os
import glob
import tarfile
import tifffile
import cv2
import pandas as pd
from scipy.io import mmread
import json
import math
import shutil
import gzip

def gunzip_folder(folder_path):
    gz_files = glob.glob(folder_path + '/*/*.gz')
    for gz_file in gz_files:
        out_path = gz_file[:-3]  # remove ".gz"
        with gzip.open(gz_file, 'rb') as f_in:
            with open(out_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(gz_file)

def process_visium_folder(folder_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Convert image.tif to jpg
    tif_files = glob.glob(os.path.join(folder_path, '*image.tif'))
    if not tif_files:
        raise FileNotFoundError("No image.tif file found.")
    tif = tifffile.imread(tif_files[0])
    jpg_path = os.path.join(output_dir, 'he-raw.jpg')
    cv2.imwrite(jpg_path, cv2.cvtColor(tif[:, :, 0:3], cv2.COLOR_BGR2RGB),
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # Step 2: Untar all *.tar.gz files and rename folders
    tar_files = glob.glob(os.path.join(folder_path, '*.tar.gz'))
    # for tar_path in tar_files:
    #     suffix = os.path.basename(tar_path).replace('.tar.gz', '')
    #     extract_dir = os.path.join(folder_path, suffix)
    #     os.makedirs(extract_dir, exist_ok=True)
    #     with tarfile.open(tar_path, "r:gz") as tar:
    #         tar.extractall(path=extract_dir)
    for tar_path in tar_files:
        suffix = os.path.basename(tar_path).replace('.tar.gz', '')
        extract_dir = os.path.join(folder_path, suffix)
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)

        # Gunzip inner .gz files if any
        if "feature_bc_matrix" in suffix:
            gunzip_folder(extract_dir)

    # Step 3: Generate locs_raw.tsv
    spatial_dir = glob.glob(os.path.join(folder_path, '*spatial'))[0]
    spatial_dir = os.path.join(spatial_dir, 'spatial')
    tissue_csv_path = os.path.join(spatial_dir, 'tissue_positions_list.csv')
    df = pd.read_csv(tissue_csv_path, header=None)
    df_locs = df[[0, 4, 5]]
    df_locs.columns = ['spot', 'x', 'y']
    df_locs['x'] = df_locs['x'].astype(float)
    df_locs['y'] = df_locs['y'].astype(float)
    locs_out_path = os.path.join(output_dir, 'locs-raw.tsv')
    df_locs.to_csv(locs_out_path, sep='\t', index=False)
    

    # Step 4: Create cnts.tsv
    raw_matrix_dir = glob.glob(os.path.join(folder_path, '*raw_feature_bc_matrix'))[0]
    raw_matrix_dir = os.path.join(raw_matrix_dir, 'raw_feature_bc_matrix')
    print(raw_matrix_dir)
    matrix = mmread(os.path.join(raw_matrix_dir, 'matrix.mtx')).todense()
    barcodes = pd.read_csv(os.path.join(raw_matrix_dir, 'barcodes.tsv'), header=None, sep='\t')[0]
    features = pd.read_csv(os.path.join(raw_matrix_dir, 'features.tsv'), header=None, sep='\t')[0]
    df_counts = pd.DataFrame(matrix.T, columns=features, index=barcodes)
    df_counts.index.name = 'spot'
    cnts_out_path = os.path.join(output_dir, 'cnts.tsv')
    df_counts.to_csv(cnts_out_path, sep='\t')

    # Step 5: Check that locs and cnts match index
    locs_raw = df_locs.set_index('spot')
    cns = df_counts
    common_index = locs_raw.index.intersection(cns.index)
    locs_raw_common = locs_raw.loc[common_index]
    cns_common = cns.loc[common_index]
    assert locs_raw_common.index.equals(cns_common.index), "Indices of locs and counts do not match!"

    # Step 6: Read scalefactors and compute pixel size
    scale_json_path = os.path.join(spatial_dir, 'scalefactors_json.json')
    with open(scale_json_path, 'r') as f:
        scalefactors = json.load(f)

    tissue_hires_scalef = scalefactors['tissue_hires_scalef']
    pixel_size_raw = 8000 / 2000 * tissue_hires_scalef  # => 4 * tissue_hires_scalef

    pixel_out_path = os.path.join(output_dir, 'pixel-size-raw.txt')
    with open(pixel_out_path, 'w') as f:
        f.write(str(pixel_size_raw))

    # Step 7: Compute radius from spot diameter
    spot_diameter_fullres = scalefactors['spot_diameter_fullres']
    radius_raw = spot_diameter_fullres * 0.5

    expected_radius = 55 * 0.5 / pixel_size_raw
    if not math.isclose(radius_raw, expected_radius, rel_tol=0.1):
        print(f"⚠️  Warning: Computed radius ({radius_raw:.3f}) not close to expected ({expected_radius:.3f})")

    radius_out_path = os.path.join(output_dir, 'radius-raw.txt')
    with open(radius_out_path, 'w') as f:
        f.write(str(radius_raw))

    print(f"✅ Processing complete. Outputs saved to: {output_dir}")
