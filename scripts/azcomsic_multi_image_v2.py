import argparse
import os
import glob
import numpy as np
from astropy.io import fits
import dask.array as da
import dask.dataframe as dd
import dask
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
import time
from matplotlib import colors
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from dask.distributed import Client, progress
global CONVERSION_GAIN, READNOISE, EVENT_THRESHOLD_SIGMA, CR_THRESHOLD_SIGMA, IMAGE_SIZE, HIST_BINS, HIST_RANGE_MIN, HIST_RANGE_MAX, LINEAR_FIT_SIGMA_MAX, LINEAR_FIT_SIGMA_MIN, HIST_PLOT_MAX, HIST_PLOT_MIN, TIME_SECTION
# Defined parameters
CONVERSION_GAIN=1.6 # e-/ADU from PTC
READNOISE=13.978 # ADU from PTC

EVENT_THRESHOLD_SIGMA = 5.5 # sigma
CR_THRESHOLD_SIGMA = 400 # sigma
IMAGE_SIZE =2048 * 1048 # pixels
HIST_BINS = 1500 # bins
HIST_RANGE_MIN = 3200 # min_range
HIST_RANGE_MAX = 65000 # max_range
LINEAR_FIT_SIGMA_MAX = 120 #emgain linear fit max
LINEAR_FIT_SIGMA_MIN = 40 #emgain linear fit min
HIST_PLOT_MAX = 10000 #histogram plot max
HIST_PLOT_MIN = 3200 #histogram plot min
GUASSIAN_FIT_WINDOW=10
TIME_SECTION = True


# Utility function for timing
def time_section(func):
    def wrapper(*args, **kwargs):
        if kwargs.get('time_sections', False):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            print(f"Section {func.__name__} took {elapsed_time:.2f} seconds")
            return result
        else:
            return func(*args, **kwargs)
    return wrapper

# Define a Gaussian function
def gaussian(x, amp, mean, stddev):
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def log_linear_model_emgain(x, slope,intercpet):
    return slope*x+intercpet

@time_section
def replace_masked_pixels_with_noise(image_data, mask, mean, stddev, time_sections=False):
    mask = da.from_array(mask, chunks=image_data.chunks)
    
    def replace_block(block, block_mask, block_info=None):
        noise = np.random.normal(mean, stddev, size=block.shape)
        block[block_mask] = noise[block_mask]
        return block

    replaced_data = da.map_blocks(replace_block, image_data, mask, dtype=image_data.dtype)
    return replaced_data

def plot_histogram(data, bin_edges, threshold,cr_threshold, mean, std, emgain,em_slope,em_intercept,output_dir,input_basename,time_sections=False):
    # Calculate the histogram
    # Plot the histogram
    linear_fit_range_max=mean + LINEAR_FIT_SIGMA_MAX * std
    linear_fit_range_min=mean + LINEAR_FIT_SIGMA_MIN * std
    plt.figure(figsize=(10, 6))
    plt.plot(bin_edges[:-1], data, label='Histogram', color='darkturquoise',alpha=0.7)
    plt.axvline(threshold, color='g', linestyle='--', label='Event Threshold')
    plt.axvline(cr_threshold, color='r', linestyle='--', label='CR Threshold')
    xrange=np.linspace(mean,linear_fit_range_max,1000)
    gaussian_fit_range = np.where((bin_edges >= mean - 3 * std) & (bin_edges <= mean + 3 * std))
    plt.plot(xrange,10**(em_slope*xrange+em_intercept), label='EM_gain Fit', color='y')
    plt.plot(gaussian_fit_range,gaussian(gaussian_fit_range, np.max(data), mean, std), label='Gaussian Fit', color='r')
    plt.axvline(linear_fit_range_max, color='g', linestyle='-', label='Linear Fit Range Max')
    plt.axvline(linear_fit_range_min, color='r', linestyle='-', label='Linear Fit Range Min') 
    # Add titles and labels
    plt.title(f'Histogram with Gaussian Fit and Threshold, \nEM Gain: {emgain:.2f}, Mean: {mean:.2f}, Stddev: {std:.2f}, Threshold: {threshold:.2f}, CR Threshold: {cr_threshold:.2f}, \nLinear Fit Range Max: {linear_fit_range_max:.2f}, Linear Fit Range Min: {linear_fit_range_min:.2f}')
    plt.xlabel('Counts (ADU)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.xlim(HIST_PLOT_MIN, HIST_PLOT_MAX)
    plt.yscale('log')

    hist_filename = os.path.join(output_dir, f'{input_basename}_histogram.png')
    # Show plot
    plt.savefig(hist_filename)


@time_section
def estimate_em_gain(mean, stddev, hist, bin_centers, time_sections=False):
    try:
        start_fit = mean + LINEAR_FIT_SIGMA_MIN * stddev
        end_fit = mean + LINEAR_FIT_SIGMA_MAX * stddev
        fit_indices = np.where((bin_centers >= start_fit) & (bin_centers <= end_fit))[0]
        fit_x = bin_centers[fit_indices]
        fit_y = hist[fit_indices]
        log_fit_y = np.log10(fit_y + 1e-10)
        p0 = [0.001,1000]
        popt, pcov = curve_fit(log_linear_model_emgain, fit_x, log_fit_y, maxfev=10000, p0=p0)
        slope,intercept = popt
        #slope, intercept = np.polyfit(fit_x, log_fit_y, 1)
        emgain = (-1 / slope)
        return emgain,slope,intercept
    except IndexError as e:
        print(f"Error in estimate_em_gain: {e}")
        return None

@time_section
def compute_gaussian_parameters(hist, bin_edges, time_sections=False):
    peak_bin_index = np.argmax(hist)
    peak_bin_value = bin_edges[peak_bin_index]
    window_size=GUASSIAN_FIT_WINDOW
    window__min = 1.5*READNOISE
    window_max = 2*READNOISE 
    start_index = max(peak_bin_index - window_size, 0)
    end_index = min(peak_bin_index + window_size + 1, len(bin_edges) - 1)
    fit_bins = bin_edges[start_index:end_index]
    fit_counts = hist[start_index:end_index - 1]

    popt, _ = curve_fit(gaussian, fit_bins[:-1], fit_counts, p0=[np.max(fit_counts), peak_bin_value, READNOISE])
    amp, mean, stddev = popt

    print(f"Fitted Gaussian parameters: amp={amp}, mean={mean}, stddev={stddev}")

    return mean, stddev

@time_section
def handle_nan_values(image_data, time_sections=False):
    return da.where(da.isnan(image_data), 0, image_data)

@time_section
def compute_histogram(dask_image_data, time_sections=False):
    data_min, data_max = da.min(dask_image_data).compute(), da.max(dask_image_data).compute()
    print(f"Data range: {data_min} to {data_max}")
    hist, bin_edges = da.histogram(dask_image_data, bins=HIST_BINS, range=(data_min, data_max))
    hist_computed, bin_edges_computed = hist.compute(), bin_edges
    return hist_computed, bin_edges_computed

@time_section
def create_mask(dask_image_data, threshold, time_sections=False):
    return dask_image_data > threshold

@time_section
def compute_mask(mask, time_sections=False):
    return mask.compute()

@time_section
def apply_dbscan(coords, eps=10, min_samples=3, time_sections=False):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    return db.labels_

@time_section
def expand_cluster_partition(df, image_data, threshold, eps, min_samples):
    coords = df[['x', 'y']].values
    all_coords = set(zip(coords[:, 0], coords[:, 1]))
    for x, y in coords:
        neighbors = [(x + dx, y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if (dx, dy) != (0, 0)]
        for nx, ny in neighbors:
            if 0 <= nx < image_data.shape[1] and 0 <= ny < image_data.shape[0] and image_data[ny, nx] > threshold:
                all_coords.add((nx, ny))
    all_coords = np.array(list(all_coords))
    new_db = DBSCAN(eps=eps, min_samples=min_samples).fit(all_coords)
    new_labels = new_db.labels_
    df_expanded = pd.DataFrame(all_coords, columns=['x', 'y'])
    df_expanded['new_label'] = new_labels
    return df_expanded

@time_section
def threshold_image_and_count_dask(image_data, event_threshold):
    binary_image = da.where(image_data > event_threshold, 1, 0).astype(np.uint8)
    count_above_threshold = da.sum(binary_image).compute()
    return binary_image, count_above_threshold

@time_section
def expand_clusters_dbscan_dask(image_data, x_coords, y_coords, labels, threshold, eps=10, min_samples=3, time_sections=False):
    df = pd.DataFrame({'x': x_coords, 'y': y_coords, 'label': labels, 'value': image_data[y_coords, x_coords]})
    ddf = dd.from_pandas(df, npartitions=10)
    expanded_dfs = ddf.map_partitions(expand_cluster_partition, image_data=image_data, threshold=threshold, eps=eps, min_samples=min_samples)
    expanded_df = expanded_dfs.compute()

    expanded_coords = expanded_df[['x', 'y']].values
    original_coords = df[['x', 'y']].values
    original_labels = df['label'].values

    tree = cKDTree(original_coords)
    _, indices = tree.query(expanded_coords)
    expanded_df['label'] = original_labels[indices]

    expanded_mask = np.zeros(image_data.shape, dtype=bool)
    expanded_mask[expanded_df['y'], expanded_df['x']] = True

    return expanded_df, expanded_mask

@time_section
def find_trailing_pixels_kdtree(expanded_df, image_data, event_threshold, time_sections=False):
    coords = expanded_df[['x', 'y']].values
    kdtree = cKDTree(coords)

    trailing_pixels = []
    trailing_mask = np.zeros(image_data.shape, dtype=bool)
    grouped = expanded_df.groupby('label')

    for label, group in grouped:
        for _, row in group.iterrows():
            x, y = int(row['x']), int(row['y'])
            trailing_coords = [(x + i, y) for i in range(1, image_data.shape[1]) if x + i < image_data.shape[1]]
            
            for tx, ty in trailing_coords:
                if image_data[ty, tx] > event_threshold:
                    dist, _ = kdtree.query((tx, ty))
                    if dist > 0:
                        trailing_pixels.append((tx, ty, label))
                        trailing_mask[ty, tx] = True
    
    return trailing_pixels, trailing_mask

@time_section
def process_image(image_data, image_header, input_basename, output_dir):
    print("Starting processing of", input_basename)
    start_time = time.time()
    
    dask_image_data = da.from_array(image_data, chunks=(1000, 1000))
    dask_image_data = handle_nan_values(dask_image_data, time_sections=TIME_SECTION)

    hist_computed, bin_edges_computed = compute_histogram(dask_image_data, time_sections=TIME_SECTION)

    gaussian_mean, gaussian_stddev = compute_gaussian_parameters(hist_computed, bin_edges_computed, time_sections=TIME_SECTION)
    threshold = gaussian_mean + CR_THRESHOLD_SIGMA * gaussian_stddev
    print(f"Chosen threshold: {threshold}")

    event_threshold = gaussian_mean + EVENT_THRESHOLD_SIGMA * gaussian_stddev
    print(f"Event threshold: {event_threshold}")

    raw_emgain,raw_slope,raw_intercept = estimate_em_gain(gaussian_mean, gaussian_stddev, hist_computed, bin_edges_computed, time_sections=TIME_SECTION)
    print(f"EMGain: {raw_emgain}")
    raw_basename=input_basename+'_raw'
    plot_histogram(data=hist_computed, bin_edges=bin_edges_computed,threshold=event_threshold,cr_threshold=threshold,mean=gaussian_mean,std=gaussian_stddev,emgain=raw_emgain,em_slope=raw_slope,em_intercept=raw_intercept,output_dir=output_dir,input_basename=raw_basename)
    _, counts_above_threshold_raw = threshold_image_and_count_dask(dask_image_data, event_threshold)

    mask = create_mask(dask_image_data, threshold, time_sections=TIME_SECTION)
    mask_computed = compute_mask(mask, time_sections=TIME_SECTION)
    y_coords, x_coords = np.nonzero(mask_computed)
    values_above_threshold = image_data[mask_computed]

    pixels_above_threshold = len(values_above_threshold)
    print(f"Number of pixels above threshold: {pixels_above_threshold}")

    coords = np.column_stack((x_coords, y_coords))
    labels = apply_dbscan(coords, time_sections=TIME_SECTION)

    unique_clusters_initial = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Number of initial clusters: {unique_clusters_initial}")

    expanded_df, expanded_mask = expand_clusters_dbscan_dask(image_data, x_coords, y_coords, labels, event_threshold, eps=10, min_samples=3, time_sections=TIME_SECTION)
    pixels_after_expansion = len(expanded_df)
    print(f"Number of pixels after expansion: {pixels_after_expansion}")

    trailing_pixels, trailing_mask = find_trailing_pixels_kdtree(expanded_df, image_data, event_threshold, time_sections=TIME_SECTION)
    pixels_after_trailing = len(trailing_pixels)
    print(f"Number of pixels after trailing: {pixels_after_trailing}")

    trailing_df = pd.DataFrame(trailing_pixels, columns=['x', 'y', 'label'])
    expanded_df = pd.concat([expanded_df, trailing_df]).drop_duplicates().reset_index(drop=True)

    final_mask = np.logical_or(expanded_mask, trailing_mask)
    unique_clusters_final = expanded_df['label'].nunique()
    print(f"Number of final clusters: {unique_clusters_final}")
    # lazy person cacualte new gaussina mean and std devaition by maxking. 
    corrected_dask_image_data = replace_masked_pixels_with_noise(dask_image_data, final_mask, gaussian_mean, gaussian_stddev, time_sections=TIME_SECTION)
    hist_computed_corrected, bin_edges_computed_corrected = compute_histogram(corrected_dask_image_data, time_sections=TIME_SECTION)
    gaussian_mean_corrected, gaussian_stddev_corrected = compute_gaussian_parameters(hist_computed_corrected, bin_edges_computed_corrected, time_sections=TIME_SECTION)
    event_threshold_corrected = gaussian_mean_corrected + EVENT_THRESHOLD_SIGMA * gaussian_stddev_corrected
    corrected_emgain,corrected_slope,corrected_intercept = estimate_em_gain(gaussian_mean_corrected, gaussian_stddev_corrected, hist_computed_corrected, bin_edges_computed_corrected, time_sections=TIME_SECTION)
    corrected_basename=input_basename+'_corrected'
    plot_histogram(data=hist_computed_corrected, bin_edges=bin_edges_computed_corrected,threshold=event_threshold_corrected,cr_threshold=threshold,mean=gaussian_mean_corrected,std=gaussian_stddev_corrected,emgain=corrected_emgain,em_slope=corrected_slope,em_intercept=corrected_intercept,output_dir=output_dir,input_basename=corrected_basename)
    print(f"Event threshold: {event_threshold_corrected}")
    binary_image, count_above_threshold = threshold_image_and_count_dask(corrected_dask_image_data, event_threshold_corrected)

    df_above_threshold = pd.DataFrame({'X': expanded_df['x'], 'Y': expanded_df['y'], 'Cluster': expanded_df['label']})
    csv_filename = os.path.join(output_dir, f'{input_basename}_pixels_above_threshold_with_clusters.csv')
    exptime = image_header['EXPTIME']
    vss= image_header['VSS']
    with open(csv_filename, 'w') as f:
        f.write(f'# Input_file: {input_basename}\n')
        f.write(f'# Exptime: {exptime}\n')
        f.write(f'# Total_pixels: {np.size(image_data)}\n')
        f.write(f'# CR_threshold: {threshold}\n')
        f.write(f'# Event_threshold: {event_threshold}\n')
        f.write(f'# Gaussian_mean: {gaussian_mean}\n')
        f.write(f'# Gaussian_stddev: {gaussian_stddev}\n')
        f.write(f'# EMGain_Raw: {raw_emgain}\n')
        f.write(f'# VSS: {vss}\n')
        f.write(f'# Events_Raw: {counts_above_threshold_raw}\n')
        f.write(f'# Number_of_unique_clusters: {unique_clusters_initial}\n')
        f.write(f'# Total_Pix_above_CR_Threshold: {pixels_above_threshold}\n')
        f.write(f'# Total_Pix_after_CR_tails: {pixels_after_expansion}\n')
        f.write(f'# Total_Pix_CR_Neighbours_tails: {pixels_after_trailing}\n')
        f.write(f'# Gaussian_mean_corrected: {gaussian_mean_corrected}\n')
        f.write(f'# Number_of_unique_clusters_trails: {unique_clusters_final}\n')
        f.write(f'# Gaussian_stddev_corrected: {gaussian_stddev_corrected}\n')
        f.write(f'# Event_Threshold_Corrected: {event_threshold_corrected}\n')
        f.write(f'# Events_Corrected: {count_above_threshold}\n')
        f.write(f'# EMGain_Corrected: {corrected_emgain}\n')
        df_above_threshold.to_csv(f, index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")

@time_section
def process_all_images(input_folder, output_folder):
    fits_files = sorted(glob.glob(os.path.join(input_folder, '*.fits')))

    client = Client()
    print(f"Processing {len(fits_files)} FITS files")

    futures = []
    for fits_file in fits_files:
        input_basename = os.path.splitext(os.path.basename(fits_file))[0]
        output_dir = output_folder
        os.makedirs(output_dir, exist_ok=True)
        
        image_data = fits.getdata(fits_file)[250:1250,1072+500:1072+1000]
        IMAGE_SIZE=image_data.shape[0]*image_data.shape[1]
        image_header = fits.getheader(fits_file)

        # Scatter the data to the workers
        scattered_image_data = client.scatter(image_data)
        scattered_image_header = client.scatter(image_header)
        
        futures.append(client.submit(process_image, scattered_image_data, scattered_image_header, input_basename, output_dir))

    results = client.gather(futures)
    client.close()
    return results

def main(input_folder, output_folder):
    results = process_all_images(input_folder, output_folder)
    return results
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the input folder containing data files.")
    parser.add_argument(
        '--input_folder', 
        type=str, 
        required=True, 
        help='Path to the input folder containing data files.'
    )
    
    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = os.path.join(input_folder, 'output')
    
    if not os.path.isdir(input_folder):
        raise ValueError(f"The provided path '{input_folder}' is not a valid directory.")
    else:
        print(f"Processing data in folder '{input_folder}'")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
            print(f"Output will be saved in '{output_folder}'")
    # input_folder = '/Users/arkhan/Documents/Dark_Results/dark_SPIE_examples/test_data_warm/dark_07022024_g8020_minus100_vss0'
    results = main(input_folder, output_folder)