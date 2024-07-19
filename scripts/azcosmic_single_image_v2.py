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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

cmap = plt.cm.rainbow

# Defined parameters
EVENT_THRESHOLD_SIGMA = 5.5
CR_THRESHOLD_SIGMA = 600
IMAGE_SIZE = 2048 * 1048
HIST_BINS = 2000
HIST_RANGE_MIN = 0
HIST_RANGE_MAX = 65000
LINEAR_FIT_SIGMA_MAX = 100
LINEAR_FIT_SIGMA_MIN = 30
HIST_PLOT_MAX = 40000
HIST_PLOT_MIN = 3000
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

@time_section
def replace_masked_pixels_with_noise(image_data, mask, mean, stddev, time_sections=False):
    # Ensure mask has the same chunk size as image_data
    mask = da.from_array(mask, chunks=image_data.chunks)
    
    # Define a function to replace masked pixels within each block
    def replace_block(block, block_mask, block_info=None):
        noise = np.random.normal(mean, stddev, size=block.shape)
        block[block_mask] = noise[block_mask]
        return block

    # Use Dask's map_blocks to apply the replacement function in parallel
    replaced_data = da.map_blocks(replace_block, image_data, mask, dtype=image_data.dtype)

    return replaced_data

@time_section
def estimate_em_gain(mean, stddev, hist, bin_centers, time_sections=False):
    # Estimate EM gain by fitting a straight line to the part of the histogram
    start_fit = mean + LINEAR_FIT_SIGMA_MIN * stddev
    end_fit = mean + LINEAR_FIT_SIGMA_MAX * stddev
    fit_indices = np.where((bin_centers >= start_fit) & (bin_centers <= end_fit))[0]
    fit_x = bin_centers[fit_indices]
    fit_y = hist[fit_indices]  # Adjust to match the length of fit_x

    # Perform linear fit
    log_fit_y = np.log(fit_y + 1e-10)  # Add small value to avoid log(0)
    slope, intercept = np.polyfit(fit_x, log_fit_y, 1)
    emgain = -1 / slope
    return emgain

@time_section
def compute_gaussian_parameters(hist, bin_edges, time_sections=False):
    # Find the bin with the maximum count (peak of the histogram)
    peak_bin_index = np.argmax(hist)
    peak_bin_value = bin_edges[peak_bin_index]

    # Select a window around the peak for fitting the Gaussian
    window_size = 5  # Adjust window size as needed
    start_index = max(peak_bin_index - window_size, 0)
    end_index = min(peak_bin_index + window_size + 1, len(bin_edges) - 1)
    fit_bins = bin_edges[start_index:end_index]
    fit_counts = hist[start_index:end_index - 1]  # Adjust to match the length of fit_bins[:-1]

    # Fit the Gaussian to the selected window
    popt, _ = curve_fit(gaussian, fit_bins[:-1], fit_counts, p0=[np.max(fit_counts), peak_bin_value, np.std(fit_bins)])

    # Extract the fitted Gaussian parameters
    amp, mean, stddev = popt

    print(f"Fitted Gaussian parameters: amp={amp}, mean={mean}, stddev={stddev}")

    return mean, stddev

@time_section
def handle_nan_values(image_data, time_sections=False):
    # Handle NaN values by replacing them with zero
    return da.where(da.isnan(image_data), 0, image_data)

@time_section
def compute_histogram(dask_image_data, time_sections=False):
    # Compute the range for the histogram
    data_min, data_max = da.min(dask_image_data).compute(), da.max(dask_image_data).compute()
    print(f"Data range: {data_min} to {data_max}")
    # Compute the histogram using Dask, specify the range
    hist, bin_edges = da.histogram(dask_image_data, bins=HIST_BINS, range=(data_min, data_max))
    hist_computed, bin_edges_computed = hist.compute(), bin_edges
    return hist_computed, bin_edges_computed

@time_section
def create_mask(dask_image_data, threshold, time_sections=False):
    # Use Dask to create a mask
    return dask_image_data > threshold

@time_section
def compute_mask(mask, time_sections=False):
    # Compute the mask to get the coordinates of the pixels above the threshold
    return mask.compute()

@time_section
def apply_dbscan(coords, eps=10, min_samples=3, time_sections=False):
    # Apply DBSCAN clustering to the filtered pixels
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
    """
    Apply a threshold to the image data using Dask to create a binary image and count the number of pixels above the threshold.

    Parameters:
    - image_data: np.ndarray, the input image data
    - event_threshold: float, the threshold value to classify pixels

    Returns:
    - binary_image: dask.array.Array, the binary image with 1 for pixels above the threshold
                    and 0 for pixels below or equal to the threshold
    - count_above_threshold: int, the number of pixels above the threshold
    """
    binary_image = da.where(image_data > event_threshold, 1, 0).astype(np.uint8)
    count_above_threshold = da.sum(binary_image).compute()
    return binary_image, count_above_threshold

@time_section
def expand_clusters_dbscan_dask(image_data, x_coords, y_coords, labels, threshold, eps=10, min_samples=3, time_sections=False):
    df = pd.DataFrame({'x': x_coords, 'y': y_coords, 'label': labels, 'value': image_data[y_coords, x_coords]})
    ddf = dd.from_pandas(df, npartitions=10)
    expanded_dfs = ddf.map_partitions(expand_cluster_partition, image_data=image_data, threshold=threshold, eps=eps, min_samples=min_samples)
    expanded_df = expanded_dfs.compute()

    # Assign expanded pixels to the nearest original cluster
    expanded_coords = expanded_df[['x', 'y']].values
    original_coords = df[['x', 'y']].values
    original_labels = df['label'].values

    tree = cKDTree(original_coords)
    _, indices = tree.query(expanded_coords)
    expanded_df['label'] = original_labels[indices]

    # Create an image mask for expanded clusters
    expanded_mask = np.zeros(image_data.shape, dtype=bool)
    expanded_mask[expanded_df['y'], expanded_df['x']] = True

    return expanded_df, expanded_mask

def plot_clusters_on_fits(image_data, expanded_df, output_basename):
    """
    Plots clusters on a FITS image and saves the result as HTML and PNG files.

    Parameters:
    fits_file (str): Path to the FITS file.
    expanded_df (pd.DataFrame): DataFrame containing 'x', 'y', and 'label' columns.
    output_basename (str): Base name for the output files.
    """
    # Load the FITS data
    fits_data = image_data
    fits_data = np.log10(fits_data) 
    height, width = fits_data.shape

    # Filter the DataFrame to include only points within the image boundaries
    expanded_df = expanded_df[(expanded_df['x'] >= 0) & (expanded_df['x'] < width) & 
                              (expanded_df['y'] >= 0) & (expanded_df['y'] < height)]

    # Plot the FITS data using Plotly Express imshow
    fig = px.imshow(fits_data, origin='lower', color_continuous_scale='gray', template='plotly_white', zmin=np.log10(4000), zmax=np.log10(15000))

    # Update the layout to match the original Matplotlib plot style and match the image size
    fig.update_layout(
        xaxis_title='Pixel X',
        yaxis_title='Pixel Y',
        title='Clusters of Pixels Above Threshold',
        xaxis=dict(
            ticks="inside"
        ),
        yaxis=dict(
            ticks="inside"
        ),
        autosize=False,
        width=1200,  # Increase the width
        height=1000,  # Increase the height
        coloraxis_colorbar=dict(
            thicknessmode="pixels",
            thickness=15,
            lenmode="pixels",
            len=600,
            x=0.9, 
            xanchor='left'  # Match the height of the image
        )
    )
    
    # Create a scatter plot of the clusters
    scatter = go.Scatter(
        x=expanded_df['x'],
        y=expanded_df['y'],
        mode='markers',
        marker=dict(
            color=expanded_df['label'],
            colorscale='Viridis',
            size=3,
            colorbar=dict(
                title='Cluster Label',
                thickness=15,
                lenmode='pixels',
                len=600,  # Adjust the colorbar length
                xanchor='center',
                x=0.5,
                yanchor='top',
                y=-0.15,
                orientation='h'  # Horizontal colorbar
            )
        )
    )

    # Add the scatter plot to the figure
    fig.add_trace(scatter)

    # Update layout to match the original Matplotlib plot style
    fig.update_layout(
        dragmode='zoom',
        xaxis_title='<b>Pixel X</b>',
        yaxis_title='<b>Pixel Y</b>',
        xaxis=dict(ticks="inside"),
        yaxis=dict(ticks="inside"),
        autosize=False,
        width=800,
        height=1200,
        font_family='Times New Roman',
        font_size=20,
        font_color='black',
        coloraxis_colorbar=dict(
            title='Log10(Counts)',
            title_side='right',
            title_font=dict(family='Times New Roman', color='black', size=20),
            tickfont=dict(family='Times New Roman', color='black', size=20),
            thicknessmode="pixels",
            thickness=50,
            lenmode="fraction",
            len=1,
            x=0.86,
            y=0.5,
            xanchor='center'
        )
    )

    fig.update_xaxes(minallowed=0, title_font=dict(family='Times New Roman', size=20), tickfont=dict(family='Times New Roman', color='black', size=20))
    fig.update_xaxes(maxallowed=width)
    fig.update_yaxes(minallowed=0, title_font=dict(family='Times New Roman', size=20), tickfont=dict(family='Times New Roman', color='black', size=20))
    fig.update_yaxes(maxallowed=height)

    # Save the plot to an HTML file
    fig.write_html(f'{output_basename}_clusters.html')

    # Optionally, save the plot as a PNG file
    fig.write_image(f'{output_basename}_clusters.png', scale=2)

    # Show the plot
    fig.show()

@time_section
def find_trailing_pixels_kdtree(expanded_df, image_data, event_threshold, time_sections=False):
    # Build KD-tree for expanded pixels
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
    # Start timing the script
    print("Starting processing of", input_basename)
    start_time = time.time()
    
    # Handle NaN values
    dask_image_data = da.from_array(image_data, chunks=(1000, 1000))
    dask_image_data = handle_nan_values(dask_image_data, time_sections=TIME_SECTION)

    # Compute the histogram
    hist_computed, bin_edges_computed = compute_histogram(dask_image_data, time_sections=TIME_SECTION)

    # # Plot the histogram
    # plt.figure()
    # plt.bar(bin_edges_computed[:-1], hist_computed, width=np.diff(bin_edges_computed), edgecolor='black', align='edge')
    # plt.xlabel('Pixel value')
    # plt.ylabel('Frequency')
    # plt.yscale('log')
    # plt.title('Histogram of Pixel Values')
    # plt.xlim(HIST_PLOT_MIN,HIST_PLOT_MAX)
    # plt.savefig(os.path.join(output_dir, f'{input_basename}_histogram.png'))
    # plt.close()

    # Define the threshold using the Gaussian fit method
    gaussian_mean, gaussian_stddev = compute_gaussian_parameters(hist_computed, bin_edges_computed, time_sections=TIME_SECTION)
    # Calculate the threshold as mean + CR_THRESHOLD_SIGMA * stddev
    threshold = gaussian_mean + CR_THRESHOLD_SIGMA * gaussian_stddev
    print(f"Chosen threshold: {threshold}")

    # Calculate the event threshold
    event_threshold = gaussian_mean + EVENT_THRESHOLD_SIGMA * gaussian_stddev
    print(f"Event threshold: {event_threshold}")

    raw_emgain = estimate_em_gain(gaussian_mean, gaussian_stddev, hist_computed, bin_edges_computed, time_sections=TIME_SECTION)
    _, counts_above_threshold_raw = threshold_image_and_count_dask(dask_image_data, event_threshold)

    # Create a mask
    mask = create_mask(dask_image_data, threshold, time_sections=TIME_SECTION)

    # Compute the mask
    mask_computed = compute_mask(mask, time_sections=TIME_SECTION)
    y_coords, x_coords = np.nonzero(mask_computed)
    values_above_threshold = image_data[mask_computed]

    # Display the number of pixels above the threshold
    pixels_above_threshold = len(values_above_threshold)
    print(f"Number of pixels above threshold: {pixels_above_threshold}")

    # Prepare data for clustering
    coords = np.column_stack((x_coords, y_coords))

    # Apply DBSCAN
    labels = apply_dbscan(coords, time_sections=TIME_SECTION)

    # Number of clusters in labels, ignoring noise if present
    unique_clusters_initial = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Number of initial clusters: {unique_clusters_initial}")

    # Expand clusters to include neighboring pixels above the threshold using Dask and DBSCAN
    expanded_df, expanded_mask = expand_clusters_dbscan_dask(image_data, x_coords, y_coords, labels, event_threshold, eps=10, min_samples=3, time_sections=TIME_SECTION)
    pixels_after_expansion = len(expanded_df)
    print(f"Number of pixels after expansion: {pixels_after_expansion}")

    # Find trailing pixels
    trailing_pixels, trailing_mask = find_trailing_pixels_kdtree(expanded_df, image_data, event_threshold, time_sections=TIME_SECTION)
    pixels_after_trailing = len(trailing_pixels)
    print(f"Number of pixels after trailing: {pixels_after_trailing}")

    # Update expanded dataframe with trailing pixels
    trailing_df = pd.DataFrame(trailing_pixels, columns=['x', 'y', 'label'])
    expanded_df = pd.concat([expanded_df, trailing_df]).drop_duplicates().reset_index(drop=True)

    # Update mask with trailing pixels
    final_mask = np.logical_or(expanded_mask, trailing_mask)
    # Number of clusters after trailing
    unique_clusters_final = expanded_df['label'].nunique()
    print(f"Number of final clusters: {unique_clusters_final}")

    corrected_dask_image_data = replace_masked_pixels_with_noise(dask_image_data, final_mask, gaussian_mean, gaussian_stddev, time_sections=TIME_SECTION)

    # Generate a new histogram after replacing masked pixels
    hist_computed_corrected, bin_edges_computed_corrected = compute_histogram(corrected_dask_image_data, time_sections=TIME_SECTION)
    gaussian_mean_corrected, gaussian_stddev_corrected = compute_gaussian_parameters(hist_computed_corrected, bin_edges_computed_corrected, time_sections=TIME_SECTION)
    # Calculate the event threshold
    event_threshold_corrected = gaussian_mean_corrected + EVENT_THRESHOLD_SIGMA * gaussian_stddev_corrected
    corrected_emgain = estimate_em_gain(gaussian_mean_corrected, gaussian_stddev_corrected, hist_computed_corrected, bin_edges_computed_corrected, time_sections=TIME_SECTION)
    
    print(f"Event threshold: {event_threshold_corrected}")
    binary_image, count_above_threshold = threshold_image_and_count_dask(corrected_dask_image_data, event_threshold_corrected)

    # Optionally, save the coordinates, pixel values, and cluster labels to a file
    df_above_threshold = pd.DataFrame({'X': expanded_df['x'], 'Y': expanded_df['y'], 'Cluster': expanded_df['label']})
    csv_filename = os.path.join(output_dir, f'{input_basename}_pixels_above_threshold_with_clusters.csv')
    exptime = image_header['EXPTIME']
    with open(csv_filename, 'w') as f:
        f.write(f'# Input_file: {input_basename}\n')
        f.write(f'# Exptime: {exptime}\n')
        f.write(f'# Total_pixels: {np.size(image_data)}\n')
        f.write(f'# CR_threshold: {threshold}\n')
        f.write(f'# Event_threshold: {event_threshold}\n')
        f.write(f'# Gaussian_mean: {gaussian_mean}\n')
        f.write(f'# Gaussian_stddev: {gaussian_stddev}\n')
        f.write(f'# EMGain_Raw: {raw_emgain}\n')
        f.write(f'# Events_Raw: {counts_above_threshold_raw}\n')
        f.write(f'# Number_of_unique_clusters: {unique_clusters_final}\n')
        f.write(f'# Total_Pix_above_CR_Threshold: {pixels_above_threshold}\n')
        f.write(f'# Total_Pix_after_CR_tails: {pixels_after_expansion}\n')
        f.write(f'# Total_Pix_CR_Neighbours_tails: {pixels_after_trailing}\n')
        f.write(f'# Gaussian_mean_corrected: {gaussian_mean_corrected}\n')
        f.write(f'# Gaussian_stddev_corrected: {gaussian_stddev_corrected}\n')
        f.write(f'# Event_Threshold_Corrected: {event_threshold_corrected}\n')
        f.write(f'# Events_Corrected: {count_above_threshold}\n')
        f.write(f'# EMGain_Corrected: {corrected_emgain}\n')
        df_above_threshold.to_csv(f, index=False)

    # End timing the script
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")

@time_section
def process_all_images(input_folder, output_folder):
    # Get list of FITS files
    fits_files = sorted(glob.glob(os.path.join(input_folder, '*.fits')))

    # Initialize Dask client
    client = Client()
    print(f"Processing {len(fits_files)} FITS files")

    # Process each image file
    futures = []
    for fits_file in fits_files:
        input_basename = os.path.splitext(os.path.basename(fits_file))[0]
        output_dir = output_folder
        os.makedirs(output_dir, exist_ok=True)
        
        image_data = fits.getdata(fits_file)[:, 1024:2048]
        image_header = fits.getheader(fits_file)
        futures.append(client.submit(process_image, image_data, image_header, input_basename, output_dir))

    # Wait for all futures to complete
    results = client.gather(futures)
    client.close()
    return results
def main(input_file,output_dir):
    input_basename = os.path.splitext(os.path.basename(input_file))[0] # Get the base name of the input file
    
    os.makedirs(output_dir, exist_ok=True)
    image_data = fits.getdata(input_file)[:, 1024:2048]# region is manually defined currently. 
    image_header = fits.getheader(input_file)
    process_image(image_data, image_header, input_basename, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a FITS file.")
    parser.add_argument(
        '--input_file', 
        type=str, 
        required=True, 
        help='Path to the input FITS file.'
    )
    args = parser.parse_args()
    input_file = args.input_file
    if not os.path.isfile(input_file):
        raise ValueError(f"The provided path '{input_file}' is not a valid file.")
    else:
        print(f"Processing file: {input_file}")
        output_dir = os.path.join(os.path.dirname(input_file), 'output')
        print(f"Output directory: {output_dir}")
    
    main(input_file,output_dir)
