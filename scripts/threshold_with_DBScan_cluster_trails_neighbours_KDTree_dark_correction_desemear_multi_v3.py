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
from scipy.spatial import cKDTree

cmap = plt.cm.rainbow

# Defined parameters
EVENT_THRESHOLD_SIGMA = 5.5
CR_THRESHOLD_SIGMA = 500
IMAGE_SIZE = 2048 * 1048
HIST_BINS = 5000
HIST_RANGE_MIN = 0
HIST_RANGE_MAX = 65000
LINEAR_FIT_SIGMA_MAX = 100
LINEAR_FIT_SIGMA_MIN = 50
HIST_PLOT_MAX = 10000
HIST_PLOT_MIN = 3000
SMEAR_THRESHOLD = 5.0
TIME_SECTION = True
CONVERSION_GAIN = 1.9
READ_NOISE = 12.0

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

def log_linear_model_emgain(x, slope, intercept):
    return slope * x + intercept

def plot_histogram(data, bin_edges, threshold, cr_threshold, mean, std, emgain, em_slope, em_intercept, linear_fit_sigma_max, linear_fit_sigma_min, hist_plot_max, hist_plot_min, output_dir, input_basename, time_sections=False):
    linear_fit_range_max = mean + linear_fit_sigma_max * std
    linear_fit_range_min = mean + linear_fit_sigma_min * std
    fig, ax = plt.subplots()
    ax.plot(bin_edges[:-1], data, label='Histogram', color='darkturquoise', alpha=0.7)
    ax.axvline(threshold, color='g', linestyle='--', label='Event Threshold')
    ax.axvline(cr_threshold, color='r', linestyle='--', label='CR Threshold')
    xrange = np.linspace(mean, linear_fit_range_max, 1000)
    ax.plot(xrange, np.e ** (em_slope * xrange + em_intercept), label='EM_gain Fit', color='y')
    ax.axvline(linear_fit_range_max, color='g', linestyle='-', label='Linear Fit Range Max')
    ax.axvline(linear_fit_range_min, color='r', linestyle='-', label='Linear Fit Range Min')
    plt.title(f'Histogram with Gaussian Fit and Threshold, \nEM Gain: {emgain:.2f}, Mean: {mean:.2f}, Stddev: {std:.2f}, Threshold: {threshold:.2f}, CR Threshold: {cr_threshold:.2f}, \nLinear Fit Range Max: {linear_fit_range_max:.2f}, Linear Fit Range Min: {linear_fit_range_min:.2f}')
    plt.xlabel('Counts (ADU)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.xlim(hist_plot_min, hist_plot_max)
    plt.yscale('log')
    return fig

def gaussian(x, amp, mean, stddev):
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

@time_section
def replace_masked_pixels_with_noise(image_data, mask, mean, stddev, time_sections=False):
    def replace_block(block, block_mask, block_info=None):
        noise = np.random.normal(mean, stddev, size=block.shape)
        block[block_mask] = noise[block_mask]
        return block

    mask = mask.rechunk(image_data.chunks)
    replaced_data = da.map_blocks(replace_block, image_data, mask, dtype=image_data.dtype)
    return replaced_data

@time_section
def estimate_em_gain(mean, stddev, hist, bin_centers, linear_fit_sigma_min, linear_fit_sigma_max, time_sections=False):
    try:
        start_fit = mean + linear_fit_sigma_min * stddev
        end_fit = mean + linear_fit_sigma_max * stddev
        fit_indices = np.where((bin_centers >= start_fit) & (bin_centers <= end_fit))[0]
        fit_x = bin_centers[fit_indices]
        fit_y = hist[fit_indices]
        log_fit_y = np.log(fit_y + 1e-10)
        p0 = [0.001, 1000]
        popt, pcov = curve_fit(log_linear_model_emgain, fit_x, log_fit_y, maxfev=10000, p0=p0)
        slope, intercept = popt
        emgain = (-1 / slope) * CONVERSION_GAIN
        return emgain, slope, intercept
    except IndexError as e:
        print(f"Error in estimate_em_gain: {e}")
        return None

@time_section
def compute_gaussian_parameters(hist, bin_edges, time_sections=False):
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    peak_bin_index = np.argmax(hist)
    peak_bin_value = bin_centers[peak_bin_index]
    window_size = 7
    start_index = max(peak_bin_index - window_size, 0)
    end_index = min(peak_bin_index + window_size + 1, len(bin_edges) - 1)
    gaussian_fit_bins = bin_edges[start_index:end_index]
    gaussian_fit_counts = hist[start_index:end_index - 1]

    popt, _ = curve_fit(gaussian, gaussian_fit_bins[:-1], gaussian_fit_counts, p0=[np.max(gaussian_fit_counts), peak_bin_value, np.std(gaussian_fit_bins)])
    amp, mean, stddev = popt
    print(f"Fitted Gaussian parameters: amp={amp}, mean={mean}, stddev={stddev}")
    return mean, stddev

@time_section
def handle_nan_values(image_data, time_sections=False):
    return da.where(da.isnan(image_data), 0, image_data)

@time_section
def compute_histogram(dask_image_data, hist_bins):
    data_min, data_max = da.min(dask_image_data).compute(), da.max(dask_image_data).compute()
    print(f"Data range: {data_min} to {data_max}")
    hist, bin_edges = da.histogram(dask_image_data, bins=hist_bins, range=(data_min, data_max))
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

def plot_clusters_on_fits(image_data, expanded_df, output_basename):
    fits_data = image_data
    fits_data = np.log10(fits_data) 
    height, width = fits_data.shape

    expanded_df = expanded_df[(expanded_df['x'] >= 0) & (expanded_df['x'] < width) & 
                              (expanded_df['y'] >= 0) & (expanded_df['y'] < height)]

    fig = px.imshow(fits_data, origin='lower', color_continuous_scale='gray', template='plotly_white', zmin=np.log10(4000), zmax=np.log10(15000))

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
        width=1200,
        height=1000,
        coloraxis_colorbar=dict(
            thicknessmode="pixels",
            thickness=15,
            lenmode="pixels",
            len=600,
            x=0.9, 
            xanchor='left'
        )
    )
    
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
                len=600,
                xanchor='center',
                x=0.5,
                yanchor='top',
                y=-0.15,
                orientation='h'
            )
        )
    )

    fig.add_trace(scatter)

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

    fig.write_html(f'{output_basename}_clusters.html')

    fig.write_image(f'{output_basename}_clusters.png', scale=2)

    fig.show()

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

def row_based_metric(p1, p2):
    if p1[0] == p2[0]:
        return np.abs(p1[1] - p2[1])
    return np.inf

@time_section
def desmear_original(image_data, desmear_threshold_multiplier, gaussian_sigma, gaussian_mean, method="mad", time_sections=False):
    image_data = image_data.compute()
    if method == "mad":
        medianval = np.median(image_data)
        madval = np.median(np.abs(image_data - medianval))
    else:
        ysize = 20
        xsize = 20
        medianval = np.zeros([ysize, xsize], dtype=np.float32)
        padded_img_data = np.pad(image_data, ((1, 1), (1, 1)), mode='constant', constant_values=np.nan)
        for j in range(1, ysize + 1):
            for i in range(1, xsize + 1):
                medianval[j - 1, i - 1] = np.nanmedian(padded_img_data[j - 1:j + 2, i - 1:i + 2])
        madval = np.zeros([ysize, xsize], dtype=np.float32)
        for j in range(1, ysize + 1):
            for i in range(1, xsize + 1):
                madval[j - 1, i - 1] = np.nanmedian(np.abs(padded_img_data[j - 1:j + 2, i - 1:i + 2] - medianval[j - 1, i - 1]))

    prandom_array = np.random.normal(int(gaussian_mean), int(gaussian_sigma), size=image_data.shape)

    pixel_length = []
    print(f"Desmear process")

    desmear_threshold = medianval + desmear_threshold_multiplier * madval
    idx = image_data > desmear_threshold

    percent_flag_events = (np.count_nonzero(idx) / (float(image_data.shape[0]) * image_data.shape[1])) * 100
    print(f"\t  percent pixels flagged in region: {percent_flag_events:6.2f}")

    mask_data = np.zeros(np.shape(image_data), dtype=np.int32)
    mask_data[idx] = 1
    new_img_data = np.copy(image_data).astype(np.int32)
    corr_mask_data = np.copy(mask_data)
    mask_data_ind = np.array(np.where(mask_data == 1))

    xind = mask_data_ind[1]
    yind = mask_data_ind[0]

    right_pixels = np.zeros_like(mask_data, dtype=int)
    for xi, yi in zip(xind, yind):
        if 1 < xi < mask_data.shape[1] - 1:
            if mask_data[yi, xi] == 1 and mask_data[yi, xi - 1] == 0:
                xval = xi
                counter = 0
                while xval < mask_data.shape[1] and mask_data[yi, xval] == 1:
                    xval += 1
                    counter += 1
                rpix = counter
                corr_mask_data[yi, xi + 1:xi + rpix + 1] = -1
                fixed_smear = new_img_data[yi, xi:xi + rpix + 1]
                fixed_smear_sum = np.sum(fixed_smear) - rpix * int(gaussian_mean)
                new_img_data[yi, xi] = fixed_smear_sum
                pixel_length.append(rpix)

    newmask_idx = corr_mask_data < 0
    new_img_data[newmask_idx] = prandom_array[newmask_idx]
    return {
        'original_image': image_data,
        'flagged_pixels': idx,
        'corrected_image': new_img_data,
        'desmear_threshold': desmear_threshold,
        'median': medianval,
        'madval': madval,
        'pixel_length': pixel_length,
        'percent_flag_events': percent_flag_events
    }

@time_section
def desmear_image(img_data_dask, desmear_threshold_multiplier, read_noise_std, bias_level, time_sections=False):
    ysize, xsize = img_data_dask.shape
    img_data_dask = img_data_dask.compute()
    medianval = np.median(img_data_dask)
    madval = np.median(np.abs(img_data_dask - medianval))

    print(f"Median: {medianval}, MAD: {madval}")
    desmear_threshold = medianval + desmear_threshold_multiplier * madval
    idx = img_data_dask > desmear_threshold
    mask_data_ind = np.array(np.where(idx == 1)).T
    clustering = DBSCAN(eps=1, min_samples=1, metric=row_based_metric).fit(mask_data_ind)
    labels = clustering.labels_
    new_img_data = img_data_dask
    corr_mask_data = idx
    events_above_demear_threshold = np.unique(labels)
    for label in np.unique(labels):
        cluster_indices = mask_data_ind[labels == label]
        if len(cluster_indices) > 0:
            yi, xi = cluster_indices[0]
            right_indices = cluster_indices[1:]
            rpix = len(right_indices)
            trailing_contiguous_pixel_counts = new_img_data[yi, xi + 1:xi + rpix + 1] - bias_level
            total_charge = np.sum(trailing_contiguous_pixel_counts) + new_img_data[yi, xi]
            if total_charge < CR_THRESHOLD_SIGMA * read_noise_std:
                new_img_data[yi, xi:xi + rpix + 1] = np.random.normal(loc=bias_level, scale=read_noise_std, size=rpix + 1)
                new_img_data[yi, xi] = total_charge
            else:
                new_img_data[yi, xi:xi + rpix + 1] = np.random.normal(loc=bias_level, scale=read_noise_std, size=rpix + 1)
    return {
        'original_image': img_data_dask,
        'flagged_pixels': idx,
        'correction_mask': corr_mask_data,
        'corrected_image': new_img_data,
        'labels': labels,
        'mask_data_ind': mask_data_ind,
        'desmear_threshold': desmear_threshold,
        'median': medianval,
        'madval': madval
    }

@time_section
def visualize_results(results, read_noise_std, bias_level, time_sections=False):
    img_data = results['original_image']
    idx = results['flagged_pixels']
    corr_mask_data = results['correction_mask']
    new_img_data = results['corrected_image']
    labels = results['labels']
    mask_data_ind = results['mask_data_ind']

    fig, axes = plt.subplots(3, figsize=(20, 10))
    axes[0].imshow(img_data, cmap='gray', vmin=bias_level - 50, vmax=bias_level + 50)
    axes[0].set_title('Original Image')

    axes[1].imshow(idx, cmap='gray')
    axes[1].set_title('Flagged Pixels')

    axes[2].imshow(new_img_data, cmap='gray', vmin=bias_level - 50, vmax=bias_level + 50)
    axes[2].set_title('Corrected Image')

    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_mask = (labels == label)
        coordinates = mask_data_ind[label_mask]
        axes[0].scatter(coordinates[:, 1], coordinates[:, 0], s=1, label=f'Cluster {label}')
    plt.savefig('desmear_results.png')

@time_section
def process_image(image_data, image_header, input_basename, output_dir, cr_threshold_sigma=CR_THRESHOLD_SIGMA, event_threshold_sigma=5.5, linear_fit_sigma_max=100, linear_fit_sigma_min=30, cluster_membership_radius=10, cluster_membership_threshold=3, hist_plot_max=40000, hist_plot_min=6000, hist_bins=2000, chunk_size=500, time_sections=False):
    start_time = time.time()
    dask_image_data = da.from_array(image_data, chunks=(chunk_size, chunk_size))
    dask_image_data = handle_nan_values(dask_image_data, time_sections=True)

    hist_computed, bin_edges_computed = compute_histogram(dask_image_data, HIST_BINS)
    print(f"Processing image: {input_basename}")
    gaussian_mean, gaussian_stddev = compute_gaussian_parameters(hist_computed, bin_edges_computed, time_sections=True)
    cr_threshold = gaussian_mean + cr_threshold_sigma * gaussian_stddev
    print(f"Chosen threshold: {cr_threshold}")

    event_threshold = gaussian_mean + event_threshold_sigma * gaussian_stddev
    print(f"Event threshold: {event_threshold}")

    raw_emgain, raw_slope, raw_intercept = estimate_em_gain(gaussian_mean, gaussian_stddev, hist_computed, bin_edges_computed, linear_fit_sigma_min=LINEAR_FIT_SIGMA_MIN, linear_fit_sigma_max=LINEAR_FIT_SIGMA_MAX, time_sections=True)
    hist_fig = plot_histogram(hist_computed, bin_edges_computed, event_threshold, cr_threshold, gaussian_mean, gaussian_stddev, raw_emgain, raw_slope, raw_intercept, LINEAR_FIT_SIGMA_MAX, LINEAR_FIT_SIGMA_MIN, HIST_PLOT_MAX, HIST_PLOT_MIN, output_dir, input_basename, time_sections=True)
    hist_fig.savefig(f'{output_dir}/{input_basename}_histogram_raw_hist.png')
    _, counts_above_threshold_raw = threshold_image_and_count_dask(dask_image_data, event_threshold)
    print(f"Number of pixels above threshold: {counts_above_threshold_raw}")

    mask = create_mask(dask_image_data, cr_threshold, time_sections=True)
    mask_computed = compute_mask(mask, time_sections=True)
    y_coords, x_coords = np.nonzero(mask_computed)
    values_above_threshold = image_data[mask_computed]
    pixels_above_threshold = len(values_above_threshold)

    coords = np.column_stack((x_coords, y_coords))
    labels = apply_dbscan(coords, time_sections=True)
    unique_clusters_initial = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Number of initial clusters: {unique_clusters_initial}")

    expanded_df, expanded_mask = expand_clusters_dbscan_dask(image_data, x_coords, y_coords, labels, event_threshold, eps=cluster_membership_radius, min_samples=cluster_membership_threshold, time_sections=True)
    pixels_after_expansion = len(expanded_df)
    print(f"Number of pixels after expansion: {pixels_after_expansion}")

    trailing_pixels, trailing_mask = find_trailing_pixels_kdtree(expanded_df, image_data, event_threshold, time_sections=True)
    pixels_after_trailing = len(trailing_pixels)
    print(f"Number of pixels after trailing: {pixels_after_trailing}")

    trailing_df = pd.DataFrame(trailing_pixels, columns=['x', 'y', 'label'])
    expanded_df = pd.concat([expanded_df, trailing_df]).drop_duplicates().reset_index(drop=True)
    unique_clusters_final = expanded_df['label'].nunique()
    hot_pixels_final = len(expanded_df[expanded_df['label'] == -1])
    print(f"Number of final clusters: {unique_clusters_final}")
    print(f"Number of hot pixels: {hot_pixels_final}")
    final_mask = np.logical_or(expanded_mask, trailing_mask)
    corrected_dask_image_data = replace_masked_pixels_with_noise(dask_image_data, da.from_array(final_mask, chunks=(1000, 1000)), gaussian_mean, gaussian_stddev, time_sections=True)
    hist_computed_corrected, bin_edges_computed_corrected = compute_histogram(corrected_dask_image_data, hist_bins)
    gaussian_mean_corrected, gaussian_stddev_corrected = compute_gaussian_parameters(hist_computed_corrected, bin_edges_computed_corrected)
    event_threshold_corrected = gaussian_mean_corrected + event_threshold_sigma * gaussian_stddev_corrected
    corrected_emgain, corrected_slope, corrected_intercept = estimate_em_gain(gaussian_mean_corrected, gaussian_stddev_corrected, hist_computed_corrected, bin_edges_computed_corrected, LINEAR_FIT_SIGMA_MIN, LINEAR_FIT_SIGMA_MAX, time_sections=True)
    print(f"Event threshold: {event_threshold}")
    hist_fig_corrected = plot_histogram(hist_computed_corrected, bin_edges_computed_corrected, event_threshold_corrected, cr_threshold, gaussian_mean_corrected, gaussian_stddev_corrected, corrected_emgain, corrected_slope, corrected_intercept, LINEAR_FIT_SIGMA_MAX, LINEAR_FIT_SIGMA_MIN, HIST_PLOT_MAX, HIST_PLOT_MIN, output_dir, input_basename, time_sections=True)
    hist_fig_corrected.savefig(f'{output_dir}/{input_basename}_histogram_corrected_hist.png')
    desmear_results = desmear_original(corrected_dask_image_data, SMEAR_THRESHOLD, gaussian_stddev_corrected, gaussian_mean_corrected, method='mad', time_sections=TIME_SECTION)
    desmear_corrected_image = da.from_array(desmear_results['corrected_image'], chunks=(1000, 1000))
    desmear_threshold = desmear_results['desmear_threshold']

    hist_computed_desmear, bin_edges_computed_desmeared = compute_histogram(desmear_corrected_image, hist_bins)
    gaussian_mean_desmear, gaussian_stddev_desmear = compute_gaussian_parameters(hist_computed_desmear, bin_edges_computed_desmeared)
    event_threshold_desmeared = gaussian_mean_desmear + event_threshold_sigma * gaussian_stddev_desmear
    desmear_emgain, desmeaer_slope, desmear_intercept = estimate_em_gain(gaussian_mean_desmear, gaussian_stddev_desmear, hist_computed_desmear, bin_edges_computed_desmeared, LINEAR_FIT_SIGMA_MIN, LINEAR_FIT_SIGMA_MAX, time_sections=True)
    print(f"Event threshold: {event_threshold}")
    hist_fig_desmear = plot_histogram(hist_computed_desmear, bin_edges_computed_desmeared, event_threshold_desmeared, cr_threshold, gaussian_mean_desmear, gaussian_stddev_desmear, desmear_emgain, desmeaer_slope, desmear_intercept, LINEAR_FIT_SIGMA_MAX, LINEAR_FIT_SIGMA_MIN, HIST_PLOT_MAX, HIST_PLOT_MIN, output_dir, input_basename, time_sections=True)
    hist_fig_desmear.savefig(f'{output_dir}/{input_basename}_histogram_desmear_hist.png')
    binary_image, count_above_threshold = threshold_image_and_count_dask(desmear_corrected_image, event_threshold)
    print(f"Number of pixels above threshold: {count_above_threshold}")

    df_above_threshold = pd.DataFrame({'X': expanded_df['x'], 'Y': expanded_df['y'], 'Cluster': expanded_df['label']})
    csv_filename = os.path.join(output_dir, f'{input_basename}_pixels_above_threshold_with_clusters.xlsx')
    exptime = image_header['EXPTIME']
    meta_data_dict = {
        'Input file': input_basename,
        'Exptime': exptime,
        'Total pixels': np.size(image_data),
        'CR threshold': cr_threshold,
        'Event threshold': event_threshold,
        'Gaussian mean': gaussian_mean,
        'Gaussian stddev': gaussian_stddev,
        'EMGain Raw': raw_emgain,
        'Events Raw': counts_above_threshold_raw,
        'Number of unique clusters': unique_clusters_final,
        'Total pixels above CR_Threshold': pixels_above_threshold,
        'Total pixels after CR_tails': pixels_after_expansion,
        'Total pixels after CR_CR_Neighbours_CR_tails': pixels_after_trailing,
        'Number of Hot pixels': hot_pixels_final,
        'Gaussian mean corrected': gaussian_mean_corrected,
        'Gaussian stddev corrected': gaussian_stddev_corrected,
        'Event Threshold Corrected': event_threshold_corrected,
        'Events Corrected': count_above_threshold,
        'EMGain Corrected': corrected_emgain,
        'Desmear Threshold': desmear_threshold,
        'Desmear Mad': desmear_results['madval'],
        'Desmear Median': desmear_results['median'],
        'Desmear Pixel Length': np.mean(desmear_results['pixel_length']),
        'Percent Flagged Events': desmear_results['percent_flag_events'],
        'Event Threshold Desmeared': event_threshold_desmeared,
        'EMGain Desmeared': desmear_emgain
    }

    metadata_df = pd.DataFrame(meta_data_dict, index=[0])
    with pd.ExcelWriter(csv_filename) as writer:
        metadata_df.T.to_excel(writer, sheet_name='Meta_data')
        df_above_threshold.to_excel(writer, sheet_name='CR_pixels')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")
    return expanded_df, metadata_df, corrected_dask_image_data, final_mask, hist_fig, hist_fig_corrected

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

        image_data = fits.getdata(fits_file)[500:1500, 1250:1750]
        image_header = fits.getheader(fits_file)

        # Scatter the data to the workers
        scattered_image_data = client.scatter(image_data)
        scattered_image_header = client.scatter(image_header)

        futures.append(client.submit(process_image, scattered_image_data, scattered_image_header, input_basename, output_dir))

    progress(futures)
    results = client.gather(futures)
    client.close()
    return results

def main():
    input_folder = r'/Users/arkhan/Documents/Dark_Results/dark_SPIE_examples/test_data_warm/dark_07152024_g7850_minus120_vss0'
    output_folder = os.path.join(input_folder, 'output')
    os.makedirs(output_folder, exist_ok=True)
    results = process_all_images(input_folder, output_folder)

if __name__ == "__main__":
    main()
