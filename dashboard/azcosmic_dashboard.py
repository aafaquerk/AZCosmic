import os
import numpy as np
import streamlit as st
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
from dask.distributed import Client
import plotly.express as px
import plotly.graph_objects as go

CONVERSION_GAIN=1.8 # e-/ADU
READNOISE=17 # ADU
global linear_fit_sigma_max, linear_fit_sigma_min
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
    def replace_block(block, block_mask, block_info=None):
        noise = np.random.normal(mean, stddev, size=block.shape)
        block[block_mask] = noise[block_mask]
        return block

    # Ensure mask has the same chunking as the image_data
    mask = mask.rechunk(image_data.chunks)
    replaced_data = da.map_blocks(replace_block, image_data, mask, dtype=image_data.dtype)
    return replaced_data

def plot_histogram(data, bin_edges, threshold,cr_threshold, mean, std, emgain,em_slope,em_intercept,linear_fit_sigma_max,linear_fit_sigma_min,hist_plot_max,hist_plot_min,output_dir,input_basename,time_sections=False):
    # Calculate the histogram
    # Plot the histogram
    linear_fit_range_max=mean + linear_fit_sigma_max * std
    linear_fit_range_min=mean + linear_fit_sigma_min * std
    fig,ax=plt.subplots()
    ax.plot(bin_edges[:-1], data, label='Histogram', color='darkturquoise',alpha=0.7)
    ax.axvline(threshold, color='g', linestyle='--', label='Event Threshold')
    ax.axvline(cr_threshold, color='r', linestyle='--', label='CR Threshold')
    xrange=np.linspace(mean,linear_fit_range_max,1000)
    ax.plot(xrange,10**(em_slope*xrange+em_intercept), label='EM_gain Fit', color='y')
    # gaussian_fit_range = np.where((bin_edges >= mean - 1.5 * std) & (bin_edges <= mean + 2* std))
    # ax.plot(gaussian_fit_range,gaussian(gaussian_fit_range, np.max(data), mean, std), label='Gaussian Fit', color='r')
    ax.axvline(linear_fit_range_max, color='g', linestyle='-', label='Linear Fit Range Max')
    ax.axvline(linear_fit_range_min, color='r', linestyle='-', label='Linear Fit Range Min') 
    # Add titles and labels
    plt.title(f'Histogram with Gaussian Fit and Threshold, \nEM Gain: {emgain:.2f}, Mean: {mean:.2f}, Stddev: {std:.2f}, Threshold: {threshold:.2f}, CR Threshold: {cr_threshold:.2f}, \nLinear Fit Range Max: {linear_fit_range_max:.2f}, Linear Fit Range Min: {linear_fit_range_min:.2f}')
    plt.xlabel('Counts (ADU)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.xlim(hist_plot_min, hist_plot_max)
    plt.yscale('log')

    # hist_filename = os.path.join(output_dir, f'{input_basename}_histogram.png')
    # Show plot
    # plt.savefig(hist_filename)
    return fig
    

@time_section
def estimate_em_gain(mean, stddev, hist, bin_centers,linear_fit_sigma_min, linear_fit_sigma_max,time_sections=False):
    try:
        start_fit = mean + linear_fit_sigma_min * stddev
        end_fit = mean + linear_fit_sigma_max * stddev
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
    window_size = 5
    start_index = max(peak_bin_index - window_size, 0)
    end_index = min(peak_bin_index + window_size + 1, len(bin_edges) - 1)
    fit_bins = bin_edges[start_index:end_index]
    fit_counts = hist[start_index:end_index - 1]

    popt, _ = curve_fit(gaussian, fit_bins[:-1], fit_counts, p0=[np.max(fit_counts), peak_bin_value, np.std(fit_bins)])
    amp, mean, stddev = popt
    return mean, stddev

@time_section
def handle_nan_values(image_data, time_sections=False):
    return da.where(da.isnan(image_data), 0, image_data)

@time_section
def compute_histogram(dask_image_data, hist_bins=2000, time_sections=False):
    data_min, data_max = da.min(dask_image_data).compute(), da.max(dask_image_data).compute()
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
    df_expanded['label'] = new_labels
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

    # Create an image mask for expanded clusters
    expanded_mask = np.zeros(image_data.shape, dtype=bool)
    expanded_mask[expanded_df['y'], expanded_df['x']] = True

    return expanded_df, expanded_mask

@time_section
def fits_header_to_dataframe(header):
    header_dict = {key: header[key] for key in header.keys()}
    df = pd.DataFrame(list(header_dict.items()), columns=['Key', 'Value'])
    return df

@time_section
def plot_fits(image_data, cbar_min=3000, cbar_max=40000):
    fits_data = image_data
    fits_data = np.log10(fits_data)
    height, width = fits_data.shape

    fig = px.imshow(fits_data, origin='lower', color_continuous_scale='gray', template='plotly_white', zmin=np.log10(cbar_min), zmax=np.log10(cbar_max))

    fig.update_layout(
        dragmode='zoom',
        xaxis_title='<b>Pixel X</b>',
        yaxis_title='<b>Pixel Y</b>',
        xaxis=dict(ticks="inside"),
        yaxis=dict(ticks="inside"),
        #autosize=True,
        width=1000,
        height=1200,
        font_family='Times New Roman',
        font_size=500,
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

    return fig

@time_section
def plot_clusters_on_fits(image_data, expanded_df, output_basename, cbar_min=3000, cbar_max=40000):
    fits_data = image_data
    fits_data = np.log10(fits_data)
    height, width = fits_data.shape

    expanded_df = expanded_df[(expanded_df['x'] >= 0) & (expanded_df['x'] < width) & 
                              (expanded_df['y'] >= 0) & (expanded_df['y'] < height)]

    fig = px.imshow(fits_data, origin='lower', color_continuous_scale='gray', template='plotly_white', zmin=np.log10(4000), zmax=np.log10(15000))

    scatter = go.Scatter(
        x=expanded_df['x'],
        y=expanded_df['y'],
        mode='markers',
        marker=dict(
            color=expanded_df['label'],
            colorscale=px.colors.qualitative.Pastel,
            size=3,
            colorbar=dict(
                title='Cluster Label',
                title_side='bottom',
                thickness=50,
                lenmode='fraction',
                len=0.5,
                xanchor='center',
                x=0.5,
                yanchor='top',
                y=-0.1,
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
        #autosize=True,
        width=1000,
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
    fig.write_image(f'{output_basename}_clusters.png')

    return fig

@time_section
def plot_mask(mask, output_basename):
    fits_data = mask
    height, width = fits_data.shape

    fig = px.imshow(fits_data, origin='lower', color_continuous_scale='gray', template='plotly_white')

    fig.update_layout(
        dragmode='zoom',
        xaxis_title='<b>Pixel X</b>',
        yaxis_title='<b>Pixel Y</b>',
        xaxis=dict(ticks="inside"),
        yaxis=dict(ticks="inside"),
        #autosize=True,
        width=1000,
        height=1200,
        font_family='Times New Roman',
        font_size=20,
        font_color='black',
        coloraxis_showscale=False

    )

    fig.update_xaxes(minallowed=0, title_font=dict(family='Times New Roman', size=20), tickfont=dict(family='Times New Roman', color='black', size=20))
    fig.update_xaxes(maxallowed=width)
    fig.update_yaxes(minallowed=0, title_font=dict(family='Times New Roman', size=20), tickfont=dict(family='Times New Roman', color='black', size=20))
    fig.update_yaxes(maxallowed=height)

    fig.write_html(f'{output_basename}_clusters.html')
    fig.write_image(f'{output_basename}_clusters.png')

    return fig

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
def process_image(image_data, image_header, input_basename, output_dir, cr_threshold_sigma=600, event_threshold_sigma=5.5, linear_fit_sigma_max=100, linear_fit_sigma_min=30, cluster_membership_radius=10, cluster_membership_threshold=3, hist_plot_max=40000, hist_plot_min=6000, hist_bins=2000, time_sections=False):
    start_time = time.time()
    dask_image_data = da.from_array(image_data, chunks=(1000, 1000))
    dask_image_data = handle_nan_values(dask_image_data, time_sections=True)

    hist_computed, bin_edges_computed = compute_histogram(dask_image_data, hist_bins, time_sections=True)

    gaussian_mean, gaussian_stddev = compute_gaussian_parameters(hist_computed, bin_edges_computed, time_sections=True)
    cr_threshold = gaussian_mean + cr_threshold_sigma * gaussian_stddev
    print(f"Chosen threshold: {cr_threshold}")

    event_threshold = gaussian_mean + event_threshold_sigma * gaussian_stddev
    print(f"Event threshold: {event_threshold}")

    raw_emgain,raw_slope,raw_intercept = estimate_em_gain(gaussian_mean, gaussian_stddev, hist_computed, bin_edges_computed, linear_fit_sigma_min=linear_fit_sigma_min, linear_fit_sigma_max=linear_fit_sigma_max, time_sections=True)
    hist_fig=plot_histogram(hist_computed, bin_edges_computed, event_threshold,cr_threshold, gaussian_mean, gaussian_stddev, raw_emgain,raw_slope,raw_intercept,linear_fit_sigma_max,linear_fit_sigma_min,hist_plot_max,hist_plot_min,output_dir,input_basename,time_sections=True)
    _, counts_above_threshold_raw = threshold_image_and_count_dask(dask_image_data, event_threshold)

    mask = create_mask(dask_image_data, cr_threshold, time_sections=True)
    mask_computed = compute_mask(mask, time_sections=True)
    y_coords, x_coords = np.nonzero(mask_computed)
    values_above_threshold = image_data[mask_computed]
    pixels_above_threshold = len(values_above_threshold)
    print(f"Number of pixels above threshold: {pixels_above_threshold}")

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
    print(f"Number of final clusters: {unique_clusters_final}")

    final_mask = np.logical_or(expanded_mask, trailing_mask)

    corrected_dask_image_data = replace_masked_pixels_with_noise(dask_image_data, da.from_array(final_mask, chunks=(1000, 1000)), gaussian_mean, gaussian_stddev, time_sections=True)
    hist_computed_corrected, bin_edges_computed_corrected = compute_histogram(corrected_dask_image_data, hist_bins, time_sections=True)
    gaussian_mean_corrected, gaussian_stddev_corrected = compute_gaussian_parameters(hist_computed_corrected, bin_edges_computed_corrected, time_sections=True)
    event_threshold_corrected = gaussian_mean_corrected + event_threshold_sigma * gaussian_stddev_corrected
    corrected_emgain,corrected_slope,corrected_intercept = estimate_em_gain(gaussian_mean_corrected, gaussian_stddev_corrected, hist_computed_corrected, bin_edges_computed_corrected, linear_fit_sigma_min=linear_fit_sigma_min, linear_fit_sigma_max=linear_fit_sigma_max, time_sections=True)
    print(f"Event threshold: {event_threshold}")
    hist_fig_corrected=plot_histogram(hist_computed_corrected, bin_edges_computed_corrected, event_threshold_corrected,cr_threshold, gaussian_mean_corrected, gaussian_stddev_corrected, corrected_emgain,corrected_slope,corrected_intercept,linear_fit_sigma_max,linear_fit_sigma_min,hist_plot_max,hist_plot_min,output_dir,input_basename,time_sections=True)
    binary_image, count_above_threshold = threshold_image_and_count_dask(corrected_dask_image_data, event_threshold_corrected)

    df_above_threshold = pd.DataFrame({'X': expanded_df['x'], 'Y': expanded_df['y'], 'Cluster': expanded_df['label']})
    csv_filename = os.path.join(output_dir, f'{input_basename}_pixels_above_threshold_with_clusters.csv')
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
        'Gaussian mean corrected': gaussian_mean_corrected,
        'Gaussian stddev corrected': gaussian_stddev_corrected,
        'Event Threshold Corrected': event_threshold_corrected,
        'Events Corrected': count_above_threshold,
        'EMGain Corrected': corrected_emgain
    }
    print(meta_data_dict)
    metadata_df = pd.DataFrame(meta_data_dict, index=[0])

    with open(csv_filename, 'w') as f:
        f.write(f'# Input file: {input_basename}\n')
        f.write(f'# Exptime: {exptime}\n')
        f.write(f'# Total pixels: {np.size(image_data)}\n')
        f.write(f'# CR threshold: {cr_threshold}\n')
        f.write(f'# Event threshold: {event_threshold}\n')
        f.write(f'# Gaussian mean: {gaussian_mean}\n')
        f.write(f'# Gaussian stddev: {gaussian_stddev}\n')
        f.write(f'# EMGain Raw: {raw_emgain}\n')
        f.write(f'# Events Raw: {counts_above_threshold_raw}\n')
        f.write(f'# Number of unique clusters: {unique_clusters_final}\n')
        f.write(f'# Total pixels above CR_Threshold: {pixels_above_threshold}\n')
        f.write(f'# Total pixels after CR_tails: {pixels_after_expansion}\n')
        f.write(f'# Total pixels after CR_CR_Neighbours_CR_tails: {pixels_after_trailing}\n')
        f.write(f'# Gaussian mean corrected: {gaussian_mean_corrected}\n')
        f.write(f'# Gaussian stddev corrected: {gaussian_stddev_corrected}\n')
        f.write(f'# Event Threshold Corrected: {event_threshold_corrected}\n')
        f.write(f'# Events Corrected: {count_above_threshold}\n')
        f.write(f'# EMGain Corrected: {corrected_emgain}\n')
        df_above_threshold.to_csv(f, index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")
    return expanded_df, metadata_df, corrected_dask_image_data, final_mask,hist_fig,hist_fig_corrected 

if __name__ == "__main__":
    st.set_page_config(
        page_title="AZCosmic Live: Cosmic ray detection app",
        page_icon=r"/Users/arkhan/Documents/Dark_Results/dark_SPIE_examples/images/logo/logo.png",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "# This is a header. This is an *extremely* cool app!"
        }
    )

    st.title("AZCosmic Live: A Cosmic Ray detection and removal application for EMCCDs")

    col1, col2, col3 = st.columns([0.5, 1, 0.5])

    with col1:
        uploaded_file = st.file_uploader("Select a FITS file to process", type="fits")
        output_dir = st.text_input("Output Directory", value="output")
    with col3:
        container_settings = st.container()
        with container_settings:
            expander_settings_CR = st.expander("CR detection Settings", expanded=True)
            with expander_settings_CR:
                event_threshold_sigma = st.slider("Event Threshold Sigma", min_value=1.0, max_value=10.0, value=5.5, step=0.1)
                cr_threshold_sigma = st.slider("CR Threshold Sigma", min_value=100, max_value=1000, value=600, step=50)
                linear_fit_sigma_max = st.slider("Linear Fit Sigma Max", min_value=50, max_value=600, value=100, step=10)
                linear_fit_sigma_min = st.slider("Linear Fit Sigma Min", min_value=10, max_value=50, value=30, step=5)
                cluster_membership_radius = st.slider("Cluster Membership", min_value=1, max_value=10, value=3, step=1)
                cluster_membership_threshold = st.slider("Cluster Membership Threshold", min_value=1, max_value=8, value=3, step=1)
                chunk_size = st.slider("Chunk Size", min_value=200, max_value=1000, value=500, step=100)
            expander_settings_hist = st.expander("Histogram Settings", expanded=True)
            with expander_settings_hist:
                hist_bins = st.slider("Histogram Bins", min_value=1000, max_value=5000, value=2000, step=100)
                hist_plot_max = st.slider("Histogram Plot Max", min_value=10000, max_value=50000, value=10000, step=1000)
                hist_plot_min = st.slider("Histogram Plot Min", min_value=0, max_value=10000, value=3500, step=500)

    with col2:
        placeholder = st.empty()
        if uploaded_file is not None:
            input_basename = os.path.splitext(os.path.basename(uploaded_file.name))[0]
            os.makedirs(output_dir, exist_ok=True)
            with st.status("Loading Image...", expanded=True) as status1:
                with st.spinner("FITS Imaging coming right up!"):
                    with fits.open(uploaded_file) as hdul:
                        image_data = hdul[0].data[:, 1024:2048]
                        over_scan_data = hdul[0].data[:, 0:1024]
                        image_header = hdul[0].header
                        tab_image,tab_overscan = st.tabs(["Image","Overscan"])
                        fig_fits = plot_fits(image_data, cbar_min=hist_plot_min, cbar_max=hist_plot_max)
                        fig_fits_overscan = plot_fits(over_scan_data, cbar_min=hist_plot_min, cbar_max=hist_plot_max)
                        df = fits_header_to_dataframe(image_header)
                        with tab_image:
                            st.plotly_chart(fig_fits, use_container_width=True)
                        with tab_overscan: 
                            st.plotly_chart(fig_fits, use_container_width=True)
                        status1.success("Image loaded successfully!")
                        status1.update(label=f"Image: {input_basename}", expanded=False)
                        with col1: 
                            with st.expander("FITS Header", expanded=True) as expander_results:
                                st.dataframe(df, use_container_width=True, hide_index=True)
        if col1.button("Start Processing", key="start_processing", help="Click to start processing the image", disabled=uploaded_file is None):
            status1.update(label="Original Image", expanded=False)
            try:
                with st.status("Searching for Cosmic rays.", expanded=True) as status2:
                    tab1, tab2 = st.tabs(["Image","Histogram"])
                    with st.spinner("Coming soon: Processed image and histogram...") as spinner:
                        expanded_df, metadata_df, corrected_image_data, final_mask,fig_hist,fig_hist_corrected = process_image(
                            image_data, image_header, input_basename, output_dir,
                            hist_plot_min=hist_plot_min, hist_plot_max=hist_plot_max,
                            linear_fit_sigma_max=linear_fit_sigma_max, linear_fit_sigma_min=linear_fit_sigma_min,
                            cluster_membership_radius=cluster_membership_radius, cluster_membership_threshold=cluster_membership_threshold,
                            event_threshold_sigma=event_threshold_sigma, cr_threshold_sigma=cr_threshold_sigma
                        )
                        status2.update(label="Image processing complete", expanded=True)
                        fig_output = plot_clusters_on_fits(image_data, expanded_df, input_basename, cbar_min=hist_plot_min, cbar_max=hist_plot_max)
                        with tab1:
                            st.plotly_chart(fig_output, use_container_width=True)
                        with tab2: 
                            st.pyplot(fig_hist)
                        status2.success("Processing complete!")
                        status2.update(label="Cosmic ray events", expanded=True)
                col1.success("Image processing complete!")
                col1.write(f"Processed image saved to: {output_dir}")
                expander_cleaned_image = st.expander("Cleaned Image", expanded=True)
                with expander_cleaned_image:
                    fig_output = plot_fits(corrected_image_data, cbar_min=hist_plot_min, cbar_max=hist_plot_max)
                    tab_cleaned_image,tab_hist = st.tabs(["Cleaned Image","Histogram"])
                    with tab_cleaned_image:
                        st.plotly_chart(fig_output, use_container_width=True)
                    with tab_hist:
                        st.pyplot(fig_hist_corrected)
                expander_data = st.expander("Results", expanded=True)
                placeholder2 = st.empty()
                with placeholder2:
                    expander_data = st.expander("Results", expanded=True)
                    with expander_data:
                        tab1, tab2 = st.tabs(["CR pixels", "Metadata"])
                        with tab1:
                            st.dataframe(expanded_df, use_container_width=True, hide_index=True)
                        with tab2:
                            st.dataframe(metadata_df, use_container_width=True, hide_index=True)
                expander_mask = st.expander("Final Mask", expanded=True)
                with expander_mask:
                    fig_mask = plot_mask(final_mask,input_basename)
                    st.plotly_chart(fig_mask, use_container_width=True)
            except Exception as e:
                col1.error(f"An error occurred while processing the image: {e}")
