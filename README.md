
# AZCosmic: A Cosmic Ray Detection and Removal Application for EMCCDs

AZCosmic is a code designed for detecting and removing cosmic rays from EMCCD images. The code leverages Dask for parallel processing and various scientific libraries for image analysis.

AZCosmic Live is a streamlit based dashboard that is used to test and demo the capabilities fo the AZCosmic code.

More details of the implementation coming soon! 

## Features

- Load and display FITS images
- Compute histograms and Gaussian parameters
- Detect cosmic rays using thresholds and DBSCAN clustering
- Detect if any neighbouring pixels to cosmic ray events are affected
- Detect and clean trailing pixels in comsic ray affected events
- Replace masked pixels with noise derived from the readnoise charachterisitcs of the image
- Threhold image for photon couting
- Generate plots: Image plots, histograms, map of affected pixels etc. 
- Updated feature list coming soon

## Installation

To use this application, you need to have Python installed along with the required libraries. The recommened  environment manage for AZCosmic is conda.  You can create a conda environment using a `requirements.yml` file.

1. **Create the conda environment:**

    ```bash
    conda env create -f requirements.yml
    ```

2. **Activate the conda environment:**

    ```bash
    conda activate AZCosmic
    ```

## Usage

### Dashboard

1. **Run the Streamlit app:**

    ```bash
    streamlit run dashboard/app.py
    ```

2. **Upload a FITS file:**

    Use the file uploader in the app to select a FITS file for processing.

3. **Adjust Settings:**

    Configure the CR detection settings, histogram settings, and other parameters as needed.

4. **Start Processing:**

    Click the "Start Processing" button to begin the cosmic ray detection and removal process.

### Scripts

You can also use standalone scripts to process one or multiple files without using the Streamlit dashboard.

1. **Single File Processing:**

    ```bash
    python scripts/process_single_file.py --input_file path/to/input.fits
    ```

2. **Batch Processing:**

    ```bash
    python scripts/process_multiple_files.py --input_dir path/to/input/dir
    ```

Coming soon: Conifguration files to store and read script parameters. 

## Example Code

Here's a brief overview of the main functions used in the application:

Combing soon! 

### `replace_masked_pixels_with_noise`

Replaces masked pixels in the image with noise.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License

All Rights Reserved

Copyright (c) ${AZCosmic.2024} ${Aafaque R Khan}

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.