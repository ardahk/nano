import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Set backend to 'Agg' (non-interactive)
import matplotlib.pyplot as plt
import time
import base64
import io
from io import BytesIO
from scipy.spatial import KDTree
import cv2
from PIL import Image

def create_base64_plot(plot_data):
    """Converts a plot to a base64 string"""
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.heatmap(plot_data, cmap='gray', cbar=False, square=True, linewidths=0.0, linecolor="black", ax=ax)
    ax.set_title(f'Plot Title')

    # Save plot to BytesIO buffer
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)

    # Encode image as base64
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close(fig)

    return img_base64

def encode_image_to_base64(fig):
    # Save the plot to a BytesIO buffer
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    # Encode the image to base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

def create_2d_gaussian(ksize, sigma):
    """
    Creates a 2D Gaussian kernel of given dimension and sigma.
    """
    kernel_1d = cv2.getGaussianKernel(ksize = ksize, sigma = sigma)
    kernel_2d = np.outer(kernel_1d, kernel_1d.T)
    return kernel_2d

def image_to_mask(image_file, flip=False, wafer_dim=(150,150)):
    """
    Takes an image filepath and returns a binary array as a mask scaled to the desired wafer dimensions (in nm). Provides an option to flip the mask.
    """

    # Loading image
    img = Image.open(image_file)
    img_resized = img.resize((wafer_dim[0], wafer_dim[1]), Image.LANCZOS)

    # Convert to grayscale array
    img_gray = img_resized.convert('L')  
    mask_array = np.array(img_gray)
    
    # Integer thresholding
    threshold = 128
    binary_mask = (mask_array > threshold).astype(int)
    
    # Option to flip mask
    if flip: binary_mask = np.logical_not(binary_mask)
    
    return binary_mask

class ElectronBeamLithographySimulator():
    """
    Class for electron beam lithography simulation.
    """

    def __init__(self, wafer_dim=(100,100), cnt_grid_shape=(5,4), cnt_unit_dim=(5,5), sg_ksize=5, sg_sigma=1, pixel_size_nm=5, wafer_depth=400, psf_kernel_size=5, sigma_forward=5, sigma_backward=10, weight_forward=0.9, weight_backward=0.1, exposure_thresh_low=0.2, exposure_thresh_high=1.2, exposure_target=1.0, sigmoid_steepness=10):
        """
        Initializes basic simulation information, masks, wafer, point spread function kernel, and data structures.
        """

        # CNT gun grid
        self.cnt_grid_shape = cnt_grid_shape
        self.cnt_unit_dim = cnt_unit_dim
        self.cnt_rows = self.cnt_grid_shape[0] * self.cnt_unit_dim[0]
        self.cnt_cols = self.cnt_grid_shape[1] * self.cnt_unit_dim[1]
        self.cnt_matrix = self.init_cnt_matrix()

        # Basic wafer information
        self.wafer_dim = wafer_dim
        self.wafer_depth = wafer_depth

        # Creating wafers and masks
        self.wafer = None
        self.pattern_mask = None
        self.dose_mask = None
        self.instantiate_substrate_and_masks()

        # CNT gun coordinates and empty mask target coordinates
        self.cnt_coords = self.get_cnt_coords()
        self.target_coords = None
        self.cnts_fired = np.zeros_like(self.wafer)

        # Get single gaussian electron Point Spread Function (PSF) approximiation
        self.sg_ksize = sg_ksize
        self.sg_sigma = sg_sigma
        self.sg_psf_approx = create_2d_gaussian(self.sg_ksize, self.sg_sigma)

        # Get double gaussian PSF
        # # CURRENTLY NOT IN USE!
        # self.pixel_size = pixel_size_nm
        # self.psf_kernel_size = psf_kernel_size
        # self.sigma_forward = sigma_forward
        # self.sigma_backward = sigma_backward
        # self.weight_forward = weight_forward
        # self.weight_backward = weight_backward
        # self.dg_psf_approx = self.get_point_spread_function_kernel()

        # Set parameters for nonlinearity in response
        self.sigmoid_steepness = sigmoid_steepness
        self.exposure_thresh_low = exposure_thresh_low
        self.exposure_target = exposure_target
        self.exposure_thresh_high = exposure_thresh_high

        # Create empty dataframe with columns
        self.data = pd.DataFrame(columns=["Shift", "Activated CNT Indices"])

    def update_parameters(self, wafer_dim=(150,150), cnt_grid_shape=(20,20),cnt_unit_dim=(5,5)):
        """
        Initializes basic simulation information, masks, wafer, point spread function kernel, and data structures.
        """
            
        # CNT gun grid
        self.cnt_grid_shape = cnt_grid_shape
        self.cnt_unit_dim = cnt_unit_dim
        self.cnt_rows = self.cnt_grid_shape[0] * self.cnt_unit_dim[0]
        self.cnt_cols = self.cnt_grid_shape[1] * self.cnt_unit_dim[1]
        self.cnt_matrix = self.init_cnt_matrix()

        # Basic wafer information
        self.wafer_dim = wafer_dim

        # Creating wafers and masks
        self.wafer = None
        self.pattern_mask = None
        self.dose_mask = None
        self.instantiate_substrate_and_masks()

        # reset
        self.data = pd.DataFrame(columns=["Shift", "Activated CNT Indices"])



    def init_cnt_matrix(self):
        """
        Generates a matrix of individually addressable CNT guns.
        """

        # Create integer grid with '1' in the middle to represent CNT
        cnt_grid = np.zeros(self.cnt_unit_dim, dtype=int)
        cnt_grid[2, 2] = 1

        # Repeats this pattern for desired number of CNT guns on motor stage
        cnt_matrix = np.tile(cnt_grid, (self.cnt_grid_shape[0], self.cnt_grid_shape[1]))

        return cnt_matrix
    
    def get_cnt_coords(self):
        """
        Return the starting position of the CNT guns with respect to the wafer. The CNT motor stage begins with its center aligned with the wafer center.
        """

        # Retrieve relative shift for CNT grid
        start_row = (self.wafer_dim[1] - self.cnt_rows) // 2
        start_col = (self.wafer_dim[0] - self.cnt_cols) // 2
        
        # Shift coordinates and store as list
        gun_positons = np.argwhere(self.cnt_matrix == 1)
        cnt_coords = gun_positons + np.array([start_row, start_col])

        return cnt_coords

    def sigmoid_dosage_response_correction(self, dose_psf_kernel):
        """
        Applies sigmoid correction to PSF kernels scaled by dosage.
        """

        # Flatten PSF kernel
        base_flat_response = dose_psf_kernel.flatten()

        # Apply sigmoid filtering for substrate response
        corrected_response = 1 / (1 + np.exp( -self.sigmoid_steepness * (base_flat_response - self.exposure_target)))

        # Reshape PSF Kernel
        corrected_kernel = corrected_response.reshape(dose_psf_kernel.shape)

        return corrected_kernel

    def get_point_spread_function_kernel(self):
        """
        Returns a kernel for simulating the scattering behavior of electrons.
        """

        # Scale sigmas to pixel size
        sigma_forward_scaled = self.sigma_forward / self.pixel_size
        sigma_backward_scaled = self.sigma_backward / self.pixel_size

        # Create grid based on desired kernel size
        ax = np.linspace(-(self.psf_kernel_size // 2), self.psf_kernel_size // 2, self.psf_kernel_size)
        xx, yy = np.meshgrid(ax, ax)
        
        # Gaussian for forward scattering
        gaussian_forward = np.exp(-(xx**2 + yy**2) / (2 * sigma_forward_scaled**2))
        
        # Gaussian for adjusted backward scattering
        wafer_depth_pixels = self.wafer_depth / self.pixel_size
        sigma_backward_adjusted = sigma_backward_scaled * np.sqrt(1 + self.wafer_depth / 100)  
        gaussian_backward = np.exp(-((xx)**2 + (yy + self.wafer_depth / self.pixel_size)**2) / (2 * sigma_backward_adjusted**2))

        # Combine for normalized PSF function
        psf = self.weight_forward * gaussian_forward + self.weight_backward * gaussian_backward
        psf_max = np.max(psf)
        psf = psf / psf_max 

        return psf
    


    def find_matches(self):
        """
        Finds matches between the current CNT coords and target coords.
        """

        # Convert to sets for lookup
        cnt_coords_set = set(map(tuple, self.cnt_coords))
        target_coords_set = set(map(tuple, self.target_coords))

        # Find matches through set intersection
        matches = np.array(list(cnt_coords_set.intersection(target_coords_set)))
        new_target_coords = np.array(list(target_coords_set - cnt_coords_set))

        # Get the indices of the matched CNT coordinates
        matched_indices = []
        for match in matches:
            # Find the index of the match in cnt_coords
            index = np.argwhere((self.cnt_coords == match).all(axis=1)).flatten()
            if index.size > 0:
                matched_indices.append(index[0])

        return matches, matched_indices, new_target_coords
    
    def closest_pair(self):
        """
        Finds the closest pair of CNT gun and target coordinates using a KDTree.
        """

        # Use KD Tree to query closest pairs
        tree = KDTree(np.array(self.cnt_coords))

        # query closest points
        distances, idx = tree.query(np.array(self.target_coords))

        # Get index of closest pair
        min_idx = np.argmin(distances)

        # Retrieve closest pair
        target_coord = self.target_coords[min_idx]
        cnt_coord = self.cnt_coords[idx[min_idx]]

        return cnt_coord, target_coord
    
    def determine_direction(self, cnt_coord, target_coord):
        """
        Determines directional shift necessary to move a CNT gun to a desired location.
        """

        # Extract directional coordinates
        target_y, target_x = target_coord
        cnt_y, cnt_x = cnt_coord

        # Conditional movement selection
        if target_x == cnt_x and target_y == cnt_y:
            return "STAY"
        else:
            if target_x < cnt_x:
                return "LEFT"
            elif target_x > cnt_x:
                return "RIGHT"
            elif target_y < cnt_y:
                return "UP"
            elif target_y > cnt_y:
                return "DOWN"
            
    def shift_cnt_coords(self, direction):
        """
        Shifts the motor stage by a single unit in the desired direction.
        """

        # Only continue if there is true movement
        if direction != "STAY":
            
            # Define unit based on direction
            delta = 1 if direction == "DOWN" or direction == "RIGHT" else -1

            # In-place coordinate modification
            if direction == "DOWN" or direction == "UP":
                self.cnt_coords[:, 0] += delta
            else:
                self.cnt_coords[:, 1] += delta


    # based on these matches, draw on the mask
    def draw_on_wafer(self, matches):
        """
        Perform the actual procedure of incrementally etching the wafer and dosage mask. The wafer represents a perfect etching, whereas the dosage mask represents an etching with simulated electron spread.
        """

        # Retrieve dose remaining as inverse from dose mask
        dosages = 1 - self.dose_mask[tuple(matches.T)]
        ret_dosages = []

        # Extract mask dimensions
        mh, mw = self.dose_mask.shape

        # Iterate through matches
        for i, (x, y) in enumerate(matches):

            # Retrieve dosage at position
            dosage = dosages[i]
                    
            # Proceed if dosage is above necessary threshold
            if dosage >= self.exposure_thresh_low:

                # Correct the dosage and add to list
                sigmoid_dosage = self.sigmoid_dosage_response_correction(dosage)
                ret_dosages.append(dosage)

                # Scale and correct kernel
                scaled_psf = self.sg_psf_approx * sigmoid_dosage
                corrected_psf = self.sigmoid_dosage_response_correction(scaled_psf)
        
                # Get kernel dimensions and relative bounds
                kh, kw = corrected_psf.shape

                if 0 <= x < mh and 0 <= y < mw:
                    # Look at all appropriate positions
                    for dx in range(kh):
                        for dy in range(kw):
                            if 0 <= x + dx < mh and 0 <= y + dy < mw:
                                # Add and clip kernel
                                self.dose_mask[x + dx, y + dy] += corrected_psf[dx, dy]
                                self.dose_mask[x + dx, y + dy] = np.clip(self.dose_mask[x + dx, y + dy], 0, self.exposure_thresh_high)
            else:
                # If no dosage, add 0 as placeholder
                ret_dosages.append(0)

            # Apply dosage with no scattering effect no matter what for now
            # This will be changed soon
            self.wafer[x, y] = 1
                
        return ret_dosages
    
    def instantiate_substrate_and_masks(self):
        """
        Populates masks and wafer attributes based on parameters. The pattern mask is set as an empty list to allow for passing in images to the class to serve as masks.
        """

        # Set wafer and dose mask as an empty array
        self.wafer = np.zeros((self.wafer_dim[0], self.wafer_dim[1]), dtype=np.float32)
        self.dose_mask = self.wafer.copy()

        # Set pattern mask as an empty list
        self.pattern_mask = []

    def etch(self, mask = [], display = True):
        """
        Performs etching loop by shifting and activating CNT gun stage. Returns the data necessary to replicate the algorithm on the real system.
        """

        # Create a random mask if it does not exist already
        if len(mask) == 0:
            mask = np.random.randint(2, size=(self.wafer_dim[0], self.wafer_dim[1]), dtype=np.float32)

        # Initialize target coordinates
        self.target_coords = np.argwhere(mask == 1)
        
        # Empty lists to store data from each iteration
        shifts = []
        match_idx = []
        dosages = []
        
        step = 0
        start = time.time()
        # Iterate until all target coords have met their dosage requirement
        while np.any(self.target_coords):
            step += 1

            # Find optimal movement and shift CNT motor stage
            cnt_coord, target_coord = self.closest_pair()
            direction = self.determine_direction(cnt_coord, target_coord)
            self.shift_cnt_coords(direction)
            shifts.append(direction)

            # Shift coordinates accordingly
            matches, matched_indices, self.target_coords = self.find_matches()
            match_idx.append(matched_indices)

            # Etch the wafers and retrieve dosages
            ret_dosages = self.draw_on_wafer(matches)
            dosages.append(ret_dosages)

            # Optional display of progress
            # if display:
            #     clear_output(wait=True)
            #     self.display_etch_progress(matched_indices, step)
            
        end = time.time()

        # Printing algorithm information
        print(f"Algorithm completed in {round(end - start, 2)} seconds.")
        print(f"Matches: {(np.sum(self.wafer == mask) / len(self.wafer.flatten())) * 100}%.")
        print(f"There {'WAS' if np.all(self.mask == self.wafer) else 'was NOT'} a perfect match.")

        return shifts, match_idx, dosages
    
    def display_etch_progress(self, match_idx, step_num):
        """
        Visualizes the progress of the wafer and dosage mask at each time step, alongside the activations of the CNT guns.
        """
        
        # Highlight the fired CNT guns
        if len(match_idx) > 0:
            self.cnts_fired[tuple(np.array(self.cnt_coords)[match_idx].T)] = 1

        plt.figure(figsize=(15, 5))

        # Disply CNTs fired
        plt.subplot(1, 3, 1)
        cmap = sns.color_palette(['black', 'red'], as_cmap=True)
        sns.heatmap(self.cnts_fired, cmap=cmap, cbar=False, square=True, linewidths=0.0, linecolor="black")
        plt.title(f'CNT Grid - Step {step_num}')
        plt.axis('off')

        # Display basic wafer with ONLY targets perfectly exposed
        plt.subplot(1, 3, 2)
        sns.heatmap(self.wafer, cmap='gray', cbar=False, square=True, linewidths=0.0, linecolor="black")
        plt.title(f'Etched Wafer (Targets Only) - Shift {step_num}')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        sns.heatmap(self.dose_mask, cbar=False, square=True, linewidths=0.0, linecolor="black")
        plt.title(f'Reverse Dosage Mask - Shift {step_num}')
        plt.axis('off')

        plt.tight_layout()
        plt.ioff()
        plt.show()
        
        # Reset CNT firings (this removes red from the grid after display)
        self.cnts_fired[tuple(np.array(self.cnt_coords)[match_idx].T)] = 0
    

    def run_lithography(self, mask=None, display=False, mask_flip=False):
        # If a mask is provided directly, use it
        if mask is not None:
            self.mask = np.array(mask)  # Convert the list to a numpy array

        # Run etching process
        shifts, match_idx, dosages = self.etch(self.mask, display)

        # Generate and encode plots
        plots = []
        fig1, ax1 = plt.subplots(figsize=(15, 5))
        sns.heatmap(self.wafer, cmap='gray', cbar=False, square=True, linewidths=0.0, linecolor="black", ax=ax1)
        ax1.set_title(f'Etched Wafer (Targets Only)')
        ax1.axis('off')
        plots.append(encode_image_to_base64(fig1))
        
        fig2, ax2 = plt.subplots(figsize=(15, 5))
        sns.heatmap(self.dose_mask, cbar=False, square=True, linewidths=0.0, linecolor="black", ax=ax2)
        ax2.set_title(f'Etched Wafer (w/ Scattering)')
        ax2.axis('off')
        plots.append(encode_image_to_base64(fig2))
        
        fig3, ax3 = plt.subplots(figsize=(15, 5))
        sns.heatmap(self.mask.astype(np.uint8), cmap='gray', cbar=False, square=True, linewidths=0.0, linecolor="black", ax=ax3)
        ax3.set_title('Target Mask')
        ax3.axis('off')
        plots.append(encode_image_to_base64(fig3))

        # Convert Pandas DataFrame to JSON
        self.data["Shift"] = shifts
        self.data["Activated CNT Indices"] = match_idx
        # self.data["Relative Dosages"] = dosages # no dosages now
        df_json = self.data.to_json(orient='split')

        # Reset all
        self.instantiate_substrate_and_masks()

        # reset self.data
        self.data = pd.DataFrame(columns=["Shift", "Activated CNT Indices"])

        return {
            "plots": plots,  # Return the base64-encoded images
            "df_json": df_json  # Return the DataFrame as JSON
        }