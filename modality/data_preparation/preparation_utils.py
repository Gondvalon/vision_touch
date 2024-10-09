import copy
from operator import concat

import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from matplotlib.lines import lineStyles
from scipy.ndimage import gaussian_filter, median_filter
from tqdm import tqdm

# can plot values as tau, q, dq
def plot_joint_values(file, value, struc_keys):
    with h5py.File(file, "r") as f:
        key = value
        tau = f[key][:]

        # Create a new figure
        plt.figure(figsize=(12, 8))

        for i, data_list in enumerate(struc_keys):
            plt.plot(tau[data_list], label=struc_keys[i])

        # Add title and labels
        plt.title('Plot key values along number of recordings')
        plt.xlabel('Recordings')
        plt.ylabel('Value')

        # Add legend
        plt.legend()

        # Show plot with grid
        plt.grid(True)
        plt.show()

def plot_proprio(dir):
    sorted = sort_set(dir)
    i = 0
    for file in sorted:
        with h5py.File(file, "r") as f:
            key = 'proprio'
            proprio = f[key][:]
            if i == 0:
                concat_pro = proprio
                i +=1
            else:
                concat_pro = np.concatenate((concat_pro, proprio))

    print(f'Proprio Shape: {concat_pro.shape}')


    x = concat_pro[:200, 0][::2]
    y = concat_pro[:200, 1][::2]
    z = concat_pro[:200, 2][::2]
    yaw = concat_pro[:200, 3][::2]

    timesteps = np.arange(len(x))

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': '3d'})

    # 3D Scatter Plot
    ax1 = fig.add_subplot(121, projection='3d')
    scatter = ax1.scatter(x, y, z, c='b', marker='o')
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_zlabel('Z axis')
    ax1.set_title('3D Scatter Plot of XYZ Coordinates (Every Second Value)')

    # Plot Yaw over Timesteps
    ax2 = fig.add_subplot(122)
    ax2.plot(timesteps, yaw, 'r-o')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Yaw')
    ax2.set_title('Yaw Over Timesteps (Every Second Value)')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show plot
    plt.show()


def crop_video_square(file):
    with h5py.File(file, "r") as f:
        key = "fixed_view_left"
        images = f[key][:]

        # RGB is flipped to BGR
        plt.imshow(images[0])
        plt.title(f'{key} from {file}')
        plt.show()

        steps, height, width, channels = images.shape
        print(images.shape)

        if not height == width:
            idx = choose_start(images[0])
            print(f'This is the index: {idx}')

            images = images[:, :, idx:(idx+height), :]
            images = images[:, 20:-20, 20:-20, :]
            print(f'NewShape: {images.shape}')

        # generate video from h5
        size = images.shape[1], images.shape[1]
        # duration = 2
        fps = 30
        save_dir = r"/home/philipp/Uni/14_SoSe/IRM_Prac_2/vid"
        # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
        out = cv2.VideoWriter(f'{save_dir}_CroppedPad.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
        for img in images:
            # data = np.random.randint(0, 256, size, dtype='uint8')
            # img_swap = np.swapaxes(img, 0, 2)
            out.write(img)
        out.release()

# helps to decide the visible frame of the cropped video
def choose_start(frame):
    height, width, channels = frame.shape

    mouseX = None
    mouseY = None

    def get_mouse_coordinates(event, x, y, flags, param):
        nonlocal mouseX, mouseY
        if event == cv2.EVENT_LBUTTONDOWN:
            mouseX, mouseY = x, y
            redDotDrawn = False
    orig_frame = copy.deepcopy(frame)
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", get_mouse_coordinates)
    while True:
        cv2.imshow("Frame", frame)

        if mouseX is not None and mouseY is not None:
            cv2.line(frame, (mouseX, 0), (mouseX, height-1), color=(0, 0, 255), thickness=3)
            cv2.line(frame, (mouseX+height, 0), (mouseX+height, height-1), color=(0, 0, 255), thickness=3)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to quit
            break
        if key == ord('y'):
            cv2.destroyAllWindows()
            return mouseX
        if key == ord('r'):  # Press 'r' to reset
            mouseX = None
            mouseY = None
            frame = copy.deepcopy(orig_frame)

    cv2.destroyAllWindows()

def extract_h5_data(files):
    for file in files:
        with h5py.File(file, "r") as f:
            print("-----------------------------------------------------")
            print(f"keys in file: {file}:")
            for key, value in f.items():
                print(f"key: {key}")
                print(f"value: {value}")
            print("-----------------------------------------------------")

            key = "fixed_view_left"
            images = f[key][:]
            print(f'IMG: {images.shape}')

            struc_keys = ['j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6']
            key = "tau"
            tau = f[key][:]
            print(f'Tau: {tau["j6"].shape}')

            # Create a new figure
            plt.figure(figsize=(12, 8))

            for i, data_list in enumerate(struc_keys):
                plt.plot(tau[data_list], label=struc_keys[i])

            # Add title and labels
            plt.title('Plot key values along number of recordings')
            plt.xlabel('Recordings')
            plt.ylabel('Value')

            # Add legend
            plt.legend()

            # Show plot with grid
            plt.grid(True)
            plt.show()

def read_triangle_data(file):
    with h5py.File(file, "r") as f:
        print("-----------------------------------------------------")
        print(f"keys in file: {file}:")
        for key, value in f.items():
            print(f"key: {key}")
            print(f"value: {value}")
            data = f[key][:4]
            print(f'This is the data: {data}')
        print("-----------------------------------------------------")

def list_files_in_directory(directory):
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            file_list.append(full_path)
    # print(file_list)
    return file_list

def sort_set(dir):
    set = '1'
    files_l = list_files_in_directory(dir)
    sorted = [''] * 20
    # print(f'This is: {np.array(files_l)}')
    files = [files_l[0]]
    for i in range(len(files_l)):
        filename = files_l[i][:-8]
        # print(f'List: {files_l[i]}')
        # print(f'Filename: {filename}')
        if filename[-2] == '_':
            if filename[-3] == set:
                num = int(filename[-1])
                sorted[num] = files_l[i]
        else:
            if filename[-4] == set:
                num = int(filename[-2:])
                sorted[num] = files_l[i]

    print(f'Sorted: {sorted}')
    return sorted

def vid_TrueFlow(dir):
    sorted = sort_set(dir)
    i = 0
    for file in sorted:
        with h5py.File(file, "r") as f:
            key = "optical_flow"
            images = f[key][:]

            if i == 0:
                concat_img = images
                i +=1
            else:
                concat_img = np.concatenate((concat_img, images))

            steps, height, width, channels = images.shape

    print(concat_img.shape)
    print(concat_img[400, 60:68, 60:68, :])
    size = height, height
    fps = 30
    if os.name == 'posix':
        save_dir = r"/home/philipp/Uni/14_SoSe/IRM_Prac_2/vid"
    elif os.name == 'nt':
        save_dir = r"C:\Rest\Uni\14_SoSe\IRM_Prac_2\data_test\vid"
    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    out = cv2.VideoWriter(f'{save_dir}TrueOpticalFlow.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
    for flow in concat_img:  # flow is of shape (height, width, 2)
        # Calculate the magnitude and angle of the flow
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Normalize the magnitude to fit into the range [0, 1]
        magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)

        # Create an HSV image
        hsv = np.zeros((height, width, 3), dtype=np.float32)
        hsv[..., 0] = angle * 180 / np.pi / 2  # Hue (0 - 180)
        hsv[..., 1] = 1  # Saturation
        hsv[..., 2] = magnitude  # Value (Brightness)

        # Convert HSV to RGB (or BGR in OpenCV)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Convert the image back to 8-bit
        rgb = np.uint8(rgb * 255)

        out.write(rgb)
    out.release()
    print(f'vid size: {concat_img.shape}')



# sorts a dataset video in the correct order and returns a list
def own_flow(dir):
    triangle = False
    if triangle:
        sorted = sort_set(dir)
        i = 0
        for file in sorted:
            with h5py.File(file, "r") as f:
                key = "image"
                images = f[key][:]
                depths = f['depth_data'][:]

                if i == 0:
                    concat_img = images
                    concat_depth = depths
                    i +=1
                else:
                    concat_img = np.concatenate((concat_img, images))
                    concat_depth = np.concatenate((concat_depth, depths))

                steps, height, width, channels = images.shape
    else:
        with h5py.File(dir, "r") as f:
            concat_img = f['fixed_view_left'][:]
            concat_depth = f['fixed_view_left_depth'][:]

            steps, height, width, channels = concat_img.shape

            idx = 140

        concat_img = concat_img[:, :, idx:(idx+height), :]
        concat_img = concat_img[:, 20:-20, 20:-20, :]
        concat_depth = concat_depth[:, :, idx:(idx+height)]
        concat_depth = concat_depth[:, 20:-20, 20:-20]

        downsampled_images = []
        downsampled_depths = []
        for i in range(len(concat_img)):
            # Resize the frame using cv2.resize
            resized_img_frame = cv2.resize(concat_img[i], (128, 128), interpolation=cv2.INTER_LINEAR)
            resized_depth_frame = cv2.resize(concat_depth[i], (128, 128), interpolation=cv2.INTER_LINEAR)
            downsampled_images.append(resized_img_frame)
            downsampled_depths.append(resized_depth_frame)

        concat_img = np.array(downsampled_images)
        concat_depth = np.array(downsampled_depths)

        steps, height, width, channels = concat_img.shape


    # print(concat_img.shape)

    for i in tqdm(range(concat_img.shape[0])):
        concat_depth[i] = np.nan_to_num(concat_depth[i], nan=0.0, posinf=12.0, neginf=0.0)
        mask = np.where((concat_depth[i] >= 0.3) & (concat_depth[i] < 1), 1, 0).astype(np.uint8)

        # threshold for dark pixels
        dark_threshold = 80
        non_dark_pixels_mask = np.any(concat_img[i] >= dark_threshold, axis=2)

        # Create a mask where non-dark pixels are set to 1, and dark pixels are set to 0
        dark_mask = non_dark_pixels_mask.astype(np.uint8)
        # print(f'MASK [{mask.shape}')
        # print(f'DArk MASK [{dark_mask.shape}')

        if triangle:
            # Expand the mask to shape (128 , 128, 3) to match concat_img[i]
            mask_expanded = np.repeat(mask, 3, axis=2)
            dark_mask_expanded = dark_mask[:, :, np.newaxis]
            dark_mask_expanded = np.repeat(dark_mask_expanded, 3, axis=2)
        else:
            mask_expanded = mask[:, :, np.newaxis]  # Shape: (128, 128, 1)
            mask_expanded = np.tile(mask_expanded, (1, 1, 3))  # Shape: (128, 128, 3)
            dark_mask_expanded = dark_mask[:, :, np.newaxis]
            dark_mask_expanded = np.tile(dark_mask_expanded, (1, 1, 3))
        # print(f"Shape of concat_img[{i}]:", concat_img[i].shape)  # Should be (128, 128, 3)
        # print(f"Shape of mask_expanded for index {i}:", mask_expanded.shape)
        # print(f'Shape Mask{dark_mask_expanded.shape}')

        # Apply the mask: keep RGB where mask == 1, otherwise set to black
        concat_img[i] = concat_img[i] * mask_expanded
        concat_img[i] = concat_img[i] * dark_mask_expanded
    # print(concat_img.shape)

    vid = calc_optical_flow(concat_img)
    print(vid.shape)
    print(vid[400, 60:68, 60:68, :])
    opt_flow = True
    size = vid.shape[1], vid.shape[2]
    # duration = 2
    fps = 30
    if os.name == 'posix':
        save_dir = r"/home/philipp/Uni/14_SoSe/IRM_Prac_2/vid"
    elif os.name == 'nt':
        save_dir = r"C:\Rest\Uni\14_SoSe\IRM_Prac_2\data_test\vid"

    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    out = cv2.VideoWriter(f'{save_dir}_ownFlow.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
    if not opt_flow:
        for img in vid:
            # data = np.random.randint(0, 256, size, dtype='uint8')
            # img_swap = np.swapaxes(img, 0, 2)
            out.write(img)
    else:
        for flow in tqdm(vid):  # flow is of shape (height, width, 2)
            # Calculate the magnitude and angle of the flow
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Normalize the magnitude to fit into the range [0, 1]
            magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)

            # Create an HSV image
            hsv = np.zeros((height, width, 3), dtype=np.float32)
            hsv[..., 0] = angle * 180 / np.pi / 2  # Hue (0 - 180)
            hsv[..., 1] = 1  # Saturation
            hsv[..., 2] = magnitude  # Value (Brightness)

            # Convert HSV to RGB (or BGR in OpenCV)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Convert the image back to 8-bit
            rgb = np.uint8(rgb * 255)

            out.write(rgb)
    out.release()
    print(f'vid size: {vid.shape}')

def calc_optical_flow(images):
    # Number of frames
    num_frames = images.shape[0]

    # Initialize a list to store the optical flow for each pair of frames
    optical_flows = []

    for i in tqdm(range(1, num_frames)):
        # Convert frames to grayscale if they are not already
        prev_frame = cv2.cvtColor(images[i - 1], cv2.COLOR_BGR2GRAY)
        next_frame = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

        # Calculate the optical flow using Farneback's method
        flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2,
                                            flags=0)

        # Append the optical flow to the list
        optical_flows.append(flow)

    # Convert the list to a numpy array for easier handling
    optical_flows = np.array(optical_flows)

    # normalize optical flow and smooth with gaussian filter
    for i in range(optical_flows.shape[0]):
        optical_flows[i] = normalize_optical_flow(optical_flows[i])

    # smoothe flow over 3 frames
    thresh_flow = threshold_flow(optical_flows)
    med_flow = apply_median_filter(thresh_flow)
    # print(f'Shape: {smoothed_frames.shape}')
    return med_flow

def threshold_flow(flow, threshold=0.0):
        # Compute the magnitude of the flow
        magnitude = np.sqrt(np.sum(flow ** 2, axis=-1, keepdims=True))
        # Create mask to fit dimensions to apply threshold
        mask = magnitude < threshold
        # Apply threshold: zero out flow vectors with small magnitudes
        flow[mask.repeat(2, axis=-1)] = 0
        return flow

def apply_median_filter(flow_frames, size=3):
    filtered_flow = np.copy(flow_frames)
    for i in range(2):
        filtered_flow[..., i] = median_filter(flow_frames[..., i], size=size)
    return filtered_flow

def normalize_optical_flow(flow):
    # Compute the magnitude of the flow vectors
    magnitude = np.sqrt(np.sum(flow ** 2, axis=-1, keepdims=True))  # shape (H, W, 1)

    # Find the maximum magnitude across the entire flow field
    max_magnitude = np.max(magnitude)

    if max_magnitude > 0:
        # Normalize the flow by dividing by the maximum magnitude
        normalized_flow = flow / max_magnitude
    else:
        # If the maximum magnitude is 0, the flow is all zero; return it as-is
        normalized_flow = flow

    return normalized_flow

def concat_vid(dir):
    sorted = sort_set(dir)
    i = 0
    for file in sorted:
        with h5py.File(file, "r") as f:
            key = "image"
            images = f[key][:]

            if i == 0:
                vid = images
                i +=1
            else:
                vid = np.concatenate((vid, images))

            steps, height, width, channels = images.shape

    opt_flow = False
    # vid = vid[:200]
    size = height, height
    # duration = 2
    fps = 30
    if os.name == 'posix':
        save_dir = r"/home/philipp/Uni/14_SoSe/IRM_Prac_2/vid"
    elif os.name == 'nt':
        save_dir = r"C:\Rest\Uni\14_SoSe\IRM_Prac_2\data_test\vid"
    # print('Reached')
    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    out = cv2.VideoWriter(f'{save_dir}_concatVidImg.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
    if not opt_flow:
        for img in vid:
            # data = np.random.randint(0, 256, size, dtype='uint8')
            # img_swap = np.swapaxes(img, 0, 2)
            out.write(img)
    else:
        for flow in vid:  # flow is of shape (height, width, 2)
            # Calculate the magnitude and angle of the flow
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Normalize the magnitude to fit into the range [0, 1]
            magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)

            # Create an HSV image
            hsv = np.zeros((height, width, 3), dtype=np.float32)
            hsv[..., 0] = angle * 180 / np.pi / 2  # Hue (0 - 180)
            hsv[..., 1] = 1  # Saturation
            hsv[..., 2] = magnitude  # Value (Brightness)

            # Convert HSV to RGB (or BGR in OpenCV)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Convert the image back to 8-bit
            rgb = np.uint8(rgb * 255)

            out.write(rgb)
    out.release()
    print(f'vid size: {vid.shape}')

def save_img(file):
    with h5py.File(file, "r") as f:
        depths = f["fixed_view_left_depth"][:]
        imgs = f["fixed_view_left"][:]

        save_dir = r"/home/philipp/Uni/14_SoSe/IRM_Prac_2/"

        depth = depths[250]
        img = imgs[250]

        depth = np.nan_to_num(depth, nan=0.0, posinf=255.0, neginf=0.0)

        # 2. Normalize the depth values to [0, 255] range (assuming your values range from 0 to 1)
        depth_normalized = (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) * 255.0

        # 3. Clip the values to [0, 255] to avoid overflow or underflow
        depth_normalized = np.clip(depth_normalized, 0, 255)

        # 4. Cast the result to 8-bit unsigned integer
        depth_uint8 = np.uint8(depth_normalized)

        cv2.imwrite('/home/philipp/Uni/14_SoSe/IRM_Prac_2/depth.png', depth_uint8)
        cv2.imwrite('/home/philipp/Uni/14_SoSe/IRM_Prac_2/color.png', img)





def plot_tau(file):
    with h5py.File(file, "r") as f:
        tau = f['tau'][:]
        ext_tau = f['ext_tau'][:]
        plt.figure(1, figsize=(10, 6))

        # Define color maps for the two datasets
        colors1 = plt.cm.Blues(np.linspace(0.4, 1, 7))  # Color range for data1
        colors2 = plt.cm.Reds(np.linspace(0.4, 1, 7))  # Color range for data2

        internal_tau = tau - ext_tau
        ext_sum = np.sum(np.abs(ext_tau), axis=1)
        # Plot the lines from the first dataset
        # for i in range(7):
        #     plt.plot(tau[:, i], color=colors1[i], label=f'Data1 - Line {i + 1}')

        # Plot the lines from the second dataset
        for i in range(7):
            plt.plot(ext_tau[:, i], color=colors2[i], linestyle='--', label=f'ExtTau{i + 1}')
        plt.plot(ext_sum, color=colors1[0], label='Summed Tau')


        # Add labels and a legend
        plt.xlabel('Steps')
        plt.ylabel('Force')
        plt.title('Plot external forces and its summed absolute')
        plt.legend(loc='upper right')

        plt.figure(2, figsize=(10, 6))

        for i in range(7):
            plt.plot(np.abs(tau[:, i]), color=colors1[i], label=f'Tau{i+1}')
        for i in range(7):
            plt.plot(np.abs(internal_tau[:, i]), color=colors2[i], linestyle='--', label=f'Internal Tau{i+1}')

        plt.plot(ext_sum, color='green', label='Summed Tau')
        plt.plot(np.full(ext_sum.shape[0], 7), color='black',linestyle='--', label='Threshold')

        # Add labels and a legend
        plt.xlabel('Steps')
        plt.ylabel('Force')
        plt.title('Plot Internal forces, general forces and the summed external forces as well as the threshold')
        plt.legend(loc='upper right')

        # Show the plot
        plt.show()

def save_video(file):
    with h5py.File(file, "r") as f:
        color_bool = True
        if color_bool:
            key = "fixed_view_left"
        else:
            key = "fixed_view_left_depth"
        images = f[key][:]  # Assuming depth images are stored as single-channel

        steps, height, width = images.shape[:3]  # channels should be 1 for single-channel images
        # assert channels == 1, "Depth images should have only one channel."

        print(images[1000, 340, :])

        size = (width, height)
        fps = 30
        save_dir = r"/home/philipp/Uni/14_SoSe/IRM_Prac_2/flagsNdepth"
        out = cv2.VideoWriter(f'{save_dir}_TrueVid.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size, isColor=color_bool)

        for img in images:
            if not color_bool:
                # 1. Replace any NaN or Inf values with a valid number (e.g., 0)
                img = np.nan_to_num(img, nan=0.0, posinf=12.0, neginf=0.0)

                # 2. Normalize the depth values to [0, 255] range (assuming your values range from 0 to 1)
                img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

                # 3. Clip the values to [0, 255] to avoid overflow or underflow
                img_normalized = np.clip(img_normalized, 0, 255)

                # 4. Cast the result to 8-bit unsigned integer
                img_uint8 = np.uint8(img_normalized)

                out.write(img_uint8)
            else:
                out.write(img)

        out.release()



if __name__ == "__main__":
    joint_struc_keys = ['j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6']
    cart_struc_keys = ['x', 'y', 'z']
    if os.name == 'posix':
        print('Using Linux')
        # files = [r"/home/philipp/Uni/14_SoSe/IRM_Prac_2/triangle_data/20190408_183047_triangle_0_journal_1_0_1000.h5"]
        files = [r"/home/philipp/Uni/14_SoSe/IRM_Prac_2/dataset", r"/home/philipp/Uni/14_SoSe/IRM_Prac_2/flagsNdepth/7.h5",
                 r"/home/philipp/Uni/14_SoSe/IRM_Prac_2/recordings/interesting_f.h5"]

        # concat_vid(r"/home/philipp/Uni/14_SoSe/IRM_Prac_2/triangle_data")
        # own_flow(r"/home/philipp/Uni/14_SoSe/IRM_Prac_2/triangle_data")
        # crop_video_square(files[1])
        own_flow(files[1])
        # vid_TrueFlow(r"/home/philipp/Uni/14_SoSe/IRM_Prac_2/triangle_data")
        # plot_joint_values(files[0], "tau", joint_struc_keys)
        # extract_h5_data(files)
        # read_triangle_data(files[0])
        # plot_tau(files[1])
        save_video(files[1])
        # save_img(files[1])
    elif os.name == 'nt':
        print('Using Windows')
        files = [r"C:\Rest\Uni\14_SoSe\IRM_Prac_2\data_test\test\3.h5",
                 r"C:\Rest\Uni\14_SoSe\IRM_Prac_2\data_test\triangle_real_data\triangle_real_data",
                 r"C:\Rest\Uni\14_SoSe\IRM_Prac_2\data_test\new_dataset"
                 ]
        vid_TrueFlow(files[2])
        # own_flow(r"C:\Rest\Uni\14_SoSe\IRM_Prac_2\data_test\new_dataset")
        # own_flow(files[0])
        concat_vid(files[2])
        # plot_proprio(r"C:\Rest\Uni\14_SoSe\IRM_Prac_2\data_test\triangle_real_data\triangle_real_data")
        # concat_vid(r"C:\Rest\Uni\14_SoSe\IRM_Prac_2\data_test\triangle_real_data\triangle_real_data")
    else:
        print(f'Os name not recognized: {os.name}')



