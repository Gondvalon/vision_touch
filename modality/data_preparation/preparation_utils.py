import copy

import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

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

def crop_video_square(file):
    with h5py.File(file, "r") as f:
        key = "image"
        images = f[key][:]

        # RGB is flipped to BGR
        plt.imshow(images[0])
        plt.title(f'{key} from {file}')
        plt.show()

        steps, height, width, channels = images.shape
        print(images.shape)

        if not height == width:
            idx = choose_start(images[0])

            images = images[:, :, idx:(idx+height), :]
            print(f'NewShape: {images.shape}')

        # generate video from h5
        size = height, height
        # duration = 2
        fps = 30
        # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
        out = cv2.VideoWriter(f'{file}_view.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
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

def extract_h5_data(file):
    for file in files:
        with h5py.File(file, "r") as f:
            print("-----------------------------------------------------")
            print(f"keys in file: {file}:")
            for key, value in f.items():
                print(f"key: {key}")
                print(f"value: {value}")
            print("-----------------------------------------------------")

            key = "fixed_view_right"
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

# sorts a dataset video in the correct order and returns a list
def own_flow(dir):
    set = '1'
    files_l = list_files_in_directory(dir)
    sorted = ['']*20
    # print(f'This is: {np.array(files_l)}')
    files = [files_l[0]]
    for i in range(len(files_l)):
        filename = files_l[i][:-8]
        print(f'List: {files_l[i]}')
        print(f'Filename: {filename}')
        if filename[-2] == '_':
            if filename[-3] == set:
                num = int(filename[-1])
                sorted[num] = files_l[i]
        else:
            if filename[-4] == set:
                num = int(filename[-2:])
                sorted[num] = files_l[i]

    print(f'Sorted: {sorted}')
    i = 0
    for file in sorted:
        with h5py.File(file, "r") as f:
            key = "image"
            images = f[key][:]

            if i == 0:
                concat_img = images
                i +=1
            else:
                concat_img = np.concatenate((concat_img, images))

            steps, height, width, channels = images.shape

    vid = calc_optical_flow(concat_img)
    print(vid.shape)
    opt_flow = True
    size = height, height
    # duration = 2
    fps = 60
    save_dir = r"/home/philipp/Uni/14_SoSe/IRM_Prac_2/vid"
    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    out = cv2.VideoWriter(f'{save_dir}_ownFlow.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
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





def calc_optical_flow(images):
    # Number of frames
    num_frames = images.shape[0]

    # Initialize a list to store the optical flow for each pair of frames
    optical_flows = []

    for i in range(1, num_frames):
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
    return optical_flows

def concat_vid(dir):

    files_l = list_files_in_directory(dir)
    # print(f'This is: {np.array(files_l)}')
    # files = [files_l[0]]
    # files = np.concatenate(([files_l[20]], files_l[31:40], files_l[21:31]))
    files = np.concatenate(([files_l[33]], [files_l[29]], [files_l[18]], [files_l[45]], [files_l[5]], [files_l[0]], [files_l[17]]))
    print(files)
    i = 0
    for file in files:
        with h5py.File(file, "r") as f:
            key = "optical_flow"
            images = f[key][:]

            if i == 0:
                vid = images
                i +=1
            else:
                vid = np.concatenate((vid, images))

            steps, height, width, channels = images.shape

    opt_flow = True
    size = height, height
    # duration = 2
    fps = 60
    save_dir = r"/home/philipp/Uni/14_SoSe/IRM_Prac_2/vid"
    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    out = cv2.VideoWriter(f'{save_dir}_Flowview.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
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





if __name__ == "__main__":
    # files = [r"/home/philipp/Uni/14_SoSe/IRM_Prac_2/triangle_data/20190408_183047_triangle_0_journal_1_0_1000.h5"]
    files = [r"/home/philipp/Uni/14_SoSe/IRM_Prac_2/dataset"]

    joint_struc_keys = ['j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6']
    # cart_struc_keys = ['x', 'y', 'z']
    # extract_h5_data(files)
    # plot_joint_values(files[0], "tau", joint_struc_keys)
    # crop_video_square(files[0])
    # concat_vid(r"C:\Rest\Uni\14_SoSe\IRM_Prac_2\data_test\triangle_real_data\triangle_real_data")
    # concat_vid(files[0])
    own_flow(r"/home/philipp/Uni/14_SoSe/IRM_Prac_2/triangle_data")
    # read_triangle_data(files[0])
