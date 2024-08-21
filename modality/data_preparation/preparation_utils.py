import copy

import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt

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
        key = "fixed_view_right"
        images = f[key][:]

        # RGB is flipped to BGR
        plt.imshow(images[400])
        plt.title(f'{key} from {file}')
        plt.show()

        steps, height, width, channels = images.shape
        print(images.shape)

        idx = choose_start(images[0])

        images = images[:, :, idx:(idx+height), :]
        print(f'NewShape: {images.shape}')

        # generate video from h5
        size = height, height
        # duration = 2
        fps = 60
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




if __name__ == "__main__":
    files = [r"C:\Rest\Uni\14_SoSe\IRM_Prac_2\7_best.h5"]

    joint_struc_keys = ['j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6']
    cart_struc_keys = ['x', 'y', 'z']
    # extract_h5_data(files)
    # plot_joint_values(files[0], "tau", joint_struc_keys)
    crop_video_square(files[0])

