import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt


files = [r"C:\Rest\Uni\14_SoSe\IRM_Prac_2\7_best.h5"]

class readhd5:
    def __init__(self, data_directory, logger = None):
        self.dataset = data_directory



def extract_camera_to_mp4(files):
    for file in files:
        with h5py.File(file, "r") as f:
            print("-----------------------------------------------------")
            print(f"keys in file: {file}:")
            for key, value in f.items():
                print(f"key: {key}")
                print(f"value: {value}")
            print("-----------------------------------------------------")

            key = "fixed_view_right"
            images = f[key][()]

            # RGB is flipped to BGR
            plt.imshow(images[150])
            plt.title(f'{key} from {file}')
            plt.show()

            steps, height, width, channels = images.shape
            print(images.shape)

            # generate video from h5
            size = height, width
            # duration = 2
            fps = 60
            # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
            out = cv2.VideoWriter(f'{file}_view.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
            for img in images:
                # data = np.random.randint(0, 256, size, dtype='uint8')
                # img_swap = np.swapaxes(img, 0, 2)
                out.write(img)
            out.release()

        # If a_group_key is a group name,
        # this gets the object names in the group and returns as a list
        # data = list(f[a_group_key])
        #
        # # If a_group_key is a dataset name,
        # # this gets the dataset values and returns as a list
        # data = list(f[a_group_key])
        # # preferred methods to get dataset values:
        # ds_obj = f[a_group_key]      # returns as a h5py dataset object
        # ds_arr = f[a_group_key][()]  # returns as a numpy array
        print("final")

if __name__ == "__main__":
        files = [r"C:\Rest\Uni\14_SoSe\IRM_Prac_2\7_best.h5"]

        extract_camera_to_mp4(files)

