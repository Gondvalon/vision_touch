import h5py
import numpy as np
import cv2
import os
import sys

from tqdm import tqdm

class CreateDatasetFromRecord():
    def __init__(self, dataset_dir, recording_dir):
        self.dataset_dir = dataset_dir
        self.recording_dir = recording_dir

        self.allow_short_set = True
        # starting index to make the frames square, goes from this index + height
        self.square_idx = 0

        self.dataset_name = 'pegInsert'
        self.num_recordings = 0
        self.num_subfiles = 0
        self.data_per_file = 50

        self.keys = ["fixed_view_right", "q", "dq", "positions", "orientations", "tau"]


        # list of all files in dir that end with .h5
        self.recordings_l = [os.path.join(self.recording_dir, f) for f in os.listdir(self.recording_dir) if f.endswith('.h5')]
        print(self.recordings_l)
        if len(self.recordings_l) <= 0:
            print(f"Directory '{self.recording_dir}' seems to have no recordings")
            sys.exit()

        # create location for where to save the dataset
        if not os.path.isdir(self.dataset_dir):
            os.makedirs(self.dataset_dir, exist_ok=True)
            print(f"Directory '{self.dataset_dir}' created successfully!")

        self.create_dataset(self.recordings_l)

    def create_dataset(self, files):
        # iterate over all recordings
        self.num_recordings = len(files)
        for i, file in tqdm(enumerate(files), desc= 'Processing recordings'):
            with h5py.File(file, "r") as f:
                images = f[self.keys[0]][:]

                # shape of recordings is frames, height, width, channels
                num_entries = images.shape[0]
                # there must be 1000 datapoints to create dataset properly
                if num_entries <= 1000 and not self.allow_short_set:
                    continue
                self.num_subfiles = min(num_entries // 50, 20)
                print(f'There will get {self.num_subfiles} sub files created')

                # reshape and downsample images
                images = images[:1001, :, self.square_idx:(self.square_idx + images.shape[1]), :]
                downsampled_images = []
                for frame in images:
                    # Resize the frame using cv2.resize
                    resized_frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_LINEAR)
                    downsampled_images.append(resized_frame)

                joint_pos_struc = f[self.keys[1]][:]
                print(f'Shape: {joint_pos_struc[0]}')
                joint_vel_struc = f[self.keys[2]][:]
                ee_pos_struc = f[self.keys[3]][:]
                ee_ori_struc = f[self.keys[4]][:]

                # convert structs into np arrays
                joint_pos_l = []
                joint_vel_ori_l = []
                ee_pos_l = []
                ee_ori_l = []
                print(len(joint_pos_struc['j0']))
                for j in range(num_entries + 0):
                    joint_pos = [joint_pos_struc['j0'][j], joint_pos_struc['j1'][j], joint_pos_struc['j2'][j], joint_pos_struc['j3'][j],
                                 joint_pos_struc['j4'][j], joint_pos_struc['j5'][j], joint_pos_struc['j6'][j]]
                    joint_vel_ori = [joint_vel_struc['j0'][j], joint_vel_struc['j1'][j], joint_vel_struc['j2'][j], joint_vel_struc['j3'][j],
                                 joint_vel_struc['j4'][j], joint_vel_struc['j5'][j], joint_vel_struc['j6'][j]]
                    ee_pos = [ee_pos_struc['x'][j], ee_pos_struc['y'][j], ee_pos_struc['z'][j]]
                    ee_ori = [ee_ori_struc['x'][j], ee_ori_struc['y'][j], ee_ori_struc['z'][j],ee_ori_struc['w'][j]]

                    joint_pos_l.append(joint_pos)
                    # print(f'Shape joints: {joint_pos_l[0].size}')
                    joint_vel_ori_l.append(joint_vel_ori)
                    ee_pos_l.append(ee_pos)
                    ee_ori_l.append(ee_ori)

                images = np.array(downsampled_images)
                joint_pos_np = np.array(joint_pos_l)
                joint_vel_ori_np = np.array(joint_vel_ori_l)
                # print(f'Shape joints: {joint_pos_l[0].shape}')
                ee_pos_np = np.array(ee_pos_l)
                ee_ori_np = np.array(ee_ori_l)
                optical_flow_np = self.calc_optical_flow(images)

                # iterate over the number of files that shall get created for the dataset
                for j in range(self.num_subfiles):
                    filename = self.dataset_name + "_" + str(i) + "_" + str(j) + "_1000.h5"
                    filepath = os.path.join(self.dataset_dir, filename)
                    with h5py.File(filepath, 'w') as h5file:
                        # create datasets
                        h5file.create_dataset('image', data=images[j * self.data_per_file:(j+1) * self.data_per_file])
                        h5file.create_dataset('optical_flow', data=optical_flow_np[j * self.data_per_file:(j+1) * self.data_per_file])
                        h5file.create_dataset('joint_pos', data=joint_pos_np[j * self.data_per_file:(j+1) * self.data_per_file])
                        h5file.create_dataset('joint_vel_ori', data=joint_vel_ori_np[j * self.data_per_file:(j+1) * self.data_per_file])
                        h5file.create_dataset('ee_pos', data=ee_pos_np[j * self.data_per_file:(j+1) * self.data_per_file])
                        h5file.create_dataset('ee_ori', data=ee_ori_np[j * self.data_per_file:(j+1) * self.data_per_file])
                        # h5file.create_dataset('dataset2', data=data2, dtype='float64')

        print("Dataset successfully created.")

    def calc_optical_flow(self, images):
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





if __name__ == "__main__":
    DATASET_DIR = r'C:\Rest\Uni\14_SoSe\IRM_Prac_2\data_test\new_dataset'
    RECORDING_DIR = r'C:\Rest\Uni\14_SoSe\IRM_Prac_2\data_test'

    create = CreateDatasetFromRecord(DATASET_DIR, RECORDING_DIR)