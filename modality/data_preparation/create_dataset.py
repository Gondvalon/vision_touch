import h5py
import numpy as np
import cv2
import os
import sys

from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import median_filter

class CreateDatasetFromRecord():
    def __init__(self, dataset_dir, recording_dir):
        self.dataset_dir = dataset_dir
        self.recording_dir = recording_dir

        self.allow_short_set = True
        # starting index to make the frames square, goes from this index + height
        self.square_idx = 0
        # frequency which was used to collect the data
        self.collection_freq = 30

        self.dataset_name = 'pegInsert'
        self.num_recordings = 0
        self.num_subfiles = 0
        self.data_per_file = 50

        self.keys = ["fixed_view_left", "q", "dq", "positions", "orientations", "tau", "ext_tau", "fixed_view_left_depth"]


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
                if num_entries <= 1001 and not self.allow_short_set:
                    continue
                else:
                    num_entries = num_entries - 1
                self.num_subfiles = min((num_entries-1) // 50, 20)
                print(f'There will get {self.num_subfiles} sub files created')

                # reshape and downsample images
                images = images[:(self.num_subfiles * 50) + 1, :, self.square_idx:(self.square_idx + images.shape[1]), :]
                downsampled_images = []
                for frame in images:
                    # Resize the frame using cv2.resize
                    resized_frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_LINEAR)
                    downsampled_images.append(resized_frame)

                joint_pos_struc = f[self.keys[1]][:]
                joint_vel_struc = f[self.keys[2]][:]
                ee_pos_struc = f[self.keys[3]][:]
                ee_ori_struc = f[self.keys[4]][:]
                tau_struc = f[self.keys[5]][:]

                # convert structs into np arrays
                joint_pos_l = []
                joint_vel_ori_l = []
                ee_pos_l = []
                ee_ori_l = []
                tau_l = []
                for j in range(num_entries + 1):
                    joint_pos = [joint_pos_struc['j0'][j], joint_pos_struc['j1'][j], joint_pos_struc['j2'][j], joint_pos_struc['j3'][j],
                                 joint_pos_struc['j4'][j], joint_pos_struc['j5'][j], joint_pos_struc['j6'][j]]
                    joint_vel_ori = [joint_vel_struc['j0'][j], joint_vel_struc['j1'][j], joint_vel_struc['j2'][j], joint_vel_struc['j3'][j],
                                 joint_vel_struc['j4'][j], joint_vel_struc['j5'][j], joint_vel_struc['j6'][j]]
                    tau = [tau_struc['j0'][j], tau_struc['j1'][j], tau_struc['j2'][j], tau_struc['j3'][j], tau_struc['j4'][j],
                           tau_struc['j5'][j], tau_struc['j6'][j]]
                    ee_pos = [ee_pos_struc['x'][j], ee_pos_struc['y'][j], ee_pos_struc['z'][j]]
                    ee_ori = [ee_ori_struc['x'][j], ee_ori_struc['y'][j], ee_ori_struc['z'][j],ee_ori_struc['w'][j]]

                    joint_pos_l.append(joint_pos)
                    joint_vel_ori_l.append(joint_vel_ori)
                    tau_l.append(tau)
                    ee_pos_l.append(ee_pos)
                    ee_ori_l.append(ee_ori)

                images = np.array(downsampled_images)
                joint_pos_np = np.array(joint_pos_l)
                joint_vel_ori_np = np.array(joint_vel_ori_l)
                tau_np = np.array(tau_l)
                ee_pos_np = np.array(ee_pos_l)
                ee_ori_np = np.array(ee_ori_l)
                ee_ori_np = ee_ori_np / np.linalg.norm(ee_ori_np, axis=1, keepdims=True)  # Normalize quaternions

                optical_flow_np = self.calc_optical_flow(images)

                ee_vel_np, ee_vel_ori_np, ee_pos_diff = self.calc_ee_velo(ee_pos_np, ee_ori_np)
                # yaw is in euler angles(radians) while the other two are quaternions
                yaw_np, yaw_vel_np, yaw_diff_np, ee_yaw_np, ee_yaw_delta_np = self.calc_yaw_data(ee_ori_np)

                # create action and proprioception values
                action_np = np.hstack((ee_pos_diff, yaw_diff_np))
                proprio_np = np.hstack((ee_pos_np[:-1], yaw_np[:-1], ee_vel_np, yaw_vel_np))


                # iterate over the number of files that shall get created for the dataset
                for j in range(self.num_subfiles):
                    filename = self.dataset_name + "_" + str(i) + "_" + str(j) + "_1000.h5"
                    filepath = os.path.join(self.dataset_dir, filename)
                    with h5py.File(filepath, 'w') as h5file:
                        # create datasets
                        h5file.create_dataset('proprio', data=proprio_np[j * self.data_per_file:(j + 1) * self.data_per_file])
                        h5file.create_dataset('action', data=action_np[j * self.data_per_file:(j + 1) * self.data_per_file])
                        h5file.create_dataset('tau', data=tau_np[j * self.data_per_file:(j + 1) * self.data_per_file])
                        h5file.create_dataset('image', data=images[j * self.data_per_file:(j+1) * self.data_per_file])
                        h5file.create_dataset('optical_flow', data=optical_flow_np[j * self.data_per_file:(j+1) * self.data_per_file])
                        h5file.create_dataset('joint_pos', data=joint_pos_np[j * self.data_per_file:(j+1) * self.data_per_file])
                        h5file.create_dataset('joint_vel_ori', data=joint_vel_ori_np[j * self.data_per_file:(j+1) * self.data_per_file])
                        h5file.create_dataset('ee_pos', data=ee_pos_np[j * self.data_per_file:(j+1) * self.data_per_file])
                        h5file.create_dataset('ee_ori', data=ee_ori_np[j * self.data_per_file:(j+1) * self.data_per_file])
                        h5file.create_dataset('ee_pos_vel', data=ee_vel_np[j * self.data_per_file:(j + 1) * self.data_per_file])
                        h5file.create_dataset('ee_vel_ori',data=ee_vel_ori_np[j * self.data_per_file:(j + 1) * self.data_per_file])
                        h5file.create_dataset('ee_yaw', data=ee_yaw_np[j * self.data_per_file:(j + 1) * self.data_per_file])
                        h5file.create_dataset('ee_yaw_delta', data=ee_yaw_delta_np[j * self.data_per_file:(j + 1) * self.data_per_file])

                        # h5file.create_dataset('dataset2', data=data2, dtype='float64')

        print("Dataset successfully created.")

    def calc_yaw_data(self, ee_ori):
        dt = 1 / self.collection_freq

        # receive yaw and convert back to quaternions
        rotation = R.from_quat(ee_ori)
        euler_angles = rotation.as_euler('xyz', degrees=False)

        euler_yaw = np.zeros((len(euler_angles), 3))
        euler_yaw[:, 2] = euler_angles[:, 2]
        euler_yaw_diff = np.diff(euler_yaw, axis=0)
        euler_yaw_vel = euler_yaw_diff / dt

        rotation = R.from_euler('xyz', euler_yaw, degrees=False)
        yaw_quat = rotation.as_quat()  # Output in [x, y, z, w] format

        # calculate the yaw difference and its quaternions
        euler_yaw_diff = np.diff(euler_yaw, axis=0)
        rotation = R.from_euler('xyz', euler_yaw_diff, degrees=False)
        quat_yaw_diff = rotation.as_quat()

        # change output shape from (size,) to (size,1)
        euler_yaw = euler_yaw[:, 2].reshape(-1, 1)
        euler_yaw_diff = euler_yaw_diff[:,2].reshape(-1, 1)
        euler_yaw_vel = euler_yaw_vel[: ,2].reshape(-1, 1)

        return euler_yaw, euler_yaw_vel, euler_yaw_diff, yaw_quat, quat_yaw_diff

    def calc_ee_velo(self, ee_pos, ee_ori):
        dt = 1/ self.collection_freq

        # calculate EE linear velocity
        ee_pos_diff = np.diff(ee_pos, axis=0)
        ee_pos_vel = ee_pos_diff / dt

        # calculate quaternion velocities
        # function to compute quaternion difference
        def quaternion_difference(q1, q2):
            # Quaternion multiplication q_diff = q1_inverse * q2
            q1_conj = np.array([-q1[0], -q1[1], -q1[2], q1[3]])  # Conjugate of q1, assuming q1 = [x, y, z, w]
            q_diff = np.array([
                q1_conj[3] * q2[0] + q1_conj[0] * q2[3] + q1_conj[1] * q2[2] - q1_conj[2] * q2[1],  # x component
                q1_conj[3] * q2[1] - q1_conj[0] * q2[2] + q1_conj[1] * q2[3] + q1_conj[2] * q2[0],  # y component
                q1_conj[3] * q2[2] + q1_conj[0] * q2[1] - q1_conj[1] * q2[0] + q1_conj[2] * q2[3],  # z component
                q1_conj[3] * q2[3] - q1_conj[0] * q2[0] - q1_conj[1] * q2[1] - q1_conj[2] * q2[2]  # w component
            ])
            return q_diff

        quaternion_diffs = np.array(
            [quaternion_difference(ee_ori[i], ee_ori[i + 1]) for i in range(len(ee_ori) - 1)])

        # due to normalization w can get negelected
        ee_ori_vel = quaternion_diffs[:, :3] / dt


        return ee_pos_vel, ee_ori_vel, ee_pos_diff

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

        # normalize optical flow
        for i in range(optical_flows.shape[0]):
            optical_flows[i] = self.normalize_optical_flow(optical_flows[i])

        # apply threshold
        thresh_flow = self.threshold_flow(optical_flows)
        # apply median filter over three frames
        med_flow = self.apply_median_filter(thresh_flow)

        return med_flow

        def threshold_flow(self, flow, threshold=0.15):
            # Compute the magnitude of the flow
            magnitude = np.sqrt(np.sum(flow ** 2, axis=-1, keepdims=True))
            # Create mask to fit dimensions to apply threshold
            mask = magnitude < threshold
            # Apply threshold: zero out flow vectors with small magnitudes
            flow[mask.repeat(2, axis=-1)] = 0
            return flow

    def apply_median_filter(self, flow_frames, size=3):
        filtered_flow = np.copy(flow_frames)
        for i in range(2):
            filtered_flow[..., i] = median_filter(flow_frames[..., i], size=size)
        return filtered_flow

    def normalize_optical_flow(self, flow):
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






if __name__ == "__main__":
    if os.name == 'nt':
        print('Using Windows system')
        DATASET_DIR = r'C:\Rest\Uni\14_SoSe\IRM_Prac_2\data_test\new_dataset'
        RECORDING_DIR = r'C:\Rest\Uni\14_SoSe\IRM_Prac_2\data_test'
    elif os.name == 'posix':
        print('Using Linux system')
        RECORDING_DIR = r"/home/philipp/Uni/14_SoSe/IRM_Prac_2/recordings"
        DATASET_DIR = r"/home/philipp/Uni/14_SoSe/IRM_Prac_2/dataset"
    else:
        print(f'Os name not recognized: {os.name}')
        sys.exit(0)

    create = CreateDatasetFromRecord(DATASET_DIR, RECORDING_DIR)