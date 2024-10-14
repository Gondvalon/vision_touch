import os
import tensorboard
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import h5py
import numpy as np
import cv2

def plot_losses(dir):
    # Load the TensorBoard event logs
    event_acc = EventAccumulator(dir)
    event_acc.Reload()

    # print available tags
    print(event_acc.Tags())

    # retrieve scalar metrics from TensorBoard
    contact_loss = event_acc.Scalars('loss/contact')
    ee_delta_loss = event_acc.Scalars('loss/ee_delta')
    paired_loss = event_acc.Scalars('loss/is_paired')
    kl_loss = event_acc.Scalars('loss/kl')
    flow_loss = event_acc.Scalars('loss/optical_flow')
    total_loss = event_acc.Scalars('loss/total_loss')

    # Extracting steps and values
    steps = [x.step for x in contact_loss]
    contact_loss_values = [x.value for x in contact_loss]
    ee_delta_loss_values = [x.value for x in ee_delta_loss]
    paired_loss_values = [x.value for x in paired_loss]
    kl_loss_values = [x.value for x in kl_loss]
    flow_loss_values = [x.value for x in flow_loss]
    total_loss_values = [x.value for x in total_loss]

    # Create a figure and a 1x2 grid of subplots
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))  # 2 row, 3 columns

    # Plot each loss in separate subplots
    axs[0, 0].plot(steps, contact_loss_values, label='Contact Loss', color='b')
    axs[0, 0].set_title('Contact Loss Over Time')
    axs[0, 0].set_xlabel('Steps')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(steps, ee_delta_loss_values, label='EE Delta Loss', color='g')
    axs[0, 1].set_title('EE Delta Loss Over Time')
    axs[0, 1].set_xlabel('Steps')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[0, 2].plot(steps, paired_loss_values, label='Paired Loss', color='r')
    axs[0, 2].set_title('Paired Loss Over Time')
    axs[0, 2].set_xlabel('Steps')
    axs[0, 2].set_ylabel('Loss')
    axs[0, 2].legend()
    axs[0, 2].grid(True)

    axs[1, 0].plot(steps, kl_loss_values, label='KL Loss', color='c')
    axs[1, 0].set_title('KL Loss Over Time')
    axs[1, 0].set_xlabel('Steps')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(steps, flow_loss_values, label='Optical Flow Loss', color='m')
    axs[1, 1].set_title('Optical Flow Loss Over Time')
    axs[1, 1].set_xlabel('Steps')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    axs[1, 2].plot(steps, total_loss_values, label='Total Loss', color='orange')
    axs[1, 2].set_title('Total Loss Over Time')
    axs[1, 2].set_xlabel('Steps')
    axs[1, 2].set_ylabel('Loss')
    axs[1, 2].legend()
    axs[1, 2].grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save as PNG or show the plot
    plt.savefig('../../../plots/loss_plot_own_data.png')  # Uncomment to save as PNG
    plt.show()

def plot_predictions(file):
    with h5py.File(file, "r") as f:
        contact_in = np.array(f["contact_label"][:])
        contact_pred = np.array(f["contact_pred"][:])
        tau = np.array(f["tau"][:])
        tau_ext = np.array(f["tau_ext"][:])

        contact_in = contact_in.flatten()
        contact_pred = contact_pred.flatten()

        tau_sum = np.sum(np.abs(tau), axis=1)
        tau_exts = np.sum(np.abs(tau_ext), axis=1)

        plt.figure(figsize=(8, 6))

        # Plot the first array
        plt.plot(contact_in, label="Contact Label", color='blue')

        # Plot the second array
        plt.plot(contact_pred, label="Contact Prediction", color='red')

        plt.plot(tau_exts, label="Sum of external tau values", color='green')

        # Add title and labels
        plt.title("Plot contact labels and predictions of trained network")
        plt.xlabel("Steps")
        plt.ylabel("Contact values")

        # Add a legend to differentiate the arrays
        plt.legend()

        plt.savefig('../../../plots/tau_prediction_plot.png')
        # Show the plot
        # plt.show()

        flow = np.array(f["flow"][:])
        flow_pred = np.array(f["flow_pred"][:])
        flow_pred = np.transpose(flow_pred, (0, 2, 3, 1))

        save_video(flow[150:], "flow")
        save_video(flow_pred[150:], "flow_pred")



def save_video(vid, filename):
    images = vid

    steps, height, width = images.shape[:3]  # channels should be 1 for single-channel images
    # assert channels == 1, "Depth images should have only one channel."

    # print(images[1000, 340, :])

    size = (128, 128)
    fps = 30
    if os.name == 'posix':
        save_dir = r"/home/philipp/Uni/14_SoSe/IRM_Prac_2/flagsNdepth"
    elif os.name == 'nt':
        save_dir = r"C:\Rest\Uni\14_SoSe\IRM_Prac_2\plots\vid"
    out = cv2.VideoWriter(f'{save_dir}_{filename}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size, isColor=True)

    for flow in images:  # flow is of shape (height, width, 2)
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
        res_rgb = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_LINEAR)

        out.write(res_rgb)

    out.release()



if __name__ == '__main__':
    log_dirs = [r'C:\Rest\Uni\14_SoSe\IRM_Prac_2\vision_touch\modality\scripts\logging',
               r'/mnt/c/Rest/Uni/14_SoSe/IRM_Prac_2/vision_touch/modality/scripts/logging',
                r'C:\Rest\Uni\14_SoSe\IRM_Prac_2\vision_touch\modality\scripts\eval_data.h5']
    seeds = ['202410092240_']
    if os.name == 'posix':
        print('Using Linux')
        log_dir = os.path.join(log_dirs[0], seeds[0],'runs', seeds[0])
        if os.path.exists(log_dir):
            print(f"Using log directory: {log_dir}")
        plot_losses(log_dir)
    elif os.name == 'nt':
        print('Using Windows')
        log_dir = os.path.join(log_dirs[0], seeds[0],'runs', seeds[0])
        if os.path.exists(log_dir):
            print(f"Using log directory: {log_dir}")
        print(log_dir)
        # plot_losses(log_dir)
        plot_predictions(log_dirs[2])
    else:
        print(f'Os name not recognized: {os.name}')