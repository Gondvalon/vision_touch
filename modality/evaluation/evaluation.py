import os
import tensorboard
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

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
    plt.savefig('../../../plots/loss_plot_triangle_data.png')  # Uncomment to save as PNG
    plt.show()


if __name__ == '__main__':
    log_dirs = [r'C:\Rest\Uni\14_SoSe\IRM_Prac_2\vision_touch\modality\scripts\logging',
               r'/mnt/c/Rest/Uni/14_SoSe/IRM_Prac_2/vision_touch/modality/scripts/logging']
    seeds = ['202409230023_']
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
        plot_losses(log_dir)
    else:
        print(f'Os name not recognized: {os.name}')