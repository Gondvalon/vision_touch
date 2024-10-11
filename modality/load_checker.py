import h5py
import numpy as np
# import ipdb
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
from trainers.selfsupervised import selfsupervised


class EvalDemo:
    def __init__(self, model_path, data_path):
        super().__init__()
        # model
        self.model_path = model_path
        # load mdata
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_path = data_path


    def check_loader(self, demo_path):
        # import pdb
        # pdb.set_trace()

        # TODO: proper precoessing of data is missing
        # (optical flow)

        dataset = h5py.File(demo_path, "r", swmr=True, libver="latest")
        # if self.training_type == "selfsupervised":

        dataset_index = 0

        image = np.array(dataset["fixed_view_left"])
        depth = np.array(dataset["fixed_view_left_depth"])
        proprio = np.array(dataset["JointState"])
        tau = np.array(dataset["tau"])

        # todo: what is action in our case
        action = None
        # action = np.array(dataset["action_key"])


        # image = dataset["fixed_view_left"].to(self.device)
        # depth = dataset["fixed_view_left_depth"][dataset_index]
        # proprio = dataset["JointState"][dataset_index][:8]
        # tau = dataset["tau"][dataset_index]

        if image.shape[0] == 3:
            image = np.transpose(image, (2, 1, 0))

        if depth.ndim == 2:
            depth = depth.reshape((128, 128, 1))


        # todo add optical flow for label to compare @philipp
        # flow = np.array(dataset["optical_flow"][dataset_index])
        # # this mask is showing where the sum of the flow is not zero and gets marked with a 1
        # flow_mask = np.expand_dims(
        #     np.where(
        #         flow.sum(axis=2) == 0,
        #         np.zeros_like(flow.sum(axis=2)),
        #         np.ones_like(flow.sum(axis=2)),
        #     ),
        #     2,
        # )

        sample = {
            "image": image,
            "depth": depth,
            # "flow": flow,
            # "flow_mask": flow_mask,
            "action": action,
            "proprio": proprio,
            # todo add ee_next if possible
            # "ee_yaw_next": dataset["proprio"][dataset_index + 1][:self.action_dim],
            # "unpaired_image": unpaired_image,
            # "unpaired_proprio": unpaired_proprio,
            # "unpaired_depth": unpaired_depth,
        }
        # depending if tau or force sensor is used
        # if not self.tau_use:
        #     sample["force"] = force
        #     sample["contact_next"] = np.array(
        #         [dataset["contact"][dataset_index + 1].sum() > 0]).astype(np.float64)
        #     # sample["unpaired_force"] = unpaired_force
        # else:
        sample["tau"] = tau
        # sample["unpaired_tau"] = unpaired_tau
        # sample["contact_next"] = np.array([np.abs(dataset["tau_ext"][dataset_index + 1]).sum() > 7.0]).astype(
        #     np.float64)

        dataset.close()

        # no sample transform

        return sample



if __name__ == "__main__":
    print("start")

    model_path = None
    data_path = "/home/rickmer/Documents/Vision_Touch_pjahr/pearl_eval_data/0.h5"


    evaler = EvalDemo(model_path, data_path)

    evaler.check_loader(data_path)


    print("end")