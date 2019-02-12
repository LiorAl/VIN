import numpy as np
import os
from time import strftime, localtime
from collections import namedtuple
from PIL import Image
import matplotlib.pylab as plt
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

def get_real_position(position, imsize, window, device):
    LEG_DOWN = 18

    VIEWPORT_W = 600
    VIEWPORT_H = 400

    SCALE = 30
    W = VIEWPORT_W / SCALE
    H = VIEWPORT_H / SCALE
    helipad_y = H / 4

    pos = position.cpu().numpy()

    trans = lambda pos:   [((pos[0] * VIEWPORT_W/SCALE/2) + VIEWPORT_W/SCALE/2) /SCALE * imsize[0],
                           pos[1] * (VIEWPORT_H / SCALE / 2) + (helipad_y + LEG_DOWN / SCALE) / SCALE * imsize[1]]

    window_width_fn = lambda center : np.arange(max(0, center - int(window/2)), min(imsize[1], center + int(window/2) + 1))
    window_hieght_fn = lambda center : np.arange(max(0, center - int(window/2)), min(imsize[0], center + int(window/2) + 1))

    position_ = np.round(np.apply_along_axis(trans, 1, pos)).astype(int)
    # x_ = (x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2)
    # x_ =  (x * VIEWPORT_W/SCALE/2) + VIEWPORT_W/SCALE/2
    # y_ = y * (VIEWPORT_H/SCALE/2) + (helipad_y+LEG_DOWN/SCALE)
    # y_ = (y - (helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
    # p_x = torch.from_numpy(np.apply_along_axis(window_width_fn, 0, position_[:, 0])).to(device)
    # p_y = torch.from_numpy(np.apply_along_axis(window_hieght_fn, -1, position_[:, 1])).to(device)
    p_x = torch.from_numpy(position_[:, 0]).to(device)
    p_y = torch.from_numpy(position_[:, 1]).to(device)
    return p_x, p_y

# class LearningPlot:
#     accumulate_reward = []
#     avg_accumulate_reward = []
#     STD_accumulate_reward = []
#
#     def __init__(self, folder, game):
#         self.folder = folder
#         self.fig = plt.figure()
#
#         plt.title(game)
#         plt.xlabel('Number of Timesteps')
#         plt.ylabel('Rewards')
#
#     def update_line(self, new_reward):
#         new_data_length = len(new_reward)
#         if new_data_length < len(self.accumulate_reward):
#             recent_values = self.accumulate_reward[-new_data_length:-1]
#             self.accumulate_reward.append(np.setdiff1d(recent_values, new_reward))
#         else:
#             self.accumulate_reward.append(new_reward)
#
#
#         self.fig.clf()
#         durations_t = np.arange(len(accumulate_reward))
#         plt.title('Training - Accumulated reward and AVG reward')
#         plt.xlabel('Episode')
#         plt.ylabel('Reward')
#         plt.plot(durations_t, accumulate_reward, color='r')
#         plt.plot(durations_t, avg_accumulate_reward, color='b')
#         plt.fill_between(durations_t, np.array(avg_accumulate_reward) - np.array(STD_accumulate_reward),
#                          np.array(avg_accumulate_reward) + np.array(STD_accumulate_reward), color='b', alpha=0.2)
#         plt.tight_layout()
#         plt.legend(['Reward', 'Mean reward', 'STD'])
#
#         plt.pause(0.001)  # pause a bit so that plots are updated
#
#         def plot_durations(accumulate_reward, avg_accumulate_reward, STD_accumulate_reward):
#             fig = plt.figure(2)
#             fig.clf()
#             durations_t = np.arange(len(accumulate_reward))
#             plt.title('Training - Accumulated reward and AVG reward')
#             plt.xlabel('Episode')
#             plt.ylabel('Reward')
#             plt.plot(durations_t, accumulate_reward, color='r')
#             plt.plot(durations_t, avg_accumulate_reward, color='b')
#             plt.fill_between(durations_t, np.array(avg_accumulate_reward) - np.array(STD_accumulate_reward),
#                              np.array(avg_accumulate_reward) + np.array(STD_accumulate_reward), color='b', alpha=0.2)
#             plt.tight_layout()
#             plt.legend(['Reward', 'Mean reward', 'STD'])
#
#     def save_fig(self):
#         self.fig.savefig(self.folder + 'Train_proc.jpg')


def prepare_model_dir(work_dir):
    # Create results directory
    result_path = os.getcwd() + work_dir + '/' + strftime('%b_%d_%H_%M_%S', localtime())
    os.mkdir(result_path)

    return result_path



def fmt_row(width, row):
    out = " | ".join(fmt_item(x, width) for x in row)
    return out


def fmt_item(x, l):
    if isinstance(x, np.ndarray):
        assert x.ndim == 0
        x = x.item()
    if isinstance(x, float): rep = "%g" % x
    else: rep = str(x)
    return " " * (l - len(rep)) + rep


def get_stats(loss, predictions, labels):
    cp = np.argmax(predictions.cpu().data.numpy(), 1)
    error = np.mean(cp != labels.cpu().data.numpy())
    return loss.data[0], error


def print_stats(epoch, avg_loss, avg_error, num_batches, time_duration):
    print(
        fmt_row(10, [
            epoch + 1, avg_loss / num_batches, avg_error / num_batches,
            time_duration
        ]))


def print_header():
    print(fmt_row(10, ["Epoch", "Train Loss", "Train Error", "Epoch Time"]))
