import time
import argparse

import gym
import matplotlib
import matplotlib.pyplot as plt
import torch

from dataset.dataset import *
from utils import *
from model import *



# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_screen(env):
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)

    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)

class Constants:

    BATCH_SIZE = 32
    GAMMA = 0.999
    EPS_START = 1
    EPS_END = 0.05
    EPS_DECAY_MAX = 2000
    EPS_DECAY_MIN = 20
    TARGET_UPDATE = 500
    MAX_DURATION = 2000
    REPLAY_BUFFER = 100000
    LEARNING_RATE = 0.00025
    EPS_RMSPROP = 0.01
    ALPHA_RMSPROP = 0.95
    PURE_EXPLORATION_STEPS = 50000
    STOP_EXPLORATION_STEPS = 250000
    PRIOR_REG = 1e-5
    EPISODES_MEAN_REWARD = 10
    STOP_CONDITION = -100



def train(net, trainloader, config, criterion, optimizer, use_GPU):
    env = gym.make('LunarLander-v2')
    num_actions = env.action_space.n
    env.reset()
    current_screen = get_screen(env)




    print_header()
    for epoch in range(config.epochs):  # Loop over dataset multiple times
        avg_error, avg_loss, num_batches = 0.0, 0.0, 0.0
        start_time = time.time()
        for i, data in enumerate(trainloader):  # Loop over batches of data
            # Get input batch
            X, S1, S2, labels = data
            if X.size()[0] != config.batch_size:
                continue  # Drop those data, if not enough for a batch
            # Send Tensors to GPU if available
            if use_GPU:
                X = X.cuda()
                S1 = S1.cuda()
                S2 = S2.cuda()
                labels = labels.cuda()
            # Wrap to autograd.Variable
            X, S1 = Variable(X), Variable(S1)
            S2, labels = Variable(S2), Variable(labels)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs, predictions = net(X, S1, S2, config)
            # Loss
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Update params
            optimizer.step()
            # Calculate Loss and Error
            loss_batch, error_batch = get_stats(loss, predictions, labels)
            avg_loss += loss_batch
            avg_error += error_batch
            num_batches += 1
        time_duration = time.time() - start_time
        # Print epoch logs
        print_stats(epoch, avg_loss, avg_error, num_batches, time_duration)
    print('\nFinished training. \n')


def test(net, testloader, config):
    total, correct = 0.0, 0.0
    for i, data in enumerate(testloader):
        # Get inputs
        X, S1, S2, labels = data
        if X.size()[0] != config.batch_size:
            continue  # Drop those data, if not enough for a batch
        # Send Tensors to GPU if available
        if use_GPU:
            X = X.cuda()
            S1 = S1.cuda()
            S2 = S2.cuda()
            labels = labels.cuda()
        # Wrap to autograd.Variable
        X, S1, S2 = Variable(X), Variable(S1), Variable(S2)
        # Forward pass
        outputs, predictions = net(X, S1, S2, config)
        # Select actions with max scores(logits)
        _, predicted = torch.max(outputs, dim=1, keepdim=True)
        # Unwrap autograd.Variable to Tensor
        predicted = predicted.data
        # Compute test accuracy
        correct += (torch.eq(torch.squeeze(predicted), labels)).sum()
        total += labels.size()[0]
    print('Test Accuracy: {:.2f}%'.format(100 * (correct / total)))


if __name__ == '__main__':
    # Automatic swith of GPU mode if available
    use_GPU = torch.cuda.is_available()
    # Parsing training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datafile',
        type=str,
        default='dataset/gridworld_8x8.npz',
        help='Path to data file')
    parser.add_argument('--imsize', type=int, default=8, help='Size of image')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.005,
        help='Learning rate, [0.01, 0.005, 0.002, 0.001]')
    parser.add_argument(
        '--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument(
        '--k', type=int, default=10, help='Number of Value Iterations')
    parser.add_argument(
        '--l_i', type=int, default=2, help='Number of channels in input layer')
    parser.add_argument(
        '--l_h',
        type=int,
        default=150,
        help='Number of channels in first hidden layer')
    parser.add_argument(
        '--l_q',
        type=int,
        default=10,
        help='Number of channels in q layer (~actions) in VI-module')
    parser.add_argument(
        '--batch_size', type=int, default=128, help='Batch size')
    config = parser.parse_args()
    # Get path to save trained model
    save_path = "trained/vin_{0}x{0}.pth".format(config.imsize)
    # Instantiate a VIN model
    net = VIN(config)
    # Use GPU if available
    if use_GPU:
        net = net.cuda()
    # Loss
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.RMSprop(net.parameters(), lr=config.lr, eps=1e-6)
    # Dataset transformer: torchvision.transforms
    transform = None
    # Define Dataset
    trainset = GridworldData(
        config.datafile, imsize=config.imsize, train=True, transform=transform)
    testset = GridworldData(
        config.datafile,
        imsize=config.imsize,
        train=False,
        transform=transform)
    # Create Dataloader
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    # Train the model
    train(net, trainloader, config, criterion, optimizer, use_GPU)
    # Test accuracy
    test(net, testloader, config)
    # Save the trained model parameters
    torch.save(net.state_dict(), save_path)
