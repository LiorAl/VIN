import time
import argparse

import gym
import matplotlib
import matplotlib.pyplot as plt
import torch
from xvfbwrapper import Xvfb

from dataset.dataset import *
from utils import *
from LunarLanderModel import *



# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_on_server = True



class Configs:
    BATCH_SIZE = 32                 # Batch size
    LEARNING_RATE = 0.005           # Learning rate
    EPOCHS = 30                     # Number of epochs to train

    GAMMA = 0.999
    EPS_START = 1
    EPS_END = 0.05
    EPS_DECAY_MAX = 2000
    EPS_DECAY_MIN = 20
    TARGET_UPDATE = 500
    MAX_DURATION = 2000
    REPLAY_BUFFER = 100000
    EPS_RMSPROP = 1e-6
    ALPHA_RMSPROP = 0.95
    PURE_EXPLORATION_STEPS = 50000
    STOP_EXPLORATION_STEPS = 250000
    PRIOR_REG = 1e-5
    EPISODES_MEAN_REWARD = 10
    STOP_CONDITION = -100

    K = 10                          # Number of Value Iterations
    Input_Channels = 2              # Number of channels in input layer
    First_Hidden_Channels = 150     # Number of channels in first hidden layer
    Q_Channels = 10                 # Number of channels in q layer (~actions) in VI-module




def train(net, trainloader, config, criterion, optimizer, use_GPU):
    print_header()
    num_actions = env.action_space.n
    env.reset()


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
    # Prepare results folder
    save_path = prepare_model_dir('/trained')
    # Load configs object
    config = Configs
    # Init GYM environment
    env = gym.make('LunarLander-v2')
    # Instantiate a VIN model
    net = VIN(config, env.action_space.n, device)
    # Loss
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.RMSprop(net.parameters(), lr=config.LEARNING_RATE, eps=config.EPS_RMSPROP)
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
