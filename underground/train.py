import os
import argparse

assert torch.cuda.is_available()
device = torch.device("cuda")

from underground.models import get_model
from underground.datasets import get_dataset


def train(args):
    print(args)
    
    # create train dataset
    dataset = get_dataset(args.dataset, args.batch_size)

    # create val dataset


    # create model
    model = get_model(in_channels=1, downsampling_factors=[2,2,2],
        fmaps=32, fmul=2, kernel_size = 3)

    # create loss object
    loss_function = torch.loss.MSELoss()

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters())

    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")


    for epoch in range(args.n_epochs):
        train_epoch(dataset, model, epoch, optimizer, loss_function)

def train_epoch(dataset,model,epoch,optimizer,loss_function):
    model.train()
    model = model.to(device)
    for batch_id, x in enumerate(dataset):
        x = x.to(device)
        
        optimizer.zero_grad()

        # apply model and calculate loss
        prediction = model(x)
        loss = torch.MSELoss(prediction, x)

        # backpropagate the loss and adjust the parameters
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment Config')
    parser.add_argument('--dataset', type = int)
    parser.add_argument('--crop_size', type = int)
    parser.add_argument('--batch_size', type = int)
    parser.add_argument('--n_epochs', type = int)
    args = parser.parse_args()
    print(args.crop_size)
