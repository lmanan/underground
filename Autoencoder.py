### Team: Underground
### Title: Autoencoder
### Source: https://github.com/funkelab/funlib.learn.torch/blob/master/funlib/learn/torch/models/autoencoder.py

import torch

assert torch.cuda.is_available()
device = torch.device("cuda")

class Autoencoder(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            downsampling_factors,
            fmaps,
            fmul,
            kernel_size=3):

        super(Autoencoder, self).__init__()

        out_channels = in_channels

        encoder = []

        for downsampling_factor in downsampling_factors:

            encoder.append(
                    torch.nn.Conv2d(
                        in_channels,
                        fmaps,
                        kernel_size))
            encoder.append(
                    torch.nn.ReLU(inplace=True))
            encoder.append(
                    torch.nn.Conv2d(
                        fmaps,
                        fmaps,
                        kernel_size))
            encoder.append(
                    torch.nn.ReLU(inplace=True))
            encoder.append(
                    torch.nn.MaxPool2d(downsampling_factor))

            in_channels = fmaps

            fmaps = fmaps * fmul

        fmaps_bottle = fmaps

        encoder.append(
            torch.nn.Conv2d(
                in_channels,
                fmaps_bottle,
                kernel_size))
        encoder.append(
            torch.nn.ReLU(inplace=True))

        self.encoder = torch.nn.Sequential(*encoder)

        decoder = []

        fmaps = in_channels

        decoder.append(
            torch.nn.Conv2d(
                fmaps_bottle,
                fmaps,
                kernel_size))
        decoder.append(
            torch.nn.ReLU(inplace=True))

        for downsampling_factor in downsampling_factors[::-1]:

            fmaps = in_channels / fmul

            decoder.append(
                torch.nn.Upsample(
                    scale_factor=downsampling_factor,
                    mode='trilinear'))
            decoder.append(
                torch.nn.Conv2d(
                    in_channels,
                    fmaps,
                    kernel_size))
            decoder.append(
                torch.nn.ReLU(inplace=True))
            decoder.append(
                torch.nn.Conv2d(
                    fmaps,
                    fmaps,
                    kernel_size))
            decoder.append(
                torch.nn.ReLU(inplace=True))

            in_channels = fmaps

        decoder.append(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size))

        self.decoder = torch.nn.Sequential(*decoder)

    def forward(self, x):

        enc = self.encoder(x)

        dec = self.decoder(enc)

        return dec

def train(
    model,
    loader,
    optimizer,
    loss_function,
    epoch,
    log_interval=100,
    log_image_interval=20,
    tb_logger=None
):
    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set the model to train mode
    model.train()

    # move model to device
    model = model.to(device)

    # iterate over the batches of this epoch
    for batch_id, (x, y) in enumerate(loader):
        # move input and target to the active device (either cpu or gpu)
        x, y = x.to(device), y.to(device)

        # zero the gradients for this iteration
        optimizer.zero_grad()

        # apply model and calculate loss
        prediction = model(x)
        loss = loss_function(prediction, y)

        # backpropagate the loss and adjust the parameters
        loss.backward()
        optimizer.step()

        # log to console
        if batch_id % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_id * len(x),
                    len(loader.dataset),
                    100.0 * batch_id / len(loader),
                    loss.item(),
                )
            )

        # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            tb_logger.add_scalar(
                tag="train_loss", scalar_value=loss.item(), global_step=step
            )
            # check if we log images in this iteration
            if step % log_image_interval == 0:
                tb_logger.add_images(
                    tag="input", img_tensor=x.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="target", img_tensor=y.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="prediction",
                    img_tensor=prediction.to("cpu").detach(),
                    global_step=step,
                )


############################################################################################

train_data = NucleiDataset(TRAIN_DATA_PATH, transforms.RandomCrop(256))
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

val_data = NucleiDataset(VAL_DATA_PATH, transforms.RandomCrop(256))
val_loader = DataLoader(val_data, batch_size=32)




model = Autoencoder(in_channels=1, downsampling_factors=[2,2,2],
					fmaps=32, fmul=2, kernel_size = 3)
optimizer = torch.optim.Adam(model.parameters())
loss_function = torch.loss.MSELoss()

n_epochs = 25
for epoch_count in range(n_epochs):
    # train
    train(
        model,
        train_loader,
        optimizer=optimizer,
        loss_function=loss_function,
        epoch=epoch_count,
        log_interval=5,
        tb_logger=logger,
    )
    step = epoch * num_train_pairs
    # validate
    validate(model, val_loader, loss_function, metric, step=step, tb_logger=logger)

logger = SummaryWriter('runs/Autoencoder')
%tensorboard --logdir runs
