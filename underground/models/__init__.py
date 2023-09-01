from underground.models.autoencoder_net import Autoencoder

def get_model(in_channels=1, downsampling_factors=[2,2,2], fmaps=32, fmul=2, kernel_size = 3):
    model = Autoencoder(in_channels, downsampling_factors, fmaps, fmul, kernel_size)
