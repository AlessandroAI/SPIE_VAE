
from model import EncoderModel, DecoderModel, bit_rate_and_reconstruction
import sys # for sys.stdout.flush
import torch

def train_one_epoch(epoch, beta, train_loader, encoder_model, decoder_model,
                    entropy_bottleneck, optimizer, batches_between_logging=75,device = 'cuda'):
    # Set all parts of the model into training mode.
    encoder_model.train()
    decoder_model.train()
    entropy_bottleneck.train()

    total_bit_rate = 0
    total_distortion = 0
    total_num_pixels = 0
    for batch_idx, batch in enumerate(train_loader):
        batch = batch['image'].to(device)
        # Select the first three channels of the image
        imager = batch[ 3, :]
        imageb = batch[ 2, :]
        imageg = batch[ 1, :]
        # Stack the three channels together 
        batch = torch.stack([imager, imageg, imageb], dim=0)

        optimizer.zero_grad()
        bit_rate, distortion, _, num_z = bit_rate_and_reconstruction(
            encoder_model, decoder_model, entropy_bottleneck, batch)
        loss = beta * bit_rate + distortion
        loss.backward()
        optimizer.step()

        total_bit_rate += bit_rate
        total_distortion += distortion
        total_num_pixels += batch.numel()

        if batch_idx % batches_between_logging == 0:
            bpp = bit_rate / batch.numel()
            mse = distortion / batch.numel() # (MSE = mean squared error)
            loss_pp = loss / batch.numel() # (MSE = mean squared error)
            entropy_pz = entropy_bottleneck.entropy_per_latent_dim()
            entropy_pp = entropy_pz * num_z / batch.numel()
            print(
                f'Training epoch {epoch}, batch {batch_idx: 3d} of {len(train_loader)}: ' +
                f'BPP = {bpp:.4f}; MSE = {mse:.4f}; loss per pixel = {loss_pp:.4f};'
            )
            print(f'  prior entropy (per pixel, per latent dim): {entropy_pp:.4f}, {entropy_pz:.4f}')
            sys.stdout.flush()

    bpp = total_bit_rate / total_num_pixels
    mse = total_distortion / total_num_pixels
    return bpp, mse
