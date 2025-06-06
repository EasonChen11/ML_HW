import torch, torchvision
from torch import nn, optim
from torch.nn import init as init
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from torchviz import make_dot

import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from IPython import display
from IPython.display import HTML

# import os
# print(os.getcwd())
# print(os.path.exists('/content/drive/MyDrive/CelebA'))
# print(list(Path('/content/drive/MyDrive/CelebA/img_align_celeba').glob("*.jpg")))
class dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.img_paths = list(Path(root_dir).glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        img = self.transform(img)

        return img

data_dir = "./CelebA/img_align_celeba/"
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
data = dataset(root_dir=data_dir, transform=transform)

print(f"Number of data: {len(data)}")

dataloader = DataLoader(
    data,
    batch_size=128,
    shuffle=True,
    num_workers=0
)

sample_images = next(iter(dataloader))
print(sample_images.size())

vis = torchvision.transforms.ToPILImage()(make_grid(sample_images, nrow=16, padding=5, normalize=True))
fig, ax = plt.subplots(dpi=150)
ax.imshow(vis)
plt.axis("off")
plt.title("Sample Images")
plt.savefig("sample_images.png", dpi=150)
# plt.show()

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# # Generator Code
# class Generator(nn.Module):
#     def __init__(self, ngpu):
#         super(Generator, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(True),
#             # state size. ``(ngf*8) x 4 x 4``
#             nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. ``(ngf*4) x 8 x 8``
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # state size. ``(ngf*2) x 16 x 16``
#             nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             # state size. ``(ngf) x 32 x 32``
#             nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
#             nn.Tanh()
#             # state size. ``(nc) x 64 x 64``
#         )

#     def forward(self, input):
#         return self.main(input)

# class Discriminator(nn.Module):
#     def __init__(self, ngpu):
#         super(Discriminator, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             # input is ``(nc) x 64 x 64``
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. ``(ndf) x 32 x 32``
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. ``(ndf*2) x 16 x 16``
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. ``(ndf*4) x 8 x 8``
#             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. ``(ndf*8) x 4 x 4``
#             nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, input):
#         return self.main(input)

from torch.nn.utils import spectral_norm
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key   = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C//8)
        proj_key   = self.key(x).view(B, -1, H * W)                     # (B, C//8, HW)
        energy = torch.bmm(proj_query, proj_key)                       # (B, HW, HW)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(B, -1, H * W)                  # (B, C, HW)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))        # (B, C, HW)
        out = out.view(B, C, H, W)
        return self.gamma * out + x

# 改良後的 Generator
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),  # 1x1 -> 4x4
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),  # 4x4 -> 8x8
            nn.Conv2d(ngf * 16, ngf * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),  # 8x8 -> 16x16
            nn.Conv2d(ngf * 8, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),  # 16x16 -> 32x32
            nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            SelfAttention(ngf * 2),  # 加入 Self-Attention

            nn.Upsample(scale_factor=2),  # 32x32 -> 64x64
            nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.Conv2d(ngf, nc, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# 改良後的 Discriminator
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1)),  # 64x64 -> 32x32
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)),  # 32x32 -> 16x16
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)),  # 16x16 -> 8x8
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),

            # SelfAttention(ndf * 8),  # 加入 Self-Attention（可選）

            spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0))  # 4x4 -> 1x1（不加 sigmoid）
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64

netG = Generator(ngpu=1).to(device)
netD = Discriminator(ngpu=1).to(device)

netG.apply(weights_init)
netD.apply(weights_init)

print(netG)
print(netD)

import os
dot = make_dot(netG.forward(torch.randn(1, nz, 1, 1, device=device)), params=dict(netG.named_parameters()))
dot.render("netG", format="png")
os.system("rm -rf netG")
dot = make_dot(netD.forward(torch.randn(1, nc, 64, 64, device=device)), params=dict(netD.named_parameters()))
dot.render("netD", format="png")
os.system("rm -rf netD")
# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Initialize the ``BCELoss`` function
# criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

print(f"Dataset size: {len(dataloader.dataset)}")
print(f"Batch size: {dataloader.batch_size}")
print(f"Total batches per epoch: {len(dataloader)}")

num_epochs = 10

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        # D_x = output.mean().item()
        D_x = torch.sigmoid(output).mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        # D_G_z1 = output.mean().item()
        D_G_z1 = torch.sigmoid(output).mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        # D_G_z2 = output.mean().item()
        D_G_z2 = torch.sigmoid(output).mean().item()

        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(make_grid(fake, padding=2, normalize=True))

        iters += 1

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("losses.png", dpi=150)
# plt.show()

fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

# HTML(ani.to_jshtml())
# save the animation as mp4
from matplotlib.animation import FFMpegWriter
writer = FFMpegWriter(fps=1, extra_args=['-vcodec', 'libx264'])
ani.save('dcgan_animation.mp4', writer=writer)

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))[:64]
vis = torchvision.transforms.ToPILImage()(make_grid(real_batch, nrow=8, padding=2, normalize=True))

# Plot the real images
plt.figure(figsize=(15,15))
# plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
# plt.savefig("real_images.png", dpi=150)
plt.imshow(vis)
plt.savefig("real_images.png")

# Plot the fake images from the last epoch
# plt.subplot(1,2,2)
plt.figure(figsize=(15,15))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.savefig("fake_images.png")
# plt.show()