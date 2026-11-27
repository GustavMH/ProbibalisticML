
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Arguments
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--test-batch-size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--no-accel', action='store_true')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--log-interval', type=int, default=100)
args = parser.parse_args()

use_accel = not args.no_accel and torch.accelerator.is_available()

torch.manual_seed(args.seed)

# Select device
if use_accel:
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

#kwargs = {'num_workers': 1, 'pin_memory': True} if use_accel else {}
kwargs = {'pin_memory': True} if device.type == 'cuda' else {}

# Load binarized MNIST
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       lambda x: x > 0,
                       lambda x: x.float(),
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       lambda x: x > 0,
                       lambda x: x.float(),
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class VAEDeep(nn.Module):
    def __init__(self):
        super(VAEDeep, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

        self.fc_mu = nn.Linear(128, 2)
        self.fc_logvar = nn.Linear(128, 2)

        # Decoder
        self.fc4 = nn.Linear(2, 128)
        self.fc5 = nn.Linear(128, 256)
        self.fc6 = nn.Linear(256, 512)
        self.fc7 = nn.Linear(512, 784)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))

        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        h = F.relu(self.fc6(h))

        return torch.sigmoid(self.fc7(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Variational Autoencoder
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        #self.fc1 = nn.Linear(784, 400)
        #self.fc21 = nn.Linear(400, 2)
        #self.fc22 = nn.Linear(400, 2)
        #self.fc3 = nn.Linear(2, 400)
        #self.fc4 = nn.Linear(400, 784)

        self.fc1 = nn.Linear(784, 1024)
        self.fc21 = nn.Linear(1024, 2)
        self.fc22 = nn.Linear(1024, 2)
        self.fc3 = nn.Linear(2, 1024)
        self.fc4 = nn.Linear(1024, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Initialize model and optimizer
model = VAEDeep().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence loss
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(
        recon_x, x.view(-1, 784), reduction='sum'
    )

    KLD = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return BCE + KLD


# Training
def train(epoch):
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


# Evaluation
def test():
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)

            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(
                recon_batch, data, mu, logvar
            ).item()

    test_loss /= len(test_loader.dataset)

    print('====> Test set loss: {:.4f}'.format(test_loss))


# Plot encoded samples in the latent space
def plot_latent_space():
    model.eval()

    latent_samples = []
    digit_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)

            mu, logvar = model.encode(data.view(-1, 784))

            latent_samples.append(mu.cpu())
            digit_labels.append(labels)

    latent_samples = torch.cat(latent_samples).numpy()
    digit_labels = torch.cat(digit_labels).numpy()

    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        latent_samples[:, 0],
        latent_samples[:, 1],
        c=digit_labels,
        cmap=plt.get_cmap("tab10", 10),
        vmin=-0.5,
        vmax=9.5,
        s=6,
        alpha=0.7
    )

    plt.colorbar(scatter, ticks=range(10), label="Digit class")
    plt.xlabel("$z_1$")
    plt.ylabel("$z_2$")
    plt.title("Two-dimensional latent space")
    plt.tight_layout()
    plt.savefig("latent_space.png", dpi=300)
    plt.close()


def plot_latent_grid(k=15):
    model.eval()

    normal = torch.distributions.Normal(0.0, 1.0)
    u = torch.linspace(0.01, 0.99, k) # Equally spaced 
    z_values = normal.icdf(u) # Map through the inverse Gaussian CDF

    # Construct latent vectors
    z1, z2 = torch.meshgrid(z_values, z_values.flip(0), indexing="xy")
    z = torch.stack([z1.flatten(), z2.flatten()], dim=1).to(device)

    # Decoding the latent vectors to images
    with torch.no_grad():
        images = model.decode(z).cpu()

    # Combine the images into one large image
    images = images.reshape(k, k, 28, 28)
    grid = images.permute(0, 2, 1, 3).reshape(k * 28, k * 28)

    # Plot the grid
    plt.figure(figsize=(10, 8))
    plt.imshow(grid, cmap="gray", vmin=0, vmax=1)

    # Place ticks at the center of each image
    ticks = torch.arange(k) * 28 + 13.5

    plt.xticks(ticks,[f"{v:.2f}" for v in z_values],rotation=90)
    plt.yticks(ticks, [f"{v:.2f}" for v in z_values.flip(0)])

    plt.xlabel("$z_1$")
    plt.ylabel("$z_2$")
    plt.tight_layout()
    plt.savefig("latent_grid.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()

    plot_latent_space()
    plot_latent_grid()