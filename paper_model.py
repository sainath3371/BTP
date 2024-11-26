import secrets
import numpy as np
from pyldpc import make_ldpc, encode, decode, get_message
import random
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import pandas as pd
from PIL import Image
import os
import torch.optim as optim
import torch.nn.functional as F

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate a random k-bit binary string
def generate_random_bit_string(k):
    random_int = secrets.randbits(k)
    random_kbit_string = format(random_int, f'0{k}b')
    return random_kbit_string

# Function to encode a binary string using LDPC
def ldpc_encode(binary_string, G, snr=20):
    bit_array = np.array([int(bit) for bit in binary_string], dtype=np.int8)
    encoded_bits = encode(G, bit_array, snr)
    thresholded_bits = np.where(encoded_bits >= 0, 1, 0)
    return thresholded_bits

# Function to decode LDPC encoded bits
def ldpc_decode(encoded_bits, G, H, snr=20):
    decode_val = decode(H, encoded_bits, snr)
    decoded_bits = get_message(G, decode_val)
    return decoded_bits

# LDPC parameters
n = 200
d_v = 20
d_c = 40
snr = 20

# Generate LDPC matrices
H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
k = G.shape[1]

# Generate a random binary string and encode it
random_binary_string = generate_random_bit_string(k)
encoded_bits = ldpc_encode(random_binary_string, G)

# Create a DataFrame with class, register, and encoded output
df = pd.DataFrame({'Class': range(1, 201)})
df['Register'] = [generate_random_bit_string(k) for _ in range(200)]
df['Output'] = df['Register'].apply(lambda s: ''.join(map(str, ldpc_encode(s, G, snr).astype(int))))

# Save the DataFrame to a CSV file
df.to_csv('Class_Register.csv', index=False)

#.........................................preprocessing...........................................

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, crop_size, num_crops, transform):
        self.class_register = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.image_paths = self.get_image_paths()
        self.augmented_images = self.augment_dataset()

    def __len__(self):
        return len(self.augmented_images)

    def __getitem__(self, idx):
        image = self.augmented_images[idx]
        class_name = os.path.basename(os.path.dirname(self.image_paths[idx // self.num_crops]))
        output_row = self.class_register[self.class_register['Class'] == int(class_name)].iloc[0]
        output_string = output_row['Output']
        output = [int(bit) for bit in output_string]

        # Apply preprocessing transform
        image = self.transform(image)

        return image.to(device), torch.tensor(output, dtype=torch.float32).to(device)

    def get_image_paths(self):
        image_paths = []
        for class_name in os.listdir(self.root_dir):
            class_folder = os.path.join(self.root_dir, class_name)
            image_files = [os.path.join(class_folder, f) for f in os.listdir(class_folder) if f.endswith('.jpg')]
            image_paths.extend(image_files)
        return image_paths

    def augment_dataset(self):
        augmented_images = []
        for img_path in self.image_paths:
            ini_image = Image.open(img_path)
            image = ini_image.resize((256, 256))
            # Apply random crops and flips
            for i in range(self.num_crops):
                random_crop = transforms.RandomCrop(self.crop_size)
                cropped_image = random_crop(image)

                if random.random() > 0.5:
                    cropped_image = transforms.functional.hflip(cropped_image)

                # Append the augmented image to the list
                augmented_images.append(cropped_image)

        return augmented_images

# Parameters for preprocessing
batch_size = 32
base_dir = "/content/drive/MyDrive/images"

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Define preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Create an instance of CustomDataset
dataset = CustomDataset(csv_file='Class_Register.csv',
                         root_dir=base_dir,
                         crop_size=249,
                         num_crops=64,
                         transform=preprocess)

# Accessing the dataset length
print("Dataset length:", len(dataset))

#.......................................................DataLoader.................................................

# Function to get train and test indices for the dataset
def get_indices(dataset, crop_size):
    train_indices = []
    test_indices = []
    k = 0
    size = (256 - crop_size + 1) * (256 - crop_size + 1)
    for i in range(len(dataset) // size):
        if i % 9 < 4:  # First 4 images for training, remaining for testing
            for _ in range(size):
                train_indices.append(k)
                k += 1
        else:
            for _ in range(size):
                test_indices.append(k)
                k += 1
    return train_indices, test_indices

# Get train and test indices
train_indices, test_indices = get_indices(dataset, 249)
train_sampler = SubsetRandomSampler(train_indices)
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

#.................................................model...............................................


class MultiLabelNet(nn.Module):
    def __init__(self, output_size):
        self.output_size = output_size
        super(MultiLabelNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=2)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=2)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc6 = nn.Linear(256 * 6 * 6, 4096)  # Adjust input features size according to your input
        self.drop6 = nn.Dropout(0.5)
        self.fc7 = nn.Linear(4096, 4096)
        self.drop7 = nn.Dropout(0.5)
        self.fc8 = nn.Linear(4096, self.output_size)  # Assuming 200 is the number of classes for multi-label classification

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.norm1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.norm2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool5(x)

        x = x.view(-1, 256 * 6 * 6)  # Adjust shape according to your input
        x = F.relu(self.fc6(x))
        x = self.drop6(x)
        x = F.relu(self.fc7(x))
        x = self.drop7(x)
        x = self.fc8(x)

        return x

#..................................................Training............................................

# Move model to device
model = MultiLabelNet(n)
model = model.to(device)

# Define the optimizer for only the parameters that require gradients
optimizer = optim.SGD(
    model.parameters(),
    lr=0.001,
    momentum=0.7,
    weight_decay=0.0005
)

# Define learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.4)

# Define the binary cross-entropy loss
criterion = nn.BCELoss()

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    predicted = (outputs > 0.5).float()
    correct = (predicted == labels).float().sum()
    accuracy = correct / labels.numel()
    return accuracy

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0  # Initialize the total loss for the epoch
    num_batches = 0  # Initialize the batch counter
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass
        outputs = torch.sigmoid(outputs)  # Apply sigmoid activation
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the parameters
        total_loss += loss.item()  # Accumulate the loss
        num_batches += 1  # Increment the batch counter
    average_loss = total_loss / num_batches  # Calculate the average loss for the epoch
    
    # Evaluate the model on the entire training set
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            accuracy = calculate_accuracy(outputs, labels)
            total_correct += accuracy * labels.size(0)
            total_samples += labels.size(0)
    epoch_accuracy = total_correct / total_samples
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.6f}, Accuracy: {epoch_accuracy:.4f}')
    scheduler.step()

# ....................................predictions..............................................................
# Save the entire model
torch.save(model, 'multi_label_net_model.pth')
# Save the model state dictionary
torch.save(model.state_dict(), 'multi_label_net_state_dict.pth')

