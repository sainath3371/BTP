import secrets
import numpy as np
from pyldpc import make_ldpc, encode, decode, get_message
import random
import hashlib
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import pandas as pd
from PIL import Image
import os
import h5py


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
n = 50
d_v = 5
d_c = 10
snr = 20

# Generate LDPC matrices
H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
k = G.shape[1]
print("Binary Code Size :", k)
# Generate a random binary string and encode it
random_binary_string = generate_random_bit_string(k)
encoded_bits = ldpc_encode(random_binary_string, G)


Num_of_subjects_for_training = 150
Total_subjects = 200
# Create a DataFrame with class, register, and encoded output
df = pd.DataFrame({'Class': range(1, Total_subjects + 1)})
df['Register'] = [generate_random_bit_string(k) for _ in range(Total_subjects)]
Total_generated_list = list(df['Register'])
random_generated_list = Total_generated_list[:150]
random_generated_set = set(random_generated_list)
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
    #     self.mean = 0
    #     self.std = 0
    #     self.calculate_mean_std()

    # def calculate_mean_std(self):
    #     all_images = []
    #     for img_path in self.image_paths:
    #         image = Image.open(img_path)
    #         all_images.append(np.array(image))

    #     all_images = np.stack(all_images)
    #     self.mean = np.mean(all_images, axis=(0, 1, 2)) / 255
    #     self.std = np.std(all_images, axis=(0, 1, 2)) / 255

    def __len__(self):
        return len(self.augmented_images)

    def __getitem__(self, idx):
        image = self.augmented_images[idx]
        class_name = os.path.basename(os.path.dirname(self.image_paths[idx // self.num_crops]))
        output_row = self.class_register[self.class_register['Class'] == int(class_name)].iloc[0]
        output_string = output_row['Output']
        output = [int(bit) for bit in output_string]
        random_binary_string = output_row['Register']

        # Apply preprocessing transform
        image = self.transform(image)

        return image.to(device), torch.tensor(output, dtype=torch.float32).to(device), random_binary_string

    def get_image_paths(self):
        image_paths = []
        # print(os.listdir(self.root_dir))
        for class_name in os.listdir(self.root_dir):
            class_folder = os.path.join(self.root_dir, class_name)
            image_files = [os.path.join(class_folder, f) for f in os.listdir(class_folder) if f.endswith('.jpg')]
            image_paths.extend(image_files)
        return image_paths

    def augment_dataset(self):
        augmented_images = []
        for img_path in self.image_paths:
            image = Image.open(img_path)

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
input_size = 224
output_size = 224
batch_size = 64
base_dir = "/content/Folder_Wise_Detected_Trimmed_FEI_Research_Dataset_224_224"

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Define preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Create an instance of CustomDataset
dataset = CustomDataset(csv_file='Class_Register.csv',
                         root_dir=base_dir,
                         crop_size=222,
                         num_crops=9,
                         transform=preprocess)

# Accessing the dataset length
print("Dataset length:", len(dataset))

#.......................................................DataLoader.................................................

images_per_person = 9
crop_size = 222
size = (224 - crop_size + 1) * (224 - crop_size + 1)
# Function to get train and test indices for the dataset
def get_indices(dataset, crop_size, Num_of_subjects_for_training):
    train_indices = []
    test_indices = []
    k = 0
    size = (224 - crop_size + 1) * (224 - crop_size + 1)
    for i in range(len(dataset) // (size*images_per_person)):
        train_group_indices = random.sample(range(images_per_person), 4)
        if i < Num_of_subjects_for_training :
            for j in range(images_per_person):
                k = i*images_per_person + j*9
                if j in train_group_indices:  # First 4 images for training, remaining for testing
                    for _ in range(size):
                        train_indices.append(k)
                        k += 1
                else:
                    for _ in range(size):
                        test_indices.append(k)
                        k += 1
        else:
            for j in range(images_per_person):
                k = i*images_per_person + j*9
                for _ in range(size):
                    test_indices.append(k)
                    k += 1
    return train_indices, test_indices

# Get train and test indices
train_indices, test_indices = get_indices(dataset, crop_size , Num_of_subjects_for_training)
print("length of train_indices:", len(train_indices))
print("length of test_indices:", len(test_indices))
train_sampler = SubsetRandomSampler(train_indices)
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
print("length of train loader:" , len(train_loader))
#.................................................model...............................................

# Load the VGG16 model without pretrained weights
vgg16 = models.vgg16(pretrained=False)

# Path to the HDF5 weights file
weights_file = "/content/rcmalli_vggface_tf_vgg16.h5"

# Helper function to convert h5py dataset to torch tensor
def to_tensor(h5_dataset):
    return torch.tensor(np.array(h5_dataset))

# Open the HDF5 file and load the weights
with h5py.File(weights_file, 'r') as f:
    # Load the convolutional layers
    vgg16.features[0].weight.data = to_tensor(f['conv1_1']['conv1_1']['kernel:0']).permute(3, 2, 0, 1)
    vgg16.features[0].bias.data = to_tensor(f['conv1_1']['conv1_1']['bias:0'])
    vgg16.features[2].weight.data = to_tensor(f['conv1_2']['conv1_2']['kernel:0']).permute(3, 2, 0, 1)
    vgg16.features[2].bias.data = to_tensor(f['conv1_2']['conv1_2']['bias:0'])

    vgg16.features[5].weight.data = to_tensor(f['conv2_1']['conv2_1']['kernel:0']).permute(3, 2, 0, 1)
    vgg16.features[5].bias.data = to_tensor(f['conv2_1']['conv2_1']['bias:0'])
    vgg16.features[7].weight.data = to_tensor(f['conv2_2']['conv2_2']['kernel:0']).permute(3, 2, 0, 1)
    vgg16.features[7].bias.data = to_tensor(f['conv2_2']['conv2_2']['bias:0'])

    vgg16.features[10].weight.data = to_tensor(f['conv3_1']['conv3_1']['kernel:0']).permute(3, 2, 0, 1)
    vgg16.features[10].bias.data = to_tensor(f['conv3_1']['conv3_1']['bias:0'])
    vgg16.features[12].weight.data = to_tensor(f['conv3_2']['conv3_2']['kernel:0']).permute(3, 2, 0, 1)
    vgg16.features[12].bias.data = to_tensor(f['conv3_2']['conv3_2']['bias:0'])
    vgg16.features[14].weight.data = to_tensor(f['conv3_3']['conv3_3']['kernel:0']).permute(3, 2, 0, 1)
    vgg16.features[14].bias.data = to_tensor(f['conv3_3']['conv3_3']['bias:0'])

    vgg16.features[17].weight.data = to_tensor(f['conv4_1']['conv4_1']['kernel:0']).permute(3, 2, 0, 1)
    vgg16.features[17].bias.data = to_tensor(f['conv4_1']['conv4_1']['bias:0'])
    vgg16.features[19].weight.data = to_tensor(f['conv4_2']['conv4_2']['kernel:0']).permute(3, 2, 0, 1)
    vgg16.features[19].bias.data = to_tensor(f['conv4_2']['conv4_2']['bias:0'])
    vgg16.features[21].weight.data = to_tensor(f['conv4_3']['conv4_3']['kernel:0']).permute(3, 2, 0, 1)
    vgg16.features[21].bias.data = to_tensor(f['conv4_3']['conv4_3']['bias:0'])

    vgg16.features[24].weight.data = to_tensor(f['conv5_1']['conv5_1']['kernel:0']).permute(3, 2, 0, 1)
    vgg16.features[24].bias.data = to_tensor(f['conv5_1']['conv5_1']['bias:0'])
    vgg16.features[26].weight.data = to_tensor(f['conv5_2']['conv5_2']['kernel:0']).permute(3, 2, 0, 1)
    vgg16.features[26].bias.data = to_tensor(f['conv5_2']['conv5_2']['bias:0'])
    vgg16.features[28].weight.data = to_tensor(f['conv5_3']['conv5_3']['kernel:0']).permute(3, 2, 0, 1)
    vgg16.features[28].bias.data = to_tensor(f['conv5_3']['conv5_3']['bias:0'])

    # Load the fully connected layers
    vgg16.classifier[0].weight.data = to_tensor(f['fc6']['fc6']['kernel:0']).t()
    vgg16.classifier[0].bias.data = to_tensor(f['fc6']['fc6']['bias:0'])
    vgg16.classifier[3].weight.data = to_tensor(f['fc7']['fc7']['kernel:0']).t()
    vgg16.classifier[3].bias.data = to_tensor(f['fc7']['fc7']['bias:0'])
    vgg16.classifier[6].weight.data = to_tensor(f['fc8']['fc8']['kernel:0']).t()
    vgg16.classifier[6].bias.data = to_tensor(f['fc8']['fc8']['bias:0'])

# Verify that the weights have been loaded correctly
print("Loaded weights successfully!")

# Modify the classifier for your specific task
num_features = vgg16.classifier[6].in_features
vgg16.classifier[6] = nn.Linear(num_features, n)

#..................................................Training............................................

# Move model to device
vgg16 = vgg16.to(device)

# Set requires_grad=True only for the final layer parameters
for param in vgg16.classifier[6].parameters():
    param.requires_grad = True

# Define the optimizer for only the parameters that require gradients
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, vgg16.parameters()), lr=0.001)

# optimizer = optim.SGD(
#     filter(lambda p: p.requires_grad, vgg16.parameters()),
#     lr=0.001,
#     momentum=0.7,
#     weight_decay=0.0005
# )

# Define learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.4)

# Define the binary cross-entropy loss
criterion = nn.BCELoss()

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    correct = (outputs == labels).float().sum()
    accuracy = correct / labels.numel()
    return accuracy

# Training loop
num_epochs = 8
for epoch in range(num_epochs):
    vgg16.train()
    total_loss = 0  # Initialize the total loss for the epoch
    num_batches = 0  # Initialize the batch counter
    for images, labels,_ in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()  # Zero the gradients
        outputs = vgg16(images)  # Forward pass
        outputs = torch.sigmoid(outputs)  # Apply sigmoid activation
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the parameters
        total_loss += loss.item()  # Accumulate the loss
        num_batches += 1  # Increment the batch counter
    average_loss = total_loss / num_batches  # Calculate the average loss for the epoch

    # Evaluate the model on the entire training set
    vgg16.eval()
    total_correct = 0
    total_samples = 0
    i=1
    with torch.no_grad():
        for images, labels, binary_string in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = vgg16(images)
            outputs = torch.sigmoid(outputs)
            predicted = (outputs > 0.5).float()
            if(i==1):
                print("Output:", predicted[0])
                # print("label:", labels)
                print("random :", binary_string[0])
                temp = ldpc_decode(predicted[0].cpu().numpy().astype(int), G, H)
                decoded_binary_string = ''.join(map(str, temp))
                print("decoded:", decoded_binary_string)
                i=0
            accuracy = calculate_accuracy(predicted, labels)
            if(i==1):
                print(accuracy)
                i=0
            total_correct += accuracy * labels.size(0)
            total_samples += labels.size(0)
        i=1

    epoch_accuracy = total_correct / total_samples

    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.6f}, Accuracy: {epoch_accuracy:.4f}')
    scheduler.step()

# ....................................predictions..............................................................
from torch.utils.data import Sampler

class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


ms_limit=0.8
test_loader = DataLoader(dataset, batch_size=9, sampler =SubsetSequentialSampler(test_indices))
print("length of test loader:",len(test_loader))
# Evaluate the model on the entire training set
predictions_ms_score_all = []

# def calculate_accuracy_veri(decoded_predictions, random_generated_list):
#     correct = 0
#     for pred in decoded_predictions:
#         # pred_hash = hashlib.sha512(pred.encode('utf-8')).hexdigest()
#         # orig_hash = hashlib.sha512(orig.encode('utf-8')).hexdigest()
#         # Compare the hashes
#         if pred in random_generated_list:
#             correct += 1
            
#     accuracy = correct / len(decoded_predictions)
#     return accuracy

def calculate_accuracy_veri(decoded_predictions, random_generated_set):
    correct = sum(1 for pred in decoded_predictions if pred in random_generated_set)
    return correct / len(decoded_predictions)

vgg16.eval()
total_correct = 0
total_samples = 0
co = 0
with torch.no_grad():
    for images, labels, binary_string in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = vgg16(images)
        outputs = torch.sigmoid(outputs)
        predicted = (outputs > 0.5).float()
        decoded_predictions = []
        for pred in predicted:
            decoded_bits = ldpc_decode(pred.cpu().numpy().astype(int), G, H, snr)
            decoded_binary_string = ''.join(map(str, decoded_bits))
            decoded_predictions.append(decoded_binary_string)

        accuracy = calculate_accuracy_veri(decoded_predictions, random_generated_set)
        predictions_ms_score_all.append(accuracy)
        if co%100==0:
            print("check")
        co+=1

print("check-1")
        
# def calculate_FRR(decoded_predictions, random_generated, ms_limit):
#     false_rejections = 0
#     for pred in decoded_predictions:
#         if pred < ms_limit:
#             false_rejections += 1
#     return false_rejections 

# def calculate_FAR(decoded_predictions, random_generated, ms_limit):
#     false_acceptances = 0
#     for pred in decoded_predictions:
#         if pred > ms_limit:
#             false_acceptances += 1
#     return false_acceptances

def calculate_FRR(decoded_predictions, ms_limit):
    return sum(1 for pred in decoded_predictions if pred < ms_limit) 

def calculate_FAR(decoded_predictions, ms_limit):
    return sum(1 for pred in decoded_predictions if pred > ms_limit)

def accuracy(predictions_ms_score_all, ms_limit):
    correct_samples = 0
    for ms in predictions_ms_score_all:
        if ms > ms_limit:
            correct_samples+=1
    return correct_samples/len(predictions_ms_score_all)


# Split the predictions for FRR and FAR calculations
predictions_first = predictions_ms_score_all
predictions_next = predictions_ms_score_all

FRR = calculate_FRR(predictions_first, ms_limit) # For first 100 subjects' 4 images
FAR = calculate_FAR(predictions_next, ms_limit) # For next 100 subjects' 9 images

print(f"False Rejection Rate (FRR): {(FRR/len(predictions_ms_score_all)) * 100:.2f}")
print(f"False Acceptance Rate (FAR): {(FAR/len(predictions_ms_score_all)) * 100:.2f}")


final_accuracy = accuracy(predictions_ms_score_all, ms_limit)
print(f'Test Accuracy: {final_accuracy * 100:.2f}%')

# # Save the entire model
# torch.save(vgg16, 'multi_label_net_model.pth')
# # Save the model state dictionary
# torch.save(vgg16.state_dict(), 'multi_label_net_state_dict.pth')


