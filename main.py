import os
import torch
from torch.optim import Adam
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from pyldpc import make_ldpc, encode, decode, get_message
from model import MultiLabelNet
from hashlib import sha3_512

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def _get_image_paths(self):
        image_paths = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
        return image_paths


def tan_triggs_preprocessing(img, alpha=0.1, tau=10.0, gamma=0.2, sigma=0.5):
    img = np.array(img).astype(np.float32)
    mean = np.mean(img)
    img -= mean
    rms_contrast = np.sqrt(np.mean(np.square(img)))
    img_normalized = alpha * (img - mean) / max(rms_contrast, tau)
    img_normalized = np.tanh(img_normalized * gamma)
    img_normalized *= sigma

    return img_normalized

def extract_crops(img, num_crops):
    crops = []
    img_tensor = transforms.ToTensor()(img)
    # _, _, h, w = img_tensor.size()
    for i in range(8):
        for j in range(8):
            crop = img_tensor[:, :, i:i+256-num_crops, j:j+256-num_crops]
            crop_resized = transforms.Resize((256, 256))(crop)
            crops.append(crop_resized)
    return crops

def generate_user_labels(num_users, sequence_length):
    hashed_labels = []
    random_gen = []
    for _ in range(num_users):
        binary_sequence = torch.randint(0, 2, size=(sequence_length,), dtype=torch.float32)
        hashed_sequence = sha3_512(binary_sequence.numpy()).digest()
        for _ in range(13):
            hashed_labels.append(hashed_sequence)
            random_gen.append(binary_sequence.numpy())
    
    return random_gen, hashed_labels 

input_size = 256
output_size = 256
batch_size = 32 

base_dir = "C:\IITD\SEM 8\miniproject\codes\project"

# Define preprocessing transforms
preprocess = transforms.Compose([
    transforms.Lambda(lambda img: tan_triggs_preprocessing(img, 7)), 
    transforms.Resize((input_size, input_size)),  
    transforms.Lambda(lambda img: extract_crops(img, 7)),  
    transforms.Lambda(lambda crops: torch.stack(crops))  
])

model = MultiLabelNet()
loss_function = nn.BCEWithLogitsLoss()  
optimizer = Adam(model.parameters(), lr=0.001)

# Dataset and DataLoader
dataset = CustomDataset(root_dir=base_dir, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# LDPC parameters are assumed to be defined as provided earlier
n = 15
d_v = 4
d_c = 5
snr = 20
H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
k = G.shape[1]
v = np.random.randint(2, size=k)

# Training loop
num_users = len(dataset.classes) 
random_sequence = 512  
random_gen_sequence , hashed_labels = generate_user_labels(num_users, random_sequence)

for epoch in range(num_epochs): 
    for images in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        # Convert outputs to binary sequences for LDPC encoding
        binary_sequences = outputs > 0.5  
        binary_sequences = binary_sequences.int().numpy()
        # LDPC Encoding
        encoded_sequences = [encode(G, seq, snr) for seq in random_gen_sequence]
        # hashed_sequences = [sha3_512(seq).digest() for seq in encoded_sequences]
        loss = loss_function(binary_sequences , encoded_sequences)  
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# torch.save(model.state_dict(), 'model.pth')
# np.save('G_matrix.npy', G)
# np.save('H_matrix.npy', H)



def match_and_vote(estimated_hashes, stored_hashes):
    num_matches = np.sum(np.isin(estimated_hashes, stored_hashes))
    ms = num_matches / len(estimated_hashes)
    return ms   

def verification(image, model, G, H, snr):
    preprocessed_image = preprocess(image)
    model_output = model(preprocessed_image)

    # Convert model output to binary sequence
    binary_sequence = (model_output > 0.5).int().numpy()

    # LDPC Decoding
    decoded_sequence = decode(H, binary_sequence, snr)

    # Hashing
    hashed_sequence = sha3_512(decoded_sequence.numpy()).digest()

    score_ms = match_and_vote(hashed_sequence, hashed_labels)

    return is_match
















    
# class CustomDataset(Dataset):
#     def __init__(self, csv_file, root_dir, crop_size, num_crops=4, transform=None):
#         self.class_register = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
#         self.crop_size = crop_size
#         self.num_crops = num_crops
#         self.image_paths = self.get_image_paths()
#         self.augmented_image_paths = self.augment_dataset()

#     def __len__(self):
#         return len(self.augmented_image_paths)

#     def __getitem__(self, idx):
#         img_name = self.augmented_image_paths[idx]
#         image = Image.open(img_name)
#         class_name = os.path.basename(os.path.dirname(img_name))  # Get class name from directory name
#         output_row = self.class_register[self.class_register['Class'] == int(class_name)].iloc[0]
#         output_string = output_row['Output']
#         output = [int(bit) for bit in output_string]

#         if self.transform:
#             image = self.transform(image)

#         return image.to(device), torch.tensor(output, dtype=torch.float32).to(device)

#     def get_image_paths(self):
#         image_paths = []
#         for class_name in os.listdir(self.root_dir):
#             class_folder = os.path.join(self.root_dir, class_name)
#             image_files = [os.path.join(class_folder, f) for f in os.listdir(class_folder) if f.endswith('.jpg')]
#             image_paths.extend(image_files)
#         return image_paths

#     def augment_dataset(self):
#         augmented_image_paths = []
#         for img_path in self.image_paths:
#             image = Image.open(img_path)

#             # Apply random crops and flips
#             for _ in range(self.num_crops):
#                 random_crop = transforms.RandomCrop(self.crop_size)
#                 cropped_image = random_crop(image)

#                 if random.random() > 0.5:
#                     cropped_image = transforms.functional.hflip(cropped_image)

#                 # Save the augmented image
#                 augmented_image_path = os.path.splitext(img_path)[0] + f'_augmented_{len(augmented_image_paths)}.jpg'
#                 cropped_image.save(augmented_image_path, format='JPEG')
#                 augmented_image_paths.append(augmented_image_path)

#         return augmented_image_paths
