import numpy as np
import nibabel as nib
import torch
from torchvision.models import resnet18
from torch import nn

class MIP_transform:
    """
    Class for transforming the input MIP.
    """
    def __init__(self, resize=None, transform=None):
        """
        Initialize the transform.
        
        Args:
            resize (tuple): Target size for resizing.
            transform (callable): Optional transform function.
        """
        self.resize = resize
        self.transform = transform

    def __call__(self, sample):
        """
        Apply transformation to the sample, including resizing and converting to tensor.
        
        Args:
            sample (numpy.ndarray): Input numpy array to be transformed.
        
        Returns:
            torch.Tensor: Transformed tensor.
        """
        # Convert numpy array to torch tensor
        tensor_sample = torch.from_numpy(sample).unsqueeze(0).float()  # Add channel dimension
        tensor_sample = torch.nn.functional.interpolate(tensor_sample.unsqueeze(0), size=self.resize, mode='bilinear', align_corners=False).squeeze(0)
        if transform != None:
            tensor_sample = transform(tensor_sample)
        
        return tensor_sample    
    

def reorient_image(image_path):
    """
    Reorient the image to ensure the correct orientation using NIfTI format.
    
    Args:
        image_path (str): Path to the NIfTI image file.
        
    Returns:
        numpy.ndarray: Reoriented image array.
    """
    pet = nib.load(image_path)
    pet_arr = pet.get_fdata()
    code = nib.aff2axcodes(pet.affine)
    orientation = ['L', 'A', 'S']
    
    for (i, o) in enumerate(code):
        if code[i] != orientation[i]:
            print(f"Flipping axis {i} from {code[i]} to {orientation[i]}")
            pet_arr = np.flip(pet_arr, i)
    
    return pet_arr

def get_mips(image):
    """
    Generate maximum intensity projections (MIPs) in coronal and sagittal planes.
    
    Args:
        image (numpy.ndarray): Input image array.
        
    Returns:
        tuple: Coronal and sagittal MIPs as numpy arrays.
    """
    image = np.transpose(image, axes=(2, 1, 0))
    mip_cor = np.max(image[::-1, :], axis=1)
    mip_sag = np.max(image[::-1, :], axis=2)
    return mip_cor, mip_sag

class MipClassifier(nn.Module):
    """
    Classifier that uses ResNet18 models for coronal and sagittal MIPs, with fusion layers.
    """
    def __init__(self):
        """
        Initialize the MipClassifier with ResNet18 models for coronal and sagittal views.
        """
        super(MipClassifier, self).__init__()
        
        # Coronal model (ResNet18)
        self.model_cor = resnet18(pretrained=False)
        self.model_cor.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model_cor.fc = nn.Identity()
        
        # Sagittal model (ResNet18)
        self.model_sag = resnet18(pretrained=False)
        self.model_sag.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model_sag.fc = nn.Identity()

        # Linear layers for fusion
        self.linear = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1)
        )
    
    def forward(self, coronal_input, sagittal_input):
        """
        Forward pass through the network.
        
        Args:
            coronal_input (torch.Tensor): Coronal MIP input.
            sagittal_input (torch.Tensor): Sagittal MIP input.
        
        Returns:
            torch.Tensor: Output logits.
        """
        with torch.no_grad():
            coronal_features = self.model_cor(coronal_input)
            sagittal_features = self.model_sag(sagittal_input)

            combined_features = torch.cat((coronal_features, sagittal_features), dim=1)
            output = self.linear(combined_features)
        
        return output

def run_classification(mip_cor, mip_sag, ckpt_path):
    """
    Run the classification process using the provided MIPs and checkpoint path.
    
    Args:
        mip_cor (numpy.ndarray): Coronal MIP.
        mip_sag (numpy.ndarray): Sagittal MIP.
        ckpt_path (str): Path to the model checkpoint file.
        
    Returns:
        str: Predicted tracer type ('fdg' or 'psma').
    """
    model = MipClassifier()
    model.load_state_dict(torch.load(ckpt_path))
    model = model.to('cuda')
    model.eval()
    
    mip_cor = MIP_transform(resize=(224, 224))(mip_cor).to('cuda')
    mip_cor = mip_cor.unsqueeze(0)
    mip_sag = MIP_transform(resize=(224, 224))(mip_sag).to('cuda')
    mip_sag = mip_sag.unsqueeze(0) 
    logit = model(mip_cor, mip_sag)
    pred = (torch.sigmoid(logit) > 0.5).float()
    pred_tracer = 'fdg' if pred == 0 else 'psma'
    
    return pred_tracer

def classify_pet(image_path, ckpt_path):
    """
    Classify the PET image using the provided image and model checkpoint.
    
    Args:
        image_path (str): Path to the PET image file.
        ckpt_path (str): Path to the model checkpoint file.
        
    Returns:
        str: Predicted tracer type ('fdg' or 'psma').
    """
    new_image = reorient_image(image_path)
    mip_cor, mip_sag = get_mips(new_image)
    tracer = run_classification(mip_cor, mip_sag, ckpt_path)
    
    return tracer
