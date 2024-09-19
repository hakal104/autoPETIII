import json
import os
import subprocess

import SimpleITK
import torch

from classify_pet import classify_pet
import nibabel as nib
import numpy as np

class Autopet_model:
    """
    Class to handle the prediction pipeline for lesion segmentation in PET/CT. 
    This includes loading inputs, running a tracer classifier, performing segmentation with tracer-specific
    nnU-Net-ensemble, postprocessing, and saving the outputs.
    """

    def __init__(self):
        """
        Initialize paths and parameters. These include paths to the input/output folders and 
        checkpoints for the nnUNet and tracer classifier models.
        """
        # Set paths based on grand-challenge interfaces
        self.input_path = "/input/"  # Input images directory
        self.output_path = "/output/images/automated-petct-lesion-segmentation/"  # Output segmentation directory
        self.nii_path = "/opt/algorithm/imagesTs"  # Path for .nii converted files
        self.pred_path = "/opt/algorithm/prediction"  # Path for model predictions
        self.result_path = "/opt/algorithm/result"  # Final result path for postprocessed output
        self.nii_seg_file = "TCIA_001.nii.gz"  # Segmentation output filename
        self.output_path_category = "/output/data-centric-model.json"  # Path for storing classification result
        self.ckpt_path = "/opt/algorithm/tracer_classifier.pt"  # Path for tracer classifier model
        pass

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):
        """
        Convert an .mha file to .nii format.

        Args:
            mha_input_path (str): Path to the .mha input file.
            nii_out_path (str): Path to save the converted .nii file.
        """
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):
        """
        Convert a .nii file to .mha format.

        Args:
            nii_input_path (str): Path to the .nii input file.
            mha_out_path (str): Path to save the converted .mha file.
        """
        img = SimpleITK.ReadImage(nii_input_path)
        SimpleITK.WriteImage(img, mha_out_path, True)

    def check_gpu(self):
        """
        Checks if GPU is available.
        """
        print("Checking GPU availability")
        is_available = torch.cuda.is_available()
        print(f"Available: {is_available}")
        print(f"Device count: {torch.cuda.device_count()}")
        if is_available:
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
            print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory}")

    def load_inputs(self):
        """
        Load input images (CT and PET) from the specified directory, convert them from .mha to .nii.gz format, 
        and return the unique ID associated with the input images.

        Returns:
            uuid (str): Unique ID of the input case (based on CT file name).
        """
        ct_mha = os.listdir(os.path.join(self.input_path, "images/ct/"))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, "images/pet/"))[0]
        uuid = os.path.splitext(ct_mha)[0]

        # Convert CT and PET images to .nii.gz format
        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/ct/", ct_mha),
            os.path.join(self.nii_path, "TCIA_001_0000.nii.gz"),
        )
        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/pet/", pet_mha),
            os.path.join(self.nii_path, "TCIA_001_0001.nii.gz"),
        )
        return uuid

    def write_outputs(self, uuid):
        """
        Convert the prediction from .nii.gz back to .mha format and write it to an output file.

        Args:
            uuid (str): Unique ID of the input case to name the output file.
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.convert_nii_to_mha(
            os.path.join(self.result_path, self.nii_seg_file),
            os.path.join(self.output_path, uuid + ".mha"),
        )
        print(f"Output written to: {os.path.join(self.output_path, uuid + '.mha')}")

        
    def predict(self):
        """
        Your algorithm goes here
        """
        print("Tracer classification starting now!")
        
        pet_path = os.path.join(self.nii_path, 'TCIA_001_0001.nii.gz')
        tracer = classify_pet(pet_path, self.ckpt_path)
        dataset_nr = 1 if tracer == 'fdg' else 2

        # Run prediction
        
        print("nnUNet segmentation starting!")
        
        if dataset_nr == 1:
            cproc = subprocess.run(
                f"nnUNetv2_predict -i {self.nii_path} -o {self.pred_path} -d {dataset_nr} -c 3d_fullres -tr nnUNetTrainer_1500epochs_fasterdec -p nnUNetResEncUNetMPlans",
                shell=True,
                check=True,
            )
        else:
            cproc = subprocess.run(
                f"nnUNetv2_predict -i {self.nii_path} -o {self.pred_path} -d {dataset_nr} -c 3d_fullres",
                shell=True,
                check=True,
            )        
        print(cproc)
        
        print("Prediction finished")

        return tracer

    def save_datacentric(self, value: bool):
        """
        Saves whether the model is data-centric into a json.

        Args:
            value (bool): Whether the model is data-centric.
        """
        print(f"Saving datacentric json to {self.output_path_category}")
        with open(self.output_path_category, "w") as json_file:
            json.dump(value, json_file, indent=4)

    def postprocess(self, tracer):
        """
        Postprocess the segmentation based on tracer-specific SUV thresholds.

        Args:
            tracer (str): The type of tracer ("fdg" or "psma").
        """
        seg = nib.load(os.path.join(self.pred_path, 'TCIA_001.nii.gz'))
        seg_arr = seg.get_fdata()

        pet = nib.load(os.path.join(self.nii_path, 'TCIA_001_0001.nii.gz')).get_fdata()
        suv_threshold = 1.5 if tracer == 'fdg' else 1

        # Apply SUV threshold to filter out non-significant regions
        seg_arr[pet < suv_threshold] = 0
        seg_arr = np.where(seg_arr == 1, 1, 0)

        final_seg = nib.Nifti1Image(seg_arr.astype(np.int16), seg.affine, seg.header)
        nib.save(final_seg, os.path.join(self.result_path, self.nii_seg_file))

    def process(self):
        """
        Main processing function that loads the inputs, runs the prediction, 
        postprocessing, and saves the final outputs.
        """
        self.check_gpu()
        print("Start processing")
        
        # Load inputs and start prediction
        uuid = self.load_inputs()
        print("Start prediction")
        tracer = self.predict()
        
        # Postprocess and save outputs
        print("Start postprocessing")
        self.postprocess(tracer)
        print("Start writing final output")
        self.save_datacentric(False)
        self.write_outputs(uuid)


if __name__ == "__main__":
    print("START")
    Autopet_model().process()
