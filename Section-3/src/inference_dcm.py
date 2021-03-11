"""
Here we do inference on a DICOM volume, constructing the volume first, and then sending it to the
clinical archive

This code will do the following:
    1. Identify the series to run HippoCrop.AI algorithm on from a folder containing multiple studies
    2. Construct a NumPy volume from a set of DICOM files
    3. Run inference on the constructed volume
    4. Create report from the inference
    5. Call a shell script to push report to the storage archive
"""

import os
import sys
import datetime
import time
import shutil
import subprocess

import numpy as np
import pydicom
import matplotlib
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from inference.UNetInferenceAgent import UNetInferenceAgent
from utils.utils import med_reshape


date_and_time = datetime.datetime.now().strftime('%B %d, %Y. %H:%M:%S') 

def load_dicom_volume_as_numpy_from_list(dcmlist):
    """Loads a list of PyDicom objects a Numpy array.
    Assumes that only one series is in the array

    Arguments:
        dcmlist {list of PyDicom objects} -- path to directory

    Returns:
        tuple of (3D volume, header of the 1st image)
    """
    slices = [np.flip(dcm.pixel_array).T for dcm in sorted(dcmlist, key=lambda dcm: dcm.InstanceNumber)]

    # We return header so that we can inspect metadata properly.
    # Since for our purposes we are interested in "Series" header, we grab header of the
    # first file (assuming that any instance-specific values will be ighored - common approach)
    hdr = dcmlist[0]
   
    # We also zero-out Pixel Data since the users of this function are only interested in metadata
    hdr.PixelData = None
    
    return (np.stack(slices, 2), hdr)

compute_volume_1 = lambda x: 1 if x == 1 else 0
compute_volume_2 = lambda x: 1 if x == 2 else 0
v_compute_volume_1 = np.vectorize(compute_volume_1)
v_compute_volume_2 = np.vectorize(compute_volume_2)
    
def get_predicted_volumes(pred):
    """Gets volumes of two hippocampal structures from the predicted array

    Arguments:
        pred {Numpy array} -- array with labels. Assuming 0 is bg, 1 is anterior, 2 is posterior

    Returns:
        A dictionary with respective volumes
    """
    # Compute the volume of your hippocampal prediction
    volume_ant = np.sum(v_compute_volume_1(pred))
    volume_post = np.sum(v_compute_volume_2(pred))
    total_volume = volume_ant + volume_post

    return {"anterior": volume_ant, "posterior": volume_post, "total": total_volume}

def create_report(inference, header, orig_vol, pred_vol):
    """Generates an image with inference report

    Arguments:
        inference {Dictionary} -- dict containing anterior, posterior and full volume values
        header {PyDicom Dataset} -- DICOM header
        orig_vol {Numpy array} -- original volume
        pred_vol {Numpy array} -- predicted label

    Returns:
        PIL image
    """
    # The code below uses PIL image library to compose an RGB image that will go into the report
    # A standard way of storing measurement data in DICOM archives is creating such report and
    # sending them on as Secondary Capture IODs (http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.8.html)
    # Essentially, our report is just a standard RGB image, with some metadata, packed into 
    # DICOM format. 

    pimg = Image.new("RGB", (1000, 1000))
    draw = ImageDraw.Draw(pimg)

    header_font = ImageFont.truetype("assets/RobotoMono-Regular.ttf", size=40)
    main_font = ImageFont.truetype("assets/RobotoMono-Regular.ttf", size=20)

    def get_image(slice_index):
        slice = orig_vol[slice_index, :, :]
        slice = ((slice/np.max(slice))*0xff).astype(np.uint8)
        return Image.fromarray(slice, mode='L').convert("RGBA")
    
    def get_mask(slice_index, mask_number):
        slice = pred_vol[slice_index, :, :]
        slice = v_compute_volume_1(slice) if mask_number == 1 else v_compute_volume_2(slice)
        slice = (slice * 0xff).astype(np.uint8)
        return Image.fromarray(slice, mode='L')
    
    def get_overlay(mask_size, mask1, mask2):
        overlay = Image.new('RGBA', mask_size, (255, 255, 255, 0))
        drawing = ImageDraw.Draw(overlay)
        drawing.bitmap((0, 0), mask1, fill=(255, 0, 0, 128))
        drawing.bitmap((0, 0), mask2, fill=(0, 255, 0, 128))
        return overlay
    
    def get_slice_thumbnail(slice_index, new_size):
        image = get_image(slice_index)
        mask1 = get_mask(slice_index, 1)
        mask2 = get_mask(slice_index, 2)  
        overlay = get_overlay(mask1.size, mask1, mask2)
        thumbnail = Image.alpha_composite(image, overlay)
        thumbnail = thumbnail.resize(new_size)
        return thumbnail
    
    n_slices = orig_vol.shape[0]
    orig_vol = med_reshape(orig_vol, (n_slices, 64, 64))
    
    side = 200
    new_size = (side, side)
    range1 = range(0, 1000 - 1, side)
    n_thumbnails = 0
    for y in range1:
        for x in range1:
            n_thumbnails += 1

    black = (0, 0, 0)
    white = (255, 255, 255)
    
    slice_step = 1. * n_slices / n_thumbnails
    slice_index = 0
    for y in range1:
        for x in range1:
            slice_index += slice_step
            int_slice_index = int(np.round(slice_index))
            if int_slice_index >= n_slices: 
                continue
            pimg.paste(get_slice_thumbnail(int_slice_index, new_size), box=(x, y))
            slice_string = f'{int_slice_index}'
            offset = 25
            draw.text((x + side - offset - 50 + 1, y + side - offset + 1), slice_string, black, font=main_font)
            draw.text((x + side - offset - 50, y + side - offset), slice_string, white, font=main_font)
          
    title = "HippoVolume.AI"
    draw.text((11, 1), title, black, font=header_font)
    draw.text((10, 0), title, white, font=header_font)
    global date_and_time
    volume_factor = header.SliceThickness * header.PixelSpacing[0] * header.PixelSpacing[1]
    anterior_volume = inference['anterior'] * volume_factor
    posterior_volume = inference['posterior'] * volume_factor
    total_volume = inference['total'] * volume_factor
    long_text = (f'Patient ID: {header.PatientID}\n'
                 f'Time: {date_and_time}\n'
                 f"Anterior volume: {anterior_volume},  posterior volume: {posterior_volume}, total volume: {total_volume}\n")
    draw.multiline_text((11, 91), long_text, black, font = main_font)
    draw.multiline_text((10, 90), long_text, white, font = main_font)
    shape = [(0, 0), (1000 - 1, 1000 - 1)] 
    draw.rectangle(shape, outline ="red")
            
    return pimg

def save_report_as_dcm(header, report, path):
    """Writes the supplied image as a DICOM Secondary Capture file

    Arguments:
        header {PyDicom Dataset} -- original DICOM file header
        report {PIL image} -- image representing the report
        path {Where to save the report}

    Returns:
        N/A
    """
    # Set up DICOM metadata fields. Most of them will be the same as original file header
    print(header)
    out = pydicom.Dataset(header)

    out.file_meta = pydicom.Dataset()
    out.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    out.is_little_endian = True
    out.is_implicit_VR = False

    # We need to change class to Secondary Capture
    out.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    out.file_meta.MediaStorageSOPClassUID = out.SOPClassUID

    # Our report is a separate image series of one image
    out.SeriesInstanceUID = pydicom.uid.generate_uid()
    out.SOPInstanceUID = pydicom.uid.generate_uid()
    out.file_meta.MediaStorageSOPInstanceUID = out.SOPInstanceUID
    out.Modality = "OT" # Other
    global date_and_time
    out.SeriesDescription = f"HippoVolume.AI ({date_and_time})"

    out.Rows = report.height
    out.Columns = report.width

    out.ImageType = r"DERIVED\PRIMARY\AXIAL" # We are deriving this image from patient data
    out.SamplesPerPixel = 3 # we are building an RGB image.
    out.PhotometricInterpretation = "RGB"
    out.PlanarConfiguration = 0 # means that bytes encode pixels as R1G1B1R2G2B2... as opposed to R1R2R3...G1G2G3...
    
    out.BitsAllocated = 8 # we are using 8 bits/pixel
    out.BitsStored = 8
    out.HighBit = 7
    out.PixelRepresentation = 0

    # Set time and date
    dt = datetime.date.today().strftime("%Y%m%d")
    tm = datetime.datetime.now().strftime("%H%M%S")
    out.StudyDate = dt
    out.StudyTime = tm
    out.SeriesDate = dt
    out.SeriesTime = tm

    out.ImagesInAcquisition = 1

    # We empty these since most viewers will then default to auto W/L
    out.WindowCenter = ""
    out.WindowWidth = ""

    # Data imprinted directly into image pixels is called "burned in annotation"
    out.BurnedInAnnotation = "YES"

    out.PixelData = report.tobytes()

    pydicom.filewriter.dcmwrite(path, out, write_like_original=False)

def get_series_for_inference(path):
    """Reads multiple series from one folder and picks the one
    to run inference on.

    Arguments:
        path {string} -- location of the DICOM files

    Returns:
        Numpy array representing the series
    """
    # We are reading all files into a list of PyDicom objects so that we can filter them later
    dicoms = [pydicom.dcmread(os.path.join(path, f)) for f in os.listdir(path)]

    # Get PyDicom objects only belonging to 'HippoCrop' series.
    def is_dicom_valid(i, dicom):
        return dicom.SeriesDescription == "HippoCrop"
    
    series_for_inference = []
    i = 0
    for dicom in dicoms:
        i += 1
        if is_dicom_valid(i, dicom):
            series_for_inference.append(dicom)

    # Check if there are more than one series (using set comprehension).
    if len({f.SeriesInstanceUID for f in series_for_inference}) != 1:
        print("Error: can not figure out what series to run inference on")
        return []

    return series_for_inference

def os_command(command):
    sp = subprocess.Popen(command, shell=True)
    sp.communicate()
    
def select_study_dir(subdirs):
    for path in subdirs:
        dicoms = [pydicom.dcmread(os.path.join(path, f)) for f in os.listdir(path)]
        for dicom in dicoms:
            if dicom.SeriesDescription == "HippoCrop": return path
    return None
    
if __name__ == "__main__":
    # This code expects a single command line argument with link to the directory containing
    # routed studies
    if len(sys.argv) != 2:
        print("You should supply one command line argument pointing to the routing folder. Exiting.")
        sys.exit()

    # Find all subdirectowries within the supplied directory. We assume that 
    # one subdirectory contains a full study
    subdirs = [os.path.join(sys.argv[1], d) for d in os.listdir(sys.argv[1]) if
                os.path.isdir(os.path.join(sys.argv[1], d))]
    
    # Get the latest directory
    study_dir = select_study_dir(subdirs)

    print(f"Looking for series to run inference on in directory {study_dir}...")

    volume, header = load_dicom_volume_as_numpy_from_list(get_series_for_inference(study_dir))
    print(f"Found series of {volume.shape[2]} axial slices")
    
    # Run inference
    print("HippoVolume.AI: Running inference...")
    inference_agent = UNetInferenceAgent(device="cpu", parameter_file_path=r"../model.pth")
    pred_label = inference_agent.single_volume_inference_unpadded(np.array(volume))
    pred_volumes = get_predicted_volumes(pred_label)

    # Create and save the report
    print("Creating and pushing report...")
    report_save_path = r"/home/workspace/out/report1.dcm"
    report_img = create_report(pred_volumes, header, volume, pred_label)

    save_report_as_dcm(header, report_img, report_save_path)

    # Install dcmtk tool using 'sudo apt install dcmtk'
    # Send report to the Orthanc server (that runs on port 4242 of the local machine)
    command_storescu = f'sudo storescu 127.0.0.1 4242 -v -aec HIPPOAI +r +sd  {report_save_path}'
    os_command(command_storescu)

    print(f"Inference successful on {header['SOPInstanceUID'].value}, out: {pred_label.shape}",
          f"volume ant: {pred_volumes['anterior']}, ",
          f"volume post: {pred_volumes['posterior']}, total volume: {pred_volumes['total']}")
          
