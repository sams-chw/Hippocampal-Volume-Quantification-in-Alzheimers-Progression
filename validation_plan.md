# Validation Plan

## What is the intended use of the product?

This software is capable of quantifying the hippocampal volume of patients in order to keep track of the progression of their Alzheimer disease.

This software is not intended to replace doctors. It is rather intended to help doctors keep track of the progressions of their patients with Alzheimer disease. So that, this software will alleviate the work overload of doctors. Moreover, patients and their relatives can also see the reports of this software and be more aware of the progression of the Alzheimer disease.

## How was the training data collected?

The data we are using is the "Hippocampus" dataset from the Medical Decathlon competition. This dataset is stored as a collection of NIFTI files, with one file per volume, and one file per corresponding segmentation mask. The original images here are T2 MRI scans of the full brain. In this dataset we are using cropped volumes where only the region around the hippocampus has been cut out.

## How did you label your training data?

All data has been labeled and verified by human experts (radiologists), and with the best effort to mimic the accuracy required for clinical use. The images in the training are labeled by the following convention: - The Anterior part of the Hippocampus is labeled as 1 - The Posterior part of the Hippocampus is labeled as 2 - All other part (the background) is labeled as 0. The dataset contains the segmentations of the right hippocampus

## How was the training performance of the algorithm measured and how is the real-world performance going to be estimated?

The training performance was measured by using the Dice coefficient and the Jaccard metric. Here are the results:

mean_dice: 0.8998<br/>
mean_jaccard: 0.8189

The real-world ground truth can be established by acquiring silver standard of radiologist reading.

## What data will the algorithm perform well in the real world and what data it might not perform well on?

Since the model has been trained and tested using labels of the cropped volume where only the region around the right hippocampus has been cut out, this software will work with MRI scan in which the volume around the left/right hippocampus is cut out but may not work well with entire MRI brain scans in which the volume contains all the parts of the brain. ???
