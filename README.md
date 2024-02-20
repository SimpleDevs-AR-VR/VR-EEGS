# VR-EEGS

_Project to interpret EEG signals from VR users._

## About this Repository

This repository contains code necessary to parse BCI data from VR users and visualize findings. Parts of this repository will cover the following topics:

|Topic or Endeavor|Description|Relevant Folder(s)|
|:--|:--|:-:|
|Meta Quest Pro Data Collection|Capture the screen recording directly from the Meta Quest Pro.|[GITHUB REPOSITORY](https://github.com/SimpleDevs-AR-VR/Meta-Quest-Pro-SCRCPY.git)|
|Meta Quest Pro Footage Pre-Processing|Pre-process the screen recording captured from the Meta Quest Pro by using `crop` and `lenscorrection` filters provided by **ffmpeg**.|[GITHUB REPOSITORY](https://github.com/SimpleDevs-AR-VR/Meta-Quest-Pro-LensCorrection.git)|
|Meta Quest Pro Object Tracking|We use tracking or detection libraries like YOLO to identify objects in the street footage.|[GITHUB REPOSITORY](https://github.com/SimpleDevs-AR-VR/Meta-Quest-Pro-ObjectTracking.git)|
|EEG Data Collection|Collect EEG data captured by the **Muse 2** BCI headset.|`EEG-Collection`|
|EEG Data Processing|Once data from our BCI headset is collected as raw `.csv` data, this data must be processed by filtering out extraneous frequencies, removing noise, etc.|`EEG-Processing/`|

## Video Pipeline Process

Videos captured from the Meta Quest Pro need to be processed. This processing involves controlling for the variable frame rate from the original captured footage, the alignment of timestamps from different data collection sources, etc.

The steps to process this footage is as follows:

|Step # |Description                |Purpose|Related Code|
|:-----:|:--------------------------|:-|:-:|
|1      |Recording MQP Footage      |To record the eye footage from the device directly, we need to manually steal the footage from the cameras directly.|[SCRCPY - GITHUB REPOSITORY](https://github.com/SimpleDevs-AR-VR/Meta-Quest-Pro-SCRCPY.git)|
|2      |Extracting frame timestamps|SCRCPY footage runs at a variable frame rate. When processing using packages like OpenCV, frames are processed as if they are at a consistent frame rate. This needs to be offset by tracking each frame's original timestamp for later reference.|[LENS CORRECTION - GITHUB REPO](https://github.com/SimpleDevs-AR-VR/Meta-Quest-Pro-LensCorrection.git)|
|3      |Lens Correction            |The SCRCPY footage is distorted via barrel distortion. We need to correct that using a variation of Hugin's method. Though Hugin's method is meant to fix radial rather than barrel distortion, it is still sufficient for our needs.|[LENS CORRECTION - GITHUB REPO](https://github.com/SimpleDevs-AR-VR/Meta-Quest-Pro-LensCorrection.git)|
|4      |Eye Cursor Estimation      |We have a world-to-screenpoint estimation of the eye cursor from the VR app. We need to map that back to the footage, since we corrected the footage afterwards and the footage itself is not correctly rotated to begin with.|[LENS CORRECTION - GITHUB REPO](https://github.com/SimpleDevs-AR-VR/Meta-Quest-Pro-LensCorrection.git)|
|5      |Object Tracking            |The corrected footage must be processed further by detecting which objects are present in the footage|[OBJECT TRACKING - GITHUB REPO](https://github.com/SimpleDevs-AR-VR/Meta-Quest-Pro-ObjectTracking.git)|

## Team

* **Ryan Kim**: rk2546 [at] nyu.edu
* **Damien Chen**: wc2173 [@] nyu.edu