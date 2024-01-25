# VR-EEGS

_Project to interpret EEG signals from VR users._

## About this Repository

This repository contains code necessary to parse BCI data from VR users and visualize findings. Parts of this repository will cover the following topics:

|Topic or Endeavor|Description|Relevant Folder(s)|
|:--|:--|:-:|
|Meta Quest Pro Data Collection|Capture the screen recording directly from the Meta Quest Pro.|[GITHUB REPOSITORY](https://github.com/SimpleDevs-AR-VR/Meta-Quest-Pro-SCRCPY.git)|
|Meta Quest Pro Footage Pre-Processing|Pre-process the screen recording captured from the Meta Quest Pro by using `crop` and `lenscorrection` filters provided by **ffmpeg**.|[GITHUB REPOSITORY](https://github.com/SimpleDevs-AR-VR/Meta-Quest-Pro-LensCorrection.git)|
|EEG Data Collection|Collect EEG data captured by the **Muse 2** BCI headset.|`EEG-Collection`|
|EEG Data Processing|Once data from our BCI headset is collected as raw `.csv` data, this data must be processed by filtering out extraneous frequencies, removing noise, etc.|`processing/`|

## Team

* **Ryan Kim**: rk2546 [at] nyu.edu
* **Damien Chen**: wc2173 [@] nyu.edu