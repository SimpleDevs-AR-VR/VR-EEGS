# VR-EEGS

_Project to interpret EEG signals from VR users._

## About this Repository

This repository contains code necessary to parse BCI data from VR users and visualize findings. Parts of this repository will cover the following topics:

|Topic or Endeavor|Description|Relevant Folder(s)|
|:--|:--|:-:|
|Meta Quest Pro Data Collection|Capture the screen recording directly from the Meta Quest Pro.|`Meta-Quest-Pro-SCRCPY/`|
|EEG Data Collection|Collect EEG data captured by the **Muse 2** BCI headset.|`EEG-Collection`|
|EEG Data Processing|Once data from our BCI headset is collected as raw `.csv` data, this data must be processed by filtering out extraneous frequencies, removing noise, etc.|`processing/`| 

## Quick Installation

This repository uses many submodules. To ensure that you have the latest version of each submodule, make sure to [perform the following commands](https://stackoverflow.com/questions/11358082/empty-git-submodule-folder-when-repo-cloned):

**If cloning the repository from scratch**:

````bash
git clone https://github.com/SimpleDevs-AR-VR/VR-EEGS.git --recursive
````

**If you forgot to add the `--recursive` flag when cloning, you can still install submodules after the fact via the following command (do note the additional `--recursive` flag provided here too)**:

````bash
git submodule update --init --recursive
````

If you cannot access these submodules, please contact Ryan Kim for permission accessibility.

## Team

* **Ryan Kim**: rk2546 [at] nyu.edu
* **Damien Chen**: wc2173 [@] nyu.edu