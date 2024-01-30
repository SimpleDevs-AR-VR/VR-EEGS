# EEG Collection Procedure

_This submodule focuses on collecting EEG raw signal data from BCI hardware such as the **Muse 2 Headband**._

## Hardware Requirements

* **Muse 2 Headband:** The primary Brain-Computer Interface (BCI) used in this study. [Available here](https://choosemuse.com/products/muse-2)
* A portable PC (Mac, Windows, or Linux)

## Software Requirements

* **Petal Metrics (or some other LSL Streaming software equivalent):** The LSL pipe application that directly connects to the Muse 2 headband via bluetooth and creates an LSL stream that can be accessed by external applications and scripts. [Available here](https://petal.tech/downloads)
* (OPTIONAL) A python environment such as **Jupyter Notebook**, which comes as a standalone system or is availabe through the **Anaconda Distribution**.

## Recording Procedure

**Petal Metrics** will connect to the **Muse 2 Headband** via Bluetooth and will aggregate + save the collected EEG data to `.csv` files stored on the PC hosting **Petal Metrics**. Furthermore, **Petal Metrics** also pipes the data to an LSL stream, which enables other scripts or programs to peak into the EEG data.

### Base Procedure

1. Place the **Muse 2 Headband** onto the subject's frontal scalp and turn on the device. **DO NOT CONNECT THE DEVICE TO YOUR PC OR SMARTPHONE.**
2. Activate the **Petal Metrics** application on your PC.
3. Select the name of the stream of your own choice, and click "Stream". 
4. Once your session is completed, click the "Stop Streaming" button in **Petal Metrics**.

If done correctly, the **Muse 2** and **Petal Metrics** should be able to connect to one another and start communicating as well as log data onto your PC in real time.

To access the saved EEG data, you must navigate to the location on your PC where Petal Metrics' program data stored its cached data. This will be different for each operating system type.

* **Windows**: Navigate to the following location:

````
Users > <your_PC_username> > AppData > Roaming > petal_metrics > data_logs
````

* **MacOSX**: Navigate to the following location:

````
~/Library/Application Support/petal_metrics/data_logs
````

* **Linux**: Navigate to the following location:

````
Home > .config > petal_metrics > data_logs
````

### Testing the Connection

One thing that MUST be performed prior to any data collection procedure is to test whether the **Muse 2** is outputting sensible data. This can be done by the provided `muse2_stream.ipynb` python notebook. If you have **Jupyter Notebook** installed, you can also open this notebook file in that distribution.

The system will provide you a method to see the EEG data's frequency channels (produced via Welch's method) over epochs of time. Simply run the first half of the notebook and follow the instructions provided.