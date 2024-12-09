# Unimi-RDVA-project
_**Unimi project of a CNN for RDAV (Recognition of Drugs that Affect Vision)**_

This project utilizes a simple Convolutional Neural Network (CNN) to raise awareness about the dangers of driving under the influence of drugs. The repository includes all necessary files to train the neural network and create a dataset with three classes: Alcohol, Cannabis, and LSD. These substances are explored in detail in the accompanying PDF presentation.

The flat image dataset used for this project was sourced from [Dataset Ninja's Road Vehicle project](https://datasetninja.com/road-vehicle).

## Usage Instructions
1. Download the previously mentioned dataset;
2. Extract all the images from the folder "road-vehicle-DatasetNinja.tar\train\img" and save them in a folder named "flat_img" inside the same folder of this project;
3. Create a folder named "Dataset" inside the same folder of this project;
4. Run the script `DrugSimulatorDataset.py`;
5. Once the script has finished executing, place the images you want the neural network to recognize in the `output_img_distorted` folder.

### Side notes
The files `alcohol_Filter.py`, `cannabis_Filter.py`, and `lsd_Filter.py` take images from the `input_img_flat` folder and alter them, saving the various steps in the `output_img_distorted` folder. These files can be useful for testing the neural network with images different from those already present in the folder. If you run these scripts, you should keep only the images generated with the names "..._double_vision.jpg", "..._light_sensitive.jpg", and "_enhanced_colors.jpg" in the `output_img_distorted` folder. Alternatively, you can comment out the first two code snippets in each scripts that save the previous steps.
