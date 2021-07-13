# Convolutional Neural Network model deployment using Azure Machine Learning and Docker for intraoperative brain tumor classification
Final master thesis code and results for the [**Master of Science in Internet of Things**](http://masteriot.etsist.upm.es/) taught by [**Universidad Politécnica de Madrid**](https://www.upm.es/) (2020/21).
Developed by **Alberto Martín Pérez** in colaboration with **GDEM** research group and the [NEMESIS-3D-CM](http://nemesis3d.citsem.upm.es/) project.

---
# 1. Installation and set-up
## 1.1. Visual Studio Code extensions:
### _Azure extensions_
- [Azure Account](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azure-account)
- [Azure IoT Tools](https://marketplace.visualstudio.com/items?itemName=vsciot-vscode.azure-iot-tools)
- [Azure Machine Learning](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai)
- [Azure Machine Learning - Remote](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai-remote)
- [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
### _Python extensions_
- [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
- [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)
- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Python for VSCode](https://marketplace.visualstudio.com/items?itemName=tht13.python)
### _Optional extensions_
- [Better Comments](https://marketplace.visualstudio.com/items?itemName=aaron-bond.better-comments)
- [Bracket Pair Colorizer](https://marketplace.visualstudio.com/items?itemName=CoenraadS.bracket-pair-colorizer)

## 1.2. Microsoft Azure resources:
_Use the links to learn how to create these resources_
- Microsoft Azure account and suscription with credit.
- [Machine Learning workspace](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources#create-the-workspace) - Used to train, test and deploy models.
You can create a Resource group during the creation of the workspace. It will also create a
Storage account, a Container registry, a Key vault and an Application Insights resource.
- [Compute Clusters](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources#cluster) - Used to run the experiments to train and test models.
- [Kubernetes service](https://docs.microsoft.com/en-us/azure/aks/kubernetes-walkthrough-portal#create-an-aks-cluster) - Used to deploy containerized models as web services.

## 1.3. Folder structure and files:
### Folders
- **Examples**: Folder containing Python scripts with examples of how to use the
most basic classes from the **_hsi_manager.py_** library.
- **Libraries**: Folder containing all necessary Python files to train and measure PyTorch CNN,
manage and preprocess HSI data.
- **NEMESIS_images**: Folder where all hyperspectral images and ground truth maps are saved:
    - **datasets**: Sub-folder containing files of type _'IDXXXXCYY_dataset.mat'_
    - **GroundTruthMaps**: Sub-folder containing files of type _'SNAPgtIDXXXXCYY_cropped_Pre-processed.mat'_
    - **preProcessedImages**: Sub-folder containing files of type _'SNAPimagesIDXXXXCYY_cropped_Pre-processed.mat'_
    - **tif**: Sub-folder containing files of type _'IDXXXXCYY.tif'_ (raw HSI), _'IDXXXXCDYY.tif'_ (dark HSI reference) and _'IDXXXXCWYY.tif'_ (white HSI reference).
- **Results**: Folder where all results will be stored.
    - **Classification_maps**: Sub-folder containing '.png' images with classification maps. Names follow the next criteria:
    CNN architecture name _ ID patient classified _ if cross-validation was used during training _ version _ number .png 
    _Example: Conv2DNet_ID0018C09_noCV_version_1.png_
    - **Model_deployment**: Sub-folder containing information regarding time executions during model deployment and consumption.
    Measures where saved manually.
    - **Training_metrics**: Sub-folder containing information regarding classification and time metrics during training executions.
    Measures saved after gathering all stored data in the Azure portal. The file '7_azure_read_metrics.py' was used to automatically collect the metrics.

### Files
- **1_azure_connection.py**: Shows how to establish connection to an Azure Machine learning workspace using the config.json file downloaded from the Azure ML studio page.
- **2_azure_data_upload.py**: Shows how to upload data to the default Datastore from the Azure Blob storage of the Azure Machine Learning workspace.
- **3_azure_create_dataset.py**: Shows how to create datasets from the default Datastore containing the uploaded files.
- **4_azure_download_dataset.py**: Shows how to download created datasets.
- **5_azure_control_train.py**: Shows how to define an environment to run experiments in an Azure compute cluster. Uses one of these two training scripts:
    - **azure_train_experiments.py**: Which trains Conv2DNet models using the 5-fold double cross-validation implementation.
    - **azure_train_noCV_experiments.py**: Which trains a Conv2DNet model without using the 5-fold double cross-validation implementation.
- **6_azure_deploy_use_model.ipynb**: Shows how to deploy and consume a registered model using Azure Kubernetes Service and the Azure SDK for Python (no HTTP).
Uses the folowing scoring script for the web service:
    - **score_brain.py**: Scoring script that takes a registered model, preprocess a hyperspectral cubes and returns a predicted classification map with a JSON object.
- **7_azure_read_metrics.ipynb**: Shows how to automatically store registered metrics from the experiments run in Azure Machine learning into local .csv files.


## 1.4. Conda and pip packages for Python
**Python version used has been 3.8.10, since at the time, azureml-core did not support Python versions >= 3.9**
_These packages and versions have been used during the development of this thesis. They will install their corresponding dependencies._
| Package | Version |
| ------ | ------ |
| [**azureml-core**  ](https://pypi.org/project/azureml-core/1.31.0/)          |      1.31.0
| [**matplotlib**](https://pypi.org/project/matplotlib/3.4.2/)            |        3.4.2
| [**numpy**](https://pypi.org/project/numpy/1.19.3/)              |        1.19.3
| [**pandas**](https://pypi.org/project/pandas/1.3.0/)              |        1.3.0
|[**scikit-learn**](https://pypi.org/project/scikit-learn/0.24.2/)          |        0.24.2
| [**scipy**](https://pypi.org/project/scipy/1.7.0/)                |        1.7.0
| [**torch**](https://pypi.org/project/torch/1.9.0/)                |        1.9.0+cu111
| [**tqdm**](https://pypi.org/project/tqdm/4.61.1/)                  |        4.61.1
