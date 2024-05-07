## Final Project
### ITCS 6166: Communication And Computer Networks
#### Team 4:
- **Foram Shah**
- **Samarth Bhole**
- **Parshwa Shah**
- **Theodore Huang**

## Project Description:
This project implements A Lightweight Graph Transformer Network for Human Mesh Reconstruction from 2D Human Pose. The project is based on the paper "A Lightweight Graph Transformer Network for Human Mesh Reconstruction". The paper was published in 2021. The project is implemented in PyTorch. We connected the project to a light weight openpose implementation to get the 2D human pose and then used GTRS model to get the 3D human mesh. 

## Project Structure:
- **GTRS**
    - Contrains the implementation of the GTRS model. From [link](https://github.com/zczcwh/GTRS)
- **PoseDetector**
    - Contains the implementation of the OpenPose model. From [link](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)
- **Tests**
    - Contains the test data for the project
- **models**
    - Contains the trained models for the project (GTRS and OpenPose) (Exported)
- **requirements.txt**
    - Contains the required libraries for the project
- **liveWebCam.py**
    - Contains the code to run the project on live webcam

## How to run the project:
- Clone the repository
- Create a virtual environment using the command `conda create -n <env_name> python=3.9`
- Install the required libraries using the command `pip install -r requirements.txt`
- Run the project using the command `python liveWebCam.py`