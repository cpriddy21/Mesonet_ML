# About
For every data value recorded by the Kentucky Mesonet, a quality value of Good, Suspect, Warning, or Failure (0,1,2,3) is assigned. These assignments are currently made either through 
automated processes or after human analysis. The goal of this project was to develop a model capable of detecting missed errors in the data for differnt variables. Two team 
members worked with a client from the Kentucky Mesonet to develop the model. Overall, this project aims to enhance the quality assurance and quality control (QA/QC) processes for the 
Kentucky Mesonet, offering an enhanced solution for detecting errors in their data.

# Key Features
  - Utilizes a neural network model trained on a data of over 5 million records of climate and weather data.
  - A database was created by partitioning and segmenting the dataset given
  - Model achieves an accuracy rate of approximately 67%
  - Model accepts an input file containing climate and weather data and provides feedback on the data

# Technologies, Skills, and Frameworks
  - **Tensorflow with Keras API**: Model training and development 
  - **Scikit-learn**: Model training, evaluation, and prediction 
  - **Pandas**: data analysis 
  - **LaTeX**: documentation
  - **Waterfall methodology**
  - **Docker**: Containerization
  - **Collaboration/Communication**: Meet the client's needs, team alignment

# Running The Project

1. Install Docker if it is not on your machine
2. Run the following commands in order: 
```
  # Pull the Docker Image
  docker pull dariacasey/mesonet_ml:v1

  # Start the container 
  docker run -d --name mesonet_ml_container dariacasey/mesonet_ml:v1

  # After it is finished, stop the container and remove it
  docker stop mesonet_ml_container
  docker rm mesonet_ml_container
```
# Documentation
[Technical](/Users/daria/Downloads/CS496__MESONET_Project.pdf)

[Organizational](/Users/daria/Downloads/CS496__MESONET_ProjectOrg.pdf)



