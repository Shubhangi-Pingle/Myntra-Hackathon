<h1>AI-Driven Fashion Trend Forecasting</h1>

This project is designed to predict upcoming fashion trends by analyzing Instagram posts. It leverages a combination of machine learning, image processing, and sentiment analysis to identify similar fashion styles and gauge public sentiment.

<h2>Methodology</h2>
Our project follows a structured approach:

Data Collection: We collect data from Instagram using the Apify platform, including images, likes, comments, and a sample of 20 comments per post.

Image Segmentation: Images are processed using YOLO and SAM to isolate clothing items from their backgrounds, resulting in segmented images with only the clothes visible against a black background.

Feature Extraction and Dimensionality Reduction: Segmented images are passed through an autoencoder to generate a compact representation (latent space). PCA is then applied to reduce the dimensionality of these representations.

Clustering: The PCA-transformed latent spaces are clustered using the K-Means algorithm, with each cluster representing a distinct fashion style.

Sentiment Analysis: Comments associated with each post are analyzed using the AdaBoost algorithm to determine public sentiment towards the fashion styles.

Data Visualization: A Power BI dashboard displays statistics for each cluster, including the number of images, likes, comments, and sentiment distribution.

Repository Structure
The repository is organized as follows:

data: Contains all datasets, both raw and cleaned.
models: Holds the machine learning models' weights (excluding the large SAM model).
research: Includes Jupyter notebooks used for determining the number of clusters and training the autoencoder.
images: Contains "original_images" and "segmented_images" directories for the respective types of images.
python scripts: Series of Python files for each pipeline step, executed in the following order:
data_preprocessing.py
download_images.py
image_segmentation.py
latent_space_creator.py
latent_space_clustering.py
Dashboard: Contains the Power BI dashboard file.
How to Use
To use this project, execute the Python scripts in the specified order.

Technologies Used
Python: The primary programming language used for the project.
YOLO & SAM: Algorithms for image segmentation.
PCA & KMeans: Techniques for dimensionality reduction and clustering.
AdaBoost: Algorithm for sentiment analysis.
PowerBI: Tool for data visualization and dashboard creation.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/macedoti13/Future-Fashion-Trends-Forecasting.git
cd Future-Fashion-Trends-Forecasting
Install the required dependencies:

Execute the Python scripts in order:

bash
Copy code
python data_preprocessing.py
python download_images.py
python image_segmentation.py
python latent_space_creator.py
python latent_space_clustering.py
