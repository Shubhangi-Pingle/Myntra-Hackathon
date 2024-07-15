<h1>AI-Driven Fashion Trend Forecasting</h1>

This project is designed to predict upcoming fashion trends by analyzing Instagram posts. It leverages a combination of machine learning, image processing, and sentiment analysis to identify similar fashion styles and gauge public sentiment.

<h2>Methodology</h2>
Our project follows a structured approach:

<h3>Data Collection:</h3> We collect data from Instagram using the Apify platform, including images, likes, comments, and a sample of 20 comments per post.

<h3>Image Segmentation:</h3> Images are processed using YOLO and SAM to isolate clothing items from their backgrounds, resulting in segmented images with only the clothes visible against a black background.

<h3>Feature Extraction and Dimensionality Reduction:</h3> Segmented images are passed through an autoencoder to generate a compact representation (latent space). PCA is then applied to reduce the dimensionality of these representations.

<h3>Clustering:</h3>The PCA-transformed latent spaces are clustered using the K-Means algorithm, with each cluster representing a distinct fashion style.

<h3>Sentiment Analysis:</h3> Comments associated with each post are analyzed using the AdaBoost algorithm to determine public sentiment towards the fashion styles.

<h3>Data Visualization:</h3> A Power BI dashboard displays statistics for each cluster, including the number of images, likes, comments, and sentiment distribution.

<h2>Repository Structure</h2>
The repository is organized as follows:

<h3>data:</h3> Contains all datasets, both raw and cleaned.
<h3>models:</h3> Holds the machine learning models' weights (excluding the large SAM model).
<h3>research:</h3> Includes Jupyter notebooks used for determining the number of clusters and training the autoencoder.
<h3>images:</h3> Contains "original_images" and "segmented_images" directories for the respective types of images.
<h3>Python scripts:</h3> Series of Python files for each pipeline step, executed in the following order:
<pre>
 1.Preprocessing.py
 2.Image_Download.py
 3.Segmentation.py
 4.Creation_Latent_Spave.py
 5.Image_Clustering.py
 6.Dashboard: Contains the Power BI dashboard file.
</pre>
How to Use
To use this project, execute the Python scripts in the specified order.

<h2>Technologies Used</h2>
<pre>
Python: The primary programming language used for the project.
YOLO & SAM: Algorithms for image segmentation.
PCA & KMeans: Techniques for dimensionality reduction and clustering.
AdaBoost: Algorithm for sentiment analysis.
PowerBI: Tool for data visualization and dashboard creation.
</pre>
Installation
Clone the repository:
<pre>
bash
Copy code
git clone https://github.com/Shubhangi-Pingle/Myntra-Hackathon.git
cd Myntra-Hackathon
</pre>

Execute the Python scripts in order:

bash
Copy code
<pre>
python Preprocessing.py
python Image_Download.py
python Segmentation.py
python Creation_Latent_Spave.py
python Image_Clustering.py
</pre>
