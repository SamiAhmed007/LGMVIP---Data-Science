# LGMVIP-Data-Science
# *Task-1: Iris Flowers Classification ML Project*

This repository contains an introductory machine learning project focused on classifying Iris species using numerical features of the Iris flower. This project is commonly referred to as the "Hello World" of machine learning due to its simplicity and ease of use for beginners. The Iris flowers dataset, which consists of numerical attributes, is ideal for learning how to load, process, and apply supervised machine learning algorithms.

## Project Description

The goal of this project is to classify Iris flowers into three species—**Setosa**, **Versicolor**, and **Virginica**—based on the following attributes:

- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

The dataset is small, making it easy to load and work with in-memory, without needing special transformations or scaling.

## Project Structure

- **Iris.csv**: The dataset containing 150 samples with four features and a target species.
- **Task-1 Iris_jupiter_NB.ipynb**: Jupyter notebook containing the steps for data analysis, model training, and evaluation.
- **iris.data & iris**: Additional data files used in the project.

## Installation

To run this project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/iris-flowers-classification.git
   cd iris-flowers-classification
  
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   
4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Task-1\ Iris_jupiter_NB.ipynb

## Workflow Overview
1. Data Loading and Exploration:
   - The dataset is loaded and checked for any missing or inconsistent values.
   - Basic statistics and visualizations (histograms, scatter plots) are used to understand the data distribution.
2. Model Selection:
   - Multiple machine learning models are implemented, including:
      - **Logistic Regression**
      - **KNeighbors Classifier**
      - **Random Forest**
      - **Decision Tree**
   - Hyperparameter tuning is performed to optimize model performance.

3. Model Evaluation:
   - Models are evaluated using accuracy, precision, recall, and F1-score.
   - Confusion matrices are generated to analyze misclassification.

4. Results:
   - The best-performing model(all models gave the same accuracy) achieved an accuracy of 96.67%, demonstrating good predictive capability on this small dataset.
     
## Conclusion
This project provides a hands-on approach to understanding the basics of machine learning, from data preprocessing to model evaluation. It serves as a solid introduction to supervised learning algorithms and is ideal for beginners looking to start with machine learning.

# *Task-2: Stock Market Prediction and Forecasting Using Stacked LSTM*

This project focuses on predicting and forecasting stock prices using a stacked Long Short-Term Memory (LSTM) neural network. LSTMs are ideal for time series forecasting due to their ability to remember long-term dependencies. The dataset used contains historical stock prices, and the model aims to predict future stock prices based on past trends.

## Project Description

The goal of this project is to build a predictive model using Stacked LSTM to forecast stock prices for the next 30 days. The project follows the typical machine learning workflow: data preprocessing, normalization, model building, and evaluation.

### Dataset

The dataset used is historical stock prices from [NSE TATA Global](https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv), consisting of the following columns:
- **Open**: Opening price of the stock.
- **High**: Highest price of the stock during the day.
- **Low**: Lowest price of the stock during the day.
- **Close**: Closing price of the stock (used for prediction).
- **Volume**: Number of stocks traded.
- **Date**: The date of each stock record.

## Project Structure

- **Task-2 Stock Market Prediction And Forecasting Using Stacked LSTM.ipynb**: Jupyter notebook containing the data analysis, model building, and predictions.
- **NSE-TATAGLOBAL.csv**: Dataset file with historical stock prices.
- **requirements.txt**: List of dependencies to install to run the project.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-market-prediction-lstm.git
   cd stock-market-prediction-lstm

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt

3. Open the Jupyter notebook:
   ```bash
   jupyter notebook Task-2\ Stock\ Market\ Prediction\ And\ Forecasting\ Using\ Stacked\  LSTM.ipynb

## Workflow Overview
1. Data Preprocessing:
    - The dataset is loaded and only the Close price is used for predictions.
    - Data is normalized using MinMaxScaler to scale the values between 0 and 1, as LSTMs are sensitive to input scale.

2. Train/Test Split:
    - The data is split into a training set (75%) and a test set (25%).
    - Time series data is transformed to create sequences of 100 time steps, which are used as inputs for the LSTM model.

3. Model Building:

    - A stacked LSTM model is created using Keras, with multiple LSTM layers and a dense output layer.
    - The model is compiled using the mean squared error loss function and the Adam optimizer.
    - The model is trained for 100 epochs with a batch size of 64.

4. Evaluation:

    - The model’s performance is evaluated using Root Mean Squared Error (RMSE).
    - Predictions are generated for both the training and test sets, and the results are inverse-transformed back to the original scale.

5. Stock Price Forecasting:

    - The trained model is used to predict stock prices for the next 30 days.
    - Predictions are visualized, showing the difference between actual and predicted values.

## Results
The model was able to predict stock prices with reasonable accuracy, as shown by the RMSE values for both the training and test sets.
The prediction for the next 30 days shows a trend that is aligned with the recent stock performance.

## Conclusion
This project demonstrates how to use stacked LSTM models for stock market prediction. By training on historical data and using LSTMs for sequence prediction, the model can generate accurate forecasts for future stock prices.

## Future Work
Implement more advanced techniques such as GRU or attention-based models for potentially better results.
Integrate additional financial indicators such as moving averages or volume to improve model accuracy.

# *Task-3: Exploratory Data Analysis on Global Terrorism Dataset*

This project focuses on performing an exploratory data analysis (EDA) on the Global Terrorism Database to extract meaningful insights and visualize trends related to terrorist activities across the globe. The dataset contains detailed information on various terrorist attacks, including dates, locations, attack types, targets, and casualties.

## Project Description

The goal of this project is to explore and visualize the data to understand key patterns and trends in terrorist activities over the years. The dataset used contains over 180,000 incidents of terrorism between 1970 and 2017.

### Dataset

The dataset used is the **Global Terrorism Database (GTD)**, containing 181,691 records and 135 features. Some of the key columns used in this analysis include:
- **Year**: The year in which the attack took place.
- **Country**: The country where the attack occurred.
- **City**: The specific city of the attack.
- **Attack Type**: The method of attack (e.g., bombing, armed assault).
- **Target Type**: The type of target (e.g., civilians, military).
- **Group**: The terrorist organization responsible for the attack.
- **Casualties**: The total number of casualties (killed + wounded).

## Project Structure

- **Task-3 Exploratory Data Analysis on Dataset - Terrorism.ipynb**: Jupyter notebook containing the complete EDA workflow.
- **globalterrorismdb_0718dist.csv**: The dataset file containing global terrorism data from 1970 to 2017.
- **images/**: Contains the visualizations generated during the analysis.

## Installation

To run this project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/terrorism-eda.git
   cd terrorism-eda

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt

3. Open the Jupyter notebook:
   ```bash
   jupyter notebook Task-3\ Exploratory\ Data\ Analysis\ on\ Dataset\ -\ Terrorism.ipynb

## Workflow Overview
1. Data Preprocessing:
    - Load and clean the dataset, handling missing values in crucial columns such as kill, wound, and location.
    - Feature engineering is done to create new columns like Casualties (sum of killed and wounded).

2. Exploratory Data Analysis:
    - **Yearly Trends**: A bar plot visualizes the number of attacks each year, showing the rise of terrorism over the decades.
    - **Terrorist Activities by Region**: An area plot highlights how terrorist activities are distributed across different regions over the years.
    - **Top Affected Countries**: A bar chart showing the top 10 countries most affected by terrorism.
    - **Top Terrorist Groups**: Analysis of the most active terrorist groups, such as Taliban and Islamic State of Iraq and the Levant (ISIL).
    - **Attack Types**: A breakdown of the different types of attacks, including bombings, armed assaults, and assassinations.
    - **Casualties by Attack Type**: A bar plot showing which types of attacks resulted in the highest number of casualties.
    - **Most Affected Cities**: A plot showing the cities with the highest number of attacks.

3. Key Insights:
    - Iraq and Afghanistan are the most affected countries, with over 24,000 and 14,000 attacks respectively.
    - The Taliban and ISIL are the most active terrorist groups, responsible for the largest number of attacks and casualties.
    - Bombings/explosions are the most common type of attack, followed by armed assaults.

## Results
The EDA provides a comprehensive overview of global terrorist activities, highlighting trends in the frequency of attacks, the most affected regions, and the deadliest types of attacks.

## Conclusion
This project demonstrates the use of exploratory data analysis techniques to extract meaningful insights from a complex.

# *Task-4: Image to Pencil Sketch with Python*

This project demonstrates how to convert an image into a pencil sketch using Python and OpenCV. The steps involve reading an image, converting it to grayscale, inverting the grayscale image, blurring the inverted image, and finally blending it with the original grayscale to create the pencil sketch effect.

## Project Description

The project converts a color image into a pencil sketch using the following steps:
1. **Grayscale Conversion**: Convert the image to grayscale, resulting in a black and white version of the image.
2. **Inversion**: Create a negative of the grayscale image to enhance the details.
3. **Blurring**: Apply a Gaussian blur to the inverted image.
4. **Pencil Sketch Effect**: Blend the grayscale image and the inverted blurred image using OpenCV's `divide` function to generate the final pencil sketch.

### Dataset

The project uses an example image of **mbappe.jpeg** for the demonstration. Any image can be used for the same process.

## Project Structure

- **Task-4 Pencil Sketch.ipynb**: Jupyter notebook containing the code for converting an image to a pencil sketch.
- **mbappe.jpeg**: Example image used for creating the pencil sketch.
- **requirements.txt**: List of dependencies to install to run the project.

## Installation

To run this project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pencil-sketch-python.git
   cd pencil-sketch-python
2. Install the required Python packages:
   ```bash
    pip install -r requirements.txt
3. Open the Jupyter notebook:
   ```bash
    jupyter notebook Task-4\ Pencil\ Sketch.ipynb

## Workflow Overview

1. Loading the Image:

    - The image is loaded using OpenCV's imread function.
    - The loaded image is displayed using the cv2_imshow function.

2. Converting to Grayscale:
    - The image is converted to grayscale using cv2.cvtColor.

3. Inverting the Grayscale Image:
    - The grayscale image is inverted by subtracting it from 255 to create a negative effect.

4. Blurring the Inverted Image:
    - A Gaussian blur is applied to the inverted image using cv2.GaussianBlur with a kernel size of (21, 21).

5. Creating the Pencil Sketch:
    - The pencil sketch is created by dividing the grayscale image by the inverted blurred image using cv2.divide function.

6. Comparison:
    - Display the original image alongside the pencil sketch for comparison.


## Results
The project successfully converts an image into a pencil sketch using a combination of image inversion, blurring, and division.

## Conclusion
This project demonstrates a simple yet effective technique to transform images into pencil sketches using Python and OpenCV. The process involves basic image manipulation techniques that can be further refined for artistic effects.

## Future Work
Explore different types of image filters for blurring (e.g., Median Blur, Bilateral Filter) to see how they affect the pencil sketch.
Implement real-time pencil sketching using webcam input.
