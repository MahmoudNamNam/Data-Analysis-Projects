# Police and Weather Analysis

## Introduction
This project involves the analysis of policing activities in Rhode Island and its correlation with weather conditions. Two datasets, one related to policing ('ri.csv') and another related to weather ('weather.csv'), were used for this analysis. The project aims to explore patterns, trends, and correlations between policing data and various weather features.

## Datasets
- **Policing Dataset ('police.csv'):** Contains information about traffic stops in Rhode Island, including details about stops, outcomes, and demographics of drivers.

- **Weather Dataset ('weather.csv'):** Includes weather information such as temperature, wind speed, and various weather types for the corresponding dates.

## Code and Analysis
- **Data Loading:** The datasets were loaded into Python using the pandas library.

- **Data Cleaning:** Null values were handled, and unnecessary columns were dropped from the policing dataset.

- **Merging Datasets:** The datasets were merged based on the 'DATE' column to combine information about traffic stops and weather conditions.

- **Temporal Analysis:** Temporal patterns were explored by extracting month and hour information from the date and stop time columns. This information was used to visualize monthly and hourly trends in traffic stops based on clear weather conditions.

- **Correlation Analysis:** Correlation between weather features and traffic stops was explored, focusing on clear weather conditions.

- **Visualization:** Matplotlib and Seaborn were used for visualizations, including bar plots to show the impact of clear weather on traffic stops.

## Files
- **Analysis.ipynb:** Jupyter Notebook containing the Python code for data analysis.

- **police.csv:** Policing dataset.

- **weather.csv:** Weather dataset.

## Usage
1. Clone the repository: `git clone https://github.com/MahmoudNamNam/Data-Analysis-Projects.git`

2. Navigate to the project directory: `cd police-weather-analysis`

3. Open and run the Jupyter Notebook: `jupyter notebook Analysis.ipynb`

## Dependencies
- pandas
- matplotlib
- seaborn

Install the dependencies using: `pip install pandas matplotlib seaborn`

