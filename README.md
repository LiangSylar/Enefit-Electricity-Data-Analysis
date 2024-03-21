Enefit-Electricity-Data-Analysis

Energy imbalance, denoting a mismatch between produced energy and actual demands, directly leads to energy waste. Forecasting energy consumption is a crucial aspect of resolving energy imbalance. Enefit Electricity data is collected to address the data imbalance issue in real applications. Raw data comes in a collection of 7 files in the form of time series, where the biggest file has more than 3 million records. After preprocessing the data, a data exploration stage by Tableau revealed a distinct correlation between weather conditions and electricity consumption. The forecasting of time series data is achieved by adapting LSTM and deep learning models. Separate models are used for different counties (locations) to enable more accurate forecasting. Results show that replacing the linear prediction head in LSTM with ResNet18 improves the predicting performance. 

The presence of 7 data files, the enriched attributes in each dataset, and the big data size make the data preprocessing steps unignorable. Besides merging data and dealing with null entries, other preprocessing issues come with the original data files, such as data duplicates and the time lag issue. The preprocessing procedure includes the following steps: 1) Null Data Imputation; 2) Feature Engineering and Data Duplicates Removal; 3) retrieve missing location information for weather data; 4) Memory reduction by optimizing data types; 5) Integration of 7 data files.   

 
<img src="https://github.com/LiangSylar/Enefit-Electricity-Data-Analysis/assets/64362092/132e6b0e-0763-4a8a-8544-f2046cc4665f" alt="Image" height="300" title="Tableau visualization of temperature vs. months">
  
<img src="https://github.com/LiangSylar/Enefit-Electricity-Data-Analysis/assets/64362092/ed5ad1c6-7977-4ca8-baec-4835526f10f3" alt="Image" height="300" title="Plot of Train MSE">

<img src="https://github.com/LiangSylar/Enefit-Electricity-Data-Analysis/assets/64362092/e4106029-7070-4476-b403-1a567354bc94" alt="Image" height="300" title="Plot of Test MSE">
