## aai-530-group3-final-project

## IoT-driven Smart Air Quality Monitoring And Prediction System (SmartAQ) using deep neural networks

This project is developed using tensorflow .

The goal of this project is to develop a **medium-term prediction model** for **PM2.5 concentrations** in Beijing using **deep learning techniques**, specifically **Long Short-Term Memory (LSTM) networks**. Given the **hourly recorded PM2.5 data** and **meteorological variables** from **2010 to 2015**, the project aims to predict **PM2.5 and PRES (Atmospheric Pressure)** levels for the next **12 hours** by leveraging historical trends, seasonal variations, and meteorological influences.  

By utilizing past PM2.5 and PRES readings along with meteorological conditions such as **temperature, atmospheric pressure, rainfall, snowfall, and dew point**, the model aims to:  
- **Quantify pollution trends** over time.  
- **Incorporate meteorological dependencies** to improve the accuracy of PM2.5 predictions.  
- **Identify seasonal variations** in pollution patterns.  
- **Assess policy effectiveness** by analyzing historical and predicted trends.

### **Business Use and Impact**  

The insights generated from this forecasting model will be valuable for **policymakers, environmental agencies, and urban planners**, enabling them to:  
1. **Monitor air pollution trends** in real-time.  
2. **Implement proactive pollution control measures** to mitigate health risks.  
3. **Assess the impact of policy interventions** such as emission reductions and energy transition initiatives.  
4. **Predict high-risk periods** and inform the public and industries about potential hazards.  

By providing accurate forecasts of PM2.5 concentrations, the model supports **data-driven decision-making** and promotes **effective environmental policies** to improve air quality and public health outcomes in Beijing.

### **Loading the Dataset**

The features in the dataset are described below:

| **Feature**                     | **Description**                                                                                                 |
|----------------------------------|----------------------------------------------------------------------------------------------------------------|
| **No (Index/Serial Number)**     | **A unique identifier for each row** in the dataset.                                                            |
| **Year**                         | The year in which the **PM2.5 and meteorological data** were recorded (2010–2015).                              |
| **Month**                        | The month (1–12) in which the data was recorded. Useful for capturing **seasonal variations** in PM2.5 and IWS levels. Higher levels in winter due to **coal-based heating**. |
| **Day**                          | The day of the month when the data was recorded (1–31).                                                         |
| **Hour**                         | The hour of the day when the data was recorded (0–23).                                                          |
| **Dew Point**                    | **Dew point temperature** in Celsius.                                                                           |
| **Temperature**                  | **Atmospheric temperature** in Celsius.                                                                         |
| **Pressure**                     | **Atmospheric pressure** in hPa.                                                                                |
| **CBWD (Combined Wind Direction)** | A categorical variable representing **wind direction** (e.g., NW, NE, SW, SE).                                   |
| **IWS (Integrated Wind Speed)**  | The cumulative **wind speed** in meters per second (m/s) over a specific period.                                |
| **IS (Integrated Snowfall)**     | The cumulative **snowfall** in millimeters (mm) over a specific period.                                         |
| **IR (Integrated Rainfall)**     | The cumulative **rainfall** in millimeters (mm) over a specific period.                                         |
| **PM2.5 (Target Variable)**      | The concentration of **fine particulate matter (PM2.5)** in micrograms per cubic meter (µg/m³).                 |


## EDA observations:

1. From the above observations it is found that pm2.5 has 41,757 non-null values, while other features have 43,824 values. This means 2,067 missing values in pm2.5 (≈4.7% of the data). So we are using linear interpolation to fill missing values smoothly.

2. distribution of pm2.5 : Mean pm2.5 = 98.61 µg/m³ (High pollution on average), Std Dev = 92.05 → Large spread, meaning significant variations in pollution levels. Min = 0, Max = 994  Wide range, indicating extreme pollution days.  high standard deviation suggests frequent fluctuations in pollution levels.

3. 	Temperature (TEMP):
    Mean: 12.45°C, Min: -19°C, Max: 42°C.
    Winter temperatures go as low as -19°C, possibly linked to higher coal-based heating emissions.
    Higher PM2.5 levels are expected in winter due to increased coal burning.

4. Seasonal variations significantly affect PM2.5 levels—higher pollution in winter due to coal heating & stagnant air conditions.

5. Wind speed analysis is crucial to predicting PM2.5 levels since strong winds can clear pollution.

6. Very little snowfall overall, meaning snow is unlikely to have a major impact on pollution.

7. Rainfall events can be explored to understand PM2.5 reduction during precipitation days.

8. Comparing different years will reveal if pollution worsened or improved.

9. Higher pollution expected in winter (December–February) due to heating emissions. Lower pollution expected in summer (June–August) due to stronger winds & rainfall.

## Handling missing values:

These missing values are distributed throughout the dataset and are not uniformly distributed.Certain periods show clusters of missing values (entire sections are missing) and Other periods have sporadic gaps. Missing data in pm2.5 might be due to sensor failure, data collection issues, or external factors like weather or maintenance. This pattern suggests the need to investigate if missing data aligns with specific months, seasons, or hours of the day. Gaps in pm2.5 data could affect trend and seasonal analysis. Imputation or interpolation will likely be needed. The high number of missing values might make it challenging to perform accurate predictions without a robust handling strategy. 


![alt text](images/pm2.5%20missing%20value%20counts.png)

![alt text](images/missing%20value%20patterns.png)




## Handling outliers:

In IWS, numerous extreme outliers exceeding 300–500 m/s, with unrealistic values close to 600.

In IR, outliers clearly visible above 20 mm, but heavy rainfall may be valid in certain periods.

In IS, outliers visible beyond 15 mm, though snowfall in Beijing can vary widely.

So based on above observations remove outliers for features like pm2.5, Iws, Is, and Ir

By observing at the above plots it clearly explans that rolling/moving average describes the patters very well for predictions. 

So as part of feature engineering we would like to perform some rolling features on pm2.5 and Pressure .

## Feature Engineering:

As part of feature engineering below new lag and interaction features are extracted ,

cbwd_encoded
cbwd_angle
cbwd_sin
cbwd_cos
pm2.5_rolling_3h
pm2.5_rolling_24h
pm2.5_lag_1
Iws_lag_1
day_avg_pm2.5
temp_dew_spread
Iws_rain_interaction



## Model training and evaluations

## AI Algorithms and Model Selection

Linear regression performed well with below performance scores however struggled with huge errors and difficulty learning the seasonal trends or patterns of the prediction pm2.5.

Mean Squared Error (MSE): 280.2022
Root Mean Squared Error (RMSE): 16.7392
Mean Absolute Error (MAE): 10.0037
R² Score: 0.9685


We experimented with both Linear Regression and LSTM for PM2.5 prediction. While Linear Regression served as a baseline model, it struggled with the nonlinear dependencies, seasonal trends, and temporal patterns in air pollution data, leading to suboptimal performance. In contrast, LSTM, a deep learning model designed for time-series forecasting, significantly outperformed traditional ML approaches. LSTM effectively remembers long-term dependencies, capturing seasonal fluctuations, daily cycles, and complex interactions between meteorological factors. Unlike conventional models, it processes data sequentially, preserving historical context, which enhances predictive accuracy. Empirical results showed that LSTM achieved lower Mean Squared Error (MSE) and Mean Absolute Error (MAE), validating its superiority in forecasting PM2.5 and Pressure levels and reinforcing the importance of deep learning in air quality prediction.


The Bidirectional LSTM with Attention Mechanism was employed to improve the accuracy of multi-step time series forecasting for both PM2.5 concentration and atmospheric pressure (PRES) predictions. The model architecture consisted of two Bidirectional LSTM layers (100 and 50 units, respectively), an Attention Layer, and a Dense(12) output layer to generate predictions for the next 12 time steps. The Swish activation function was used instead of ReLU to enhance gradient flow, and Dropout (0.1) was applied to prevent overfitting. The model was trained using the Adam optimizer (learning rate = 0.0005) with EarlyStopping to avoid overfitting and ReduceLROnPlateau for adaptive learning rate adjustments. The evaluation metrics included Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² Score to assess predictive accuracy.

Below are the last 12 step predictions for pm2.5 and also atmospheric pressure,

![alt text](images/pm2.5%20predictions.png)

![alt text](images/atmospheric%20pressure%20preds.png)

The final model demonstrated notable improvements in trend prediction and generalization across both objectives. For PM2.5 forecasting, the model effectively captured long-term trends and flat regions, though sharp spikes remained slightly underpredicted. The results showed an MSE of 0.0040, an MAE of 0.0385, and an RMSE of 0.0633, with an R² score of 0.7809, indicating good predictive performance. Meanwhile, for atmospheric pressure prediction, the model achieved a significantly high accuracy, with an MSE of 0.0024, an RMSE of 0.0359, an MAE of 0.0486, and an R² score of 0.9245, demonstrating excellent variance explanation. Visual inspection confirmed that the model successfully captured the overall patterns, with minimal deviations from actual values. However, sudden PM2.5 spikes were still slightly underestimated, suggesting room for improvement in handling extreme values.

## Conclusions: 

To enhance the model further, Bidirectional LSTMs were used to incorporate past and future dependencies, while the Attention Layer allowed the model to focus on critical time steps, improving long-range dependency handling. Fine-tuning efforts, such as reducing the learning rate (0.000001), optimizing Dropout (0.1), and using EarlyStopping, contributed to model stability. Overall, the Bidirectional LSTM + Attention + Swish model provided a robust approach for multi-step forecasting. Future improvements will focus on feature engineering, data augmentation, and advanced architectures to further refine spike prediction accuracy and enhance model interpretability.


## References:

University of California, Irvine. (n.d.). Beijing PM2.5 data. UCI Machine Learning Repository. Retrieved January 19, 2025, from https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data

iang, X., Zou, T., Guo, B., Li, S., Zhang, H., Zhang, S., Huang, H., & Chen, S. X. (2015). Assessing Beijing’s PM2.5 pollution: Severity, weather impact, APEC, and winter heating. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 471(2182), 20150257. https://doi.org/10.1098/rspa.2015.0257

Padmanabhan, A., Ng, K., & Cole, M. (2019). PyTorch experiments on NLP and RNN. In Mobile artificial intelligence projects. Packt Publishing, Limited. Retrieved from https://www.packtpub.com/en-nz/product/mobile-artificial-intelligence-projects-9781789344073/chapter/pytorch-experiments-on-nlp-and-rnn-6/section/pytorch-experiments-on-nlp-and-rnn-6

Understanding LSTM networks – Colah’s blog. (n.d.). Retrieved from https://colah.github.io/posts/2015-08-Understanding-LSTMs



