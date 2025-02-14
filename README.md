## aai-530-group3-final-project-smartaq


## Objective:

The goal of this project is to develop a medium-term prediction model for PM2.5 concentrations in Beijing using deep learning techniques, specifically Long Short-Term Memory (LSTM)

networks. Given the hourly recorded PM2.5 data and meteorological variables from 2010 to 2015, the project aims to predict PM2.5 and IWS levels for the next 7 days using historical trends,

seasonal variations, and meteorological influences.



By leveraging past PM2.5 and IWS readings and meteorological conditions (temperature, atmospheric pressure, rainfall, snowfall, and dew point), the model will:

•	Quantify pollution trends over time.

•	Incorporate meteorological dependencies to refine PM2.5 predictions.

•	Identify seasonal variations in pollution patterns.

•	Assess policy effectiveness by analyzing historical and predicted trends.


The insights from this forecasting model will assist policymakers and environmental agencies in monitoring air pollution trends, implementing proactive pollution control measures, and 

assessing the impact of policy interventions like emission reductions and energy transitions.

## Dataset overview

The dataset used in this project has following features,


No (Index/Serial Number): A unique identifier for each row in the dataset.

Year: The year in which the PM2.5 and meteorological data were recorded (2010–2015).

Month: The month (1–12) in which the data was recorded.Useful for capturing seasonal variations in PM2.5 levels and IWS levels .For example, higher levels in winter due to coal-based heating.

Day: The day of the month when the data was recorded (1–31).

Hour: The hour of the day when the data was recorded (0–23).

Dew Point: Dew point temperature.

Temperature: Atmospheric temperature in Celsius.

Pressure: Atmospheric pressure in hPa.

CBWD (Combined Wind Direction): A categorical variable representing wind direction (e.g., NW, NE, SW, SE).

IWS (Integrated Wind Speed):  The cumulative wind speed in meters per second (m/s) over a specific period.

IS (Integrated Snowfall): The cumulative snowfall in millimeters (mm) over a specific period.

IR (Integrated Rainfall): The cumulative rainfall in millimeters (mm) over a specific period.

PM2.5 (Target Variable): The concentration of fine particulate matter (PM2.5) in micrograms per cubic meter (µg/m³).