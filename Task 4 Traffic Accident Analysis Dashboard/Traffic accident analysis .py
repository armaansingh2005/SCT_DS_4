# ============================================================
# TRAFFIC ACCIDENT DATA ANALYSIS
# ============================================================
# Analyze relationships between:
# 1. Road conditions
# 2. Weather conditions
# 3. Time of day
# 4. Accident severity
# 5. Accident hotspots
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. CREATE SAMPLE TRAFFIC ACCIDENT DATASET
# ------------------------------------------------------------

data = {
    "Accident_ID": range(1, 31),

    "Latitude": [
        28.4595, 28.4595, 28.4700, 28.4800, 28.4700,
        28.4595, 28.4900, 28.5000, 28.4700, 28.4800,
        28.4595, 28.4900, 28.5000, 28.4700, 28.4800,
        28.4595, 28.4900, 28.5000, 28.4700, 28.4800,
        28.4595, 28.4900, 28.5000, 28.4700, 28.4800,
        28.4595, 28.4900, 28.5000, 28.4700, 28.4800
    ],

    "Longitude": [
        77.0266, 77.0266, 77.0300, 77.0400, 77.0300,
        77.0266, 77.0500, 77.0600, 77.0300, 77.0400,
        77.0266, 77.0500, 77.0600, 77.0300, 77.0400,
        77.0266, 77.0500, 77.0600, 77.0300, 77.0400,
        77.0266, 77.0500, 77.0600, 77.0300, 77.0400,
        77.0266, 77.0500, 77.0600, 77.0300, 77.0400
    ],

    "Weather": [
        "Clear", "Rain", "Fog", "Clear", "Rain",
        "Clear", "Fog", "Rain", "Clear", "Rain",
        "Clear", "Rain", "Fog", "Clear", "Rain",
        "Clear", "Fog", "Rain", "Clear", "Rain",
        "Clear", "Rain", "Fog", "Clear", "Rain",
        "Clear", "Fog", "Rain", "Clear", "Rain"
    ],

    "Road_Condition": [
        "Dry", "Wet", "Wet", "Dry", "Wet",
        "Dry", "Wet", "Wet", "Dry", "Wet",
        "Dry", "Wet", "Wet", "Dry", "Wet",
        "Dry", "Wet", "Wet", "Dry", "Wet",
        "Dry", "Wet", "Wet", "Dry", "Wet",
        "Dry", "Wet", "Wet", "Dry", "Wet"
    ],

    "Time": [
        "08:30", "09:15", "07:45", "13:20", "18:30",
        "20:15", "07:30", "18:45", "12:10", "19:20",
        "08:10", "17:50", "06:45", "14:30", "21:15",
        "09:30", "07:20", "18:10", "13:45", "20:30",
        "08:45", "17:30", "06:30", "15:20", "19:45",
        "10:15", "07:10", "18:20", "12:30", "21:00"
    ],

    "Severity": [
        "Minor", "Major", "Major", "Minor", "Major",
        "Minor", "Major", "Major", "Minor", "Major",
        "Minor", "Major", "Major", "Minor", "Major",
        "Minor", "Major", "Major", "Minor", "Major",
        "Minor", "Major", "Major", "Minor", "Major",
        "Minor", "Major", "Major", "Minor", "Major"
    ]
}

df = pd.DataFrame(data)

print("Traffic Accident Dataset")
print(df.head())

# ------------------------------------------------------------
# 2. BASIC DATASET INFORMATION
# ------------------------------------------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

# ------------------------------------------------------------
# 3. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicate records
df = df.drop_duplicates()

# ------------------------------------------------------------
# 4. CONVERT TIME INTO DATETIME
# ------------------------------------------------------------

df["Time"] = pd.to_datetime(
    df["Time"],
    format="%H:%M"
)

# Extract hour
df["Hour"] = df["Time"].dt.hour

# ------------------------------------------------------------
# 5. CREATE TIME-OF-DAY CATEGORY
# ------------------------------------------------------------

def get_time_period(hour):

    if 5 <= hour < 12:
        return "Morning"

    elif 12 <= hour < 17:
        return "Afternoon"

    elif 17 <= hour < 21:
        return "Evening"

    else:
        return "Night"


df["Time_Period"] = df["Hour"].apply(get_time_period)

print("\nTime Period Distribution:")
print(df["Time_Period"].value_counts())

# ------------------------------------------------------------
# 6. ACCIDENTS BY WEATHER CONDITION
# ------------------------------------------------------------

weather_count = df["Weather"].value_counts()

print("\nAccidents by Weather:")
print(weather_count)

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Weather"
)

plt.title("Accidents by Weather Condition")
plt.xlabel("Weather Condition")
plt.ylabel("Number of Accidents")

plt.show()

# ------------------------------------------------------------
# 7. ACCIDENTS BY ROAD CONDITION
# ------------------------------------------------------------

road_count = df["Road_Condition"].value_counts()

print("\nAccidents by Road Condition:")
print(road_count)

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Road_Condition"
)

plt.title("Accidents by Road Condition")
plt.xlabel("Road Condition")
plt.ylabel("Number of Accidents")

plt.show()

# ------------------------------------------------------------
# 8. ACCIDENTS BY TIME OF DAY
# ------------------------------------------------------------

time_count = df["Time_Period"].value_counts()

print("\nAccidents by Time of Day:")
print(time_count)

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Time_Period",
    order=["Morning", "Afternoon", "Evening", "Night"]
)

plt.title("Accidents by Time of Day")
plt.xlabel("Time of Day")
plt.ylabel("Number of Accidents")

plt.show()

# ------------------------------------------------------------
# 9. ACCIDENT SEVERITY
# ------------------------------------------------------------

severity_count = df["Severity"].value_counts()

print("\nAccident Severity:")
print(severity_count)

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Severity"
)

plt.title("Accident Severity Distribution")
plt.xlabel("Severity")
plt.ylabel("Number of Accidents")

plt.show()

# ------------------------------------------------------------
# 10. WEATHER VS SEVERITY
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="Weather",
    hue="Severity"
)

plt.title("Weather Conditions vs Accident Severity")
plt.xlabel("Weather Condition")
plt.ylabel("Number of Accidents")

plt.show()

# ------------------------------------------------------------
# 11. ROAD CONDITION VS SEVERITY
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Road_Condition",
    hue="Severity"
)

plt.title("Road Condition vs Accident Severity")
plt.xlabel("Road Condition")
plt.ylabel("Number of Accidents")

plt.show()

# ------------------------------------------------------------
# 12. TIME OF DAY VS SEVERITY
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="Time_Period",
    hue="Severity",
    order=["Morning", "Afternoon", "Evening", "Night"]
)

plt.title("Time of Day vs Accident Severity")
plt.xlabel("Time of Day")
plt.ylabel("Number of Accidents")

plt.show()

# ------------------------------------------------------------
# 13. ACCIDENTS BY HOUR
# ------------------------------------------------------------

hourly_accidents = df["Hour"].value_counts().sort_index()

print("\nAccidents by Hour:")
print(hourly_accidents)

plt.figure(figsize=(10, 5))

plt.plot(
    hourly_accidents.index,
    hourly_accidents.values,
    marker="o"
)

plt.title("Accidents by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Number of Accidents")
plt.xticks(range(0, 24))

plt.grid(True)

plt.show()

# ------------------------------------------------------------
# 14. ACCIDENT HOTSPOT ANALYSIS
# ------------------------------------------------------------

# Group accidents by latitude and longitude
hotspots = (
    df.groupby(["Latitude", "Longitude"])
    .size()
    .reset_index(name="Accident_Count")
)

hotspots = hotspots.sort_values(
    by="Accident_Count",
    ascending=False
)

print("\nTop Accident Hotspots:")
print(hotspots.head(10))

# ------------------------------------------------------------
# 15. VISUALIZE ACCIDENT HOTSPOTS
# ------------------------------------------------------------

plt.figure(figsize=(9, 7))

scatter = plt.scatter(
    df["Longitude"],
    df["Latitude"],
    c=df["Severity"].map({
        "Minor": 1,
        "Major": 2
    }),
    s=100,
    alpha=0.7
)

plt.title("Traffic Accident Hotspots")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.grid(True)

plt.show()

# ------------------------------------------------------------
# 16. CORRELATION BETWEEN NUMERICAL VARIABLES
# ------------------------------------------------------------

numeric_columns = [
    "Latitude",
    "Longitude",
    "Hour"
]

correlation = df[numeric_columns].corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(7, 5))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()

# ------------------------------------------------------------
# 17. FIND MOST COMMON ACCIDENT FACTORS
# ------------------------------------------------------------

print("\n==========================================")
print("MOST COMMON ACCIDENT FACTORS")
print("==========================================")

print(
    "Most common weather condition:",
    df["Weather"].mode()[0]
)

print(
    "Most common road condition:",
    df["Road_Condition"].mode()[0]
)

print(
    "Most common time period:",
    df["Time_Period"].mode()[0]
)

print(
    "Most common accident severity:",
    df["Severity"].mode()[0]
)

# ------------------------------------------------------------
# 18. FINAL SUMMARY
# ------------------------------------------------------------

print("\n==========================================")
print("TRAFFIC ACCIDENT ANALYSIS COMPLETED")
print("==========================================")

print("Total accidents:", len(df))

print("\nWeather Distribution:")
print(weather_count)

print("\nRoad Condition Distribution:")
print(road_count)

print("\nTime Period Distribution:")
print(time_count)

print("\nSeverity Distribution:")
print(severity_count)

print("\nTop Accident Hotspots:")
print(hotspots.head(5))