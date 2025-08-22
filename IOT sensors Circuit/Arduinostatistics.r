# Load libraries
library(ggplot2)
library(plotly)

df <- read.csv("co2data_clean.csv")


colnames(df) <- c("Time", "RawValue", "Voltage", "CO2")

p1 <- ggplot(df, aes(x=Time, y=CO2)) +
  geom_line(color="blue") +
  geom_point() +
  labs(title="CO2 Concentration over Time", x="Time (s)", y="CO2 (ppm)") +
  theme_minimal()


p2 <- ggplot(df, aes(x=Time, y=Voltage)) +
  geom_line(color="orange") +
  geom_point() +
  labs(title="Voltage over Time", x="Time (s)", y="Voltage (V)") +
  theme_minimal()

p3 <- ggplot(df, aes(x=RawValue, y=CO2)) +
  geom_point(color="red") +
  labs(title="Raw Value vs CO2", x="Raw Sensor Value", y="CO2 (ppm)") +
  theme_minimal()

ggplotly(p1)
ggplotly(p2)
ggplotly(p3)


