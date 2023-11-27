import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/eth/hotel/pixel_pos.csv')

plt.figure(figsize=(10, 6))

plt.plot(df[1][df['pedestrianId'] == 1],
         df[2][df['pedestrianId'] == 1],
         marker='o', linestyle='-')

# Add labels and legend
plt.title('Plot of DataFrame Rows')
plt.xlabel('Column')
plt.ylabel('Values')
plt.legend()

# Show the plot
plt.show()
