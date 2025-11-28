import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {'orders': [10, 20, 30, 40, 50],
        'shipments': [15, 25, 35, 45, 55]}
df = pd.DataFrame(data)

# Linear Regression Model
class RegressionAgent:
    def process(self):
        X = df[['orders']]
        y = df['shipments']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)
        print(predictions)

# Execute the process
agent = RegressionAgent()
agent.process()
