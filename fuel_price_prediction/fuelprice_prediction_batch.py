from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

# Files available at the following link
# https://drive.google.com/drive/folders/18W4B_MBD6f4jsWIaPADnEnjxmwmAvh0-?usp=sharing

data_20 = pd.read_csv('large_files/20data.csv')
data_21 = pd.read_csv('large_files/21data.csv')
data_22 = pd.read_csv('large_files/22data.csv')

fuel_types = ['LPG',
              'e10',
              'Unleaded(91)',
              'PULP_95_96_RON',
              'PULP 98 RON',
              'Diesel',
              'Premium Diesel']
cols = ['Price', 'TransactionDateutc']

batch_sizes = [16, 32, 64, 128]
epochs = 1000
results = []


def plot_prices(ax, y_true, y_pred, fuel_type, batch_size, rmse, r2, mape):
    ax.plot(y_true, color='blue', label="Actual Price")
    ax.plot(y_pred, color='red', label=f"Predicted Price (Batch Size: {batch_size})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.set_title(f"{fuel_type} Prices (Batch Size: {batch_size})")
    ax.legend(loc='upper left')
    ax.grid()
    ax.text(0.05, 0.05, f"RMSE: {rmse:.2f}\nR-squared: {r2:.2f}\nMAPE: {mape:.2f}",
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom')


def create_model(batch_size):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(1, 2)))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


for fuel_type in fuel_types:
    results = []
    data_train = pd.concat([data_20, data_21])
    data_train = data_train[data_train['Fuel_Type'] == fuel_type][cols]
    data_test = data_22[data_22['Fuel_Type'] == fuel_type][cols]

    data_train['TransactionDateutc'] = data_train['TransactionDateutc'].apply(
        lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M').timestamp())
    data_test['TransactionDateutc'] = data_test['TransactionDateutc'].apply(
        lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M').timestamp())

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(data_train)
    test_scaled = scaler.transform(data_test)

    fig, axs = plt.subplots(3, 2, figsize=(15, 15), tight_layout=True)

    batch_size_idx = 0
    for batch_size in batch_sizes:
        X_train, y_train = [], []
        for i in range(1, len(train_scaled)):
            X_train.append(train_scaled[i - 1:i])
            y_train.append(train_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[2]))

        # Split training data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Reshape validation data
        X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[2]))

        print(f"Fuel Type: {fuel_type}, Batch Size: {batch_size}")

        model = create_model(batch_size)

        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Pass the validation data to the 'fit' method
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping],
                  validation_data=(X_val, y_val))

        # Save the trained model for each fuel type and batch size
        model.save(f"fuel_price_batch/{fuel_type}_{batch_size}.h5")

        X_test, y_test = [], []
        for i in range(1, len(test_scaled)):
            X_test.append(test_scaled[i - 1:i])
            y_test.append(test_scaled[i, 0])
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[2]))

        y_pred = model.predict(X_test)
        y_pred = scaler.inverse_transform(np.concatenate((y_pred, np.zeros((len(y_pred), 1))), axis=1))[:, 0]
        y_true = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((len(y_test), 1))), axis=1))[
                 :, 0]

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)

        results.append([batch_size, rmse, r2, mape])

        print(f"Result:, RMSE: {rmse:.2f}, R-squared: {r2:.2f}, MAPE: {mape:.2f}")

        plot_prices(axs[batch_size_idx // 2, batch_size_idx % 2], y_true, y_pred, fuel_type, batch_size, rmse, r2, mape)
        batch_size_idx += 1

    results_df = pd.DataFrame(results, columns=['batch_size', 'rmse', 'r2', 'mape'])
    results_df.to_csv(f'fuel_price_batch/{fuel_type}_results.csv', index=False)

    plt.savefig(f'fuel_price_batch/{fuel_type}_comparison.png', bbox_inches='tight')
    plt.clf()
