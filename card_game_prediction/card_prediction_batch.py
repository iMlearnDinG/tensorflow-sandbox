import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


class CustomKerasClassifier(KerasClassifier):
    def __init__(self, *args, **kwargs):
        if 'batch_size' in kwargs:
            self.batch_size = kwargs.pop('batch_size')
        else:
            self.batch_size = None
        super().__init__(*args, **kwargs)

    def fit(self, x, y, **kwargs):
        if self.batch_size is not None:
            kwargs['batch_size'] = self.batch_size
        return super().fit(x, y, **kwargs)



# Data available at the following link:
# https://drive.google.com/drive/folders/1MhSuFrvd46xdEfSPeSVpzuAWINjwBH7v?usp=sharing

# Load the data
df = pd.read_csv('game_results.csv')

# Define a mapping dictionary for card values and suits
card_value_map = {str(i): i for i in range(1, 14)}
suit_value_map = {'Spades': 4, 'Hearts': 3, 'Diamonds': 2, 'Clubs': 1}

# Apply the mapping to the relevant columns
card_cols = [f'P1_Card_{i}' for i in range(1, 6)] + \
            [f'P2_Card_{i}' for i in range(1, 6)] + \
            [f'DLR_Card_{i}' for i in range(1, 6)]

for col in card_cols:
    df[col + '_Rank'] = df[col].apply(lambda x: card_value_map[x.split()[0]])
    df[col + '_Suit'] = df[col].apply(lambda x: suit_value_map[x.split()[1]])

# Drop original card columns
df = df.drop(columns=card_cols)

# Map the target variable to numerical values
target_map = {'player1': 0, 'player2': 1, 'tie': 2}
for col in ['Winner_Column_1', 'Winner_Column_2', 'Winner_Column_3', 'Winner_Column_4', 'Winner_Column_5']:
    df[col] = df[col].map(target_map)

# Remove rows with NaN values
df = df.dropna()

# Split the data into features and target
X = df.drop(
    columns=['Winner_Column_1', 'Winner_Column_2', 'Winner_Column_3', 'Winner_Column_4', 'Winner_Column_5']).values
y = df[['Winner_Column_1', 'Winner_Column_2', 'Winner_Column_3', 'Winner_Column_4', 'Winner_Column_5']].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the input shape and number of classes
input_shape = X_train.shape[1:]
num_classes = len(np.unique(y))


class PrintParams(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        params = self.model.optimizer.get_config()
        print(f"Starting training; got optimizer parameters: {params}")


def create_model(neurons=1, optimizer='adam', learning_rate=0.001):
    print(f"Creating model with {neurons} neurons")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(neurons, input_shape=input_shape, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(5, activation='sigmoid')  # 5 output neurons for 5 columns
    ])

    # Set the optimizer with the specified learning rate
    optimizer = optimizer.lower()
    if optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',  # use binary_crossentropy for multi-label problems
                  metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model, epochs=1000, verbose=1,
                        callbacks=[EarlyStopping(monitor='accuracy', patience=3, mode='max'), PrintParams()])

param_grid = {
    'neurons': [256],
    'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
    'batch_size': [256]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, n_jobs=1, pre_dispatch='2*n_jobs')

# Fit the GridSearchCV object to the data
grid_result = grid.fit(X_train, y_train)

# Print the best parameters
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Evaluate the best model on the test data
best_model = grid_result.best_estimator_.model
loss, accuracy = best_model.evaluate(X_test, y_test)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

# Save the best model
best_model.save('best_model.h5')

# Predict on the test data for this fold
y_pred = best_model.predict(X_test)

# Round the predictions to 0 or 1 for each of the five columns
y_pred_rounded = np.round(y_pred)

# Convert actual and predicted labels to DataFrames for easier manipulation
actual_df = pd.DataFrame(y_test, columns=['Actual_Column_1', 'Actual_Column_2', 'Actual_Column_3', 'Actual_Column_4',
                                          'Actual_Column_5'])
pred_df = pd.DataFrame(y_pred_rounded,
                       columns=['Pred_Column_1', 'Pred_Column_2', 'Pred_Column_3', 'Pred_Column_4', 'Pred_Column_5'])

# Concatenate actual and predicted DataFrames
results_df = pd.concat([pred_df, actual_df], axis=1)

# Save to a CSV file
results_df.to_csv('predictions.csv', index=False)

# Get the grid search results
cv_results_df = pd.DataFrame(grid_result.cv_results_)

# Add accuracy and loss metrics to the DataFrame
cv_results_df['mean_accuracy'] = grid_result.cv_results_['mean_test_score']
cv_results_df['mean_loss'] = -grid_result.cv_results_['mean_test_score']

# Save the grid search results to a CSV file
cv_results_df.to_csv('grid_search_results.csv', index=False)
