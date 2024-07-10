# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 23:41:30 2024

@author: LENOVO
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Concatenate, Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


file = 'C:/Users/LENOVO/OneDrive/Desktop/Customer life time value prediction/online_retail_II.xlsx'
data = pd.read_excel(file)


print(data.head())


data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
data['transaction_month'] = data['InvoiceDate'].dt.to_period('M')
data_grouped = data.groupby(['Customer ID', 'transaction_month']).agg({
    'Quantity': 'sum', 
    'Price': 'mean'
}).reset_index()


data_grouped['transaction_amount'] = data_grouped['Quantity'] * data_grouped['Price']


transaction_pivot = data_grouped.pivot(index='Customer ID', columns='transaction_month', values='transaction_amount').fillna(0)
transaction_data = transaction_pivot.values

np.random.seed(42)
customer_ids = transaction_pivot.index
num_customers = len(customer_ids)
demographic_data = pd.DataFrame({
    'Customer ID': customer_ids,
    'Age': np.random.randint(18, 70, size=num_customers),
    'Gender': np.random.choice(['M', 'F'], size=num_customers),
    'Income': np.random.randint(20000, 100000, size=num_customers)
})
demographic_data['Gender'] = demographic_data['Gender'].map({'M': 0, 'F': 1})
demographic_data.set_index('Customer ID', inplace=True)
demographic_data = demographic_data.loc[customer_ids].values


scaler_transaction = StandardScaler()
scaler_demographics = StandardScaler()

transaction_data_scaled = scaler_transaction.fit_transform(transaction_data)
demographic_data_scaled = scaler_demographics.fit_transform(demographic_data)


clv = np.sum(transaction_data_scaled, axis=1)


X_train_trans, X_test_trans, X_train_demo, X_test_demo, y_train, y_test = train_test_split(
    transaction_data_scaled, demographic_data_scaled, clv, test_size=0.2, random_state=42)


X_train_trans = X_train_trans[..., np.newaxis]
X_test_trans = X_test_trans[..., np.newaxis]


transaction_input = Input(shape=(X_train_trans.shape[1], 1))
x = Conv1D(filters=32, kernel_size=3, activation='relu')(transaction_input)
x = Flatten()(x)

demographic_input = Input(shape=(X_train_demo.shape[1],))
y = Dense(16, activation='relu')(demographic_input)

combined = Concatenate()([x, y])

z = Dense(64, activation='relu')(combined)
z = Dense(1, activation='linear')(z)

model = Model(inputs=[transaction_input, demographic_input], outputs=z)
model.compile(optimizer='adam', loss='mean_squared_error')


model.summary()


history = model.fit([X_train_trans, X_train_demo], y_train, epochs=50, validation_split=0.2, batch_size=32)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


loss = model.evaluate([X_test_trans, X_test_demo], y_test)
print(f'Test Loss: {loss}')


y_pred = model.predict([X_test_trans, X_test_demo])


for i in range(5):
    print(f'Predicted CLV: {y_pred[i][0]}, True CLV: {y_test[i]}')
