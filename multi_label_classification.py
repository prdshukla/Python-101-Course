import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import GradientBoostingClassifier

train_data = pd.read_csv("/kaggle/input/playground-series-s3e18/train.csv")
train_data.drop(['EC3', 'EC4', 'EC5', 'EC6'], axis=1, inplace=True)
test_data = pd.read_csv("/kaggle/input/playground-series-s3e18/test.csv")

X = train_data.drop(['id', 'EC1', 'EC2'], axis=1)
X_test = test_data.drop(['id'], axis=1)

y_EC1 = train_data.EC1
y_EC2 = train_data.EC2

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

X_EC1, y_EC1 = RandomUnderSampler().fit_resample(X, y_EC1)
X_EC2, y_EC2 = RandomUnderSampler().fit_resample(X, y_EC2)

model_EC1 = GradientBoostingClassifier(random_state=0)
model_EC2 = GradientBoostingClassifier(random_state=0)

model_EC1.fit(X_EC1, y_EC1)
model_EC2.fit(X_EC2, y_EC2)

preds_EC1 = model_EC1.predict_proba(X_test)[:, 1]
preds_EC2 = model_EC2.predict_proba(X_test)[:, 1]

pd.DataFrame({'id': test_data.id, 'EC1': preds_EC1, 'EC2': preds_EC2}).to_csv('submission.csv', index=False)