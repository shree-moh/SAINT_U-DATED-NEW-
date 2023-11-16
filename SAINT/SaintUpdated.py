from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from pretrainmodel import SAINT

# Rest of your code

pd_list = []
pf_list = []
bal_list = []
fir_list = []


def classifier_eval(y_test, y_pred):
    print('y_test:', y_test)
    print('y_pred:', y_pred)

    from sklearn.metrics import confusion_matrix

    # Assuming you have y_test (true labels) and y_pred (predicted labels)
    cm = confusion_matrix(y_test, y_pred)

    # Calculate evaluation metrics
    PD, PF, balance, FIR = classifier_eval(y_test, y_pred)

    # Append the results to your lists
    pd_list.append(PD)
    pf_list.append(PF)
    bal_list.append(balance)
    fir_list.append(FIR)

    assert isinstance(cm, object)
    print('혼동행렬 : ', cm)

    PD = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    print('PD : ', PD)

    PF = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    print('PF : ', PF)

    balance = 1 - (((0 - PF) * (0 - PF) + (1 - PD) * (1 - PD)) / 2)
    print('balance : ', balance)

    FI = (cm[1, 1] + cm[0, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
    FIR = (PD - FI) / PD
    print('FIR : ', FIR)

    return PD, PF, balance, FIR



# CSV 파일 경로를 지정
csv_file_path = "EQ.csv"

# CSV 파일을 데이터프레임으로 읽어오기
df = pd.read_csv(csv_file_path)

# 데이터프레임에서 특징(X)과 목표 변수(y) 추출
X = df.drop(columns=['class'])
y = df['class']  # 'target' 열을 목표 변수로 사용

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# K-겹 교차 검증을 설정합니다
k = 10  # K 값 (원하는 폴드 수) 설정
kf = KFold(n_splits=k, shuffle=True, random_state=42)

scaler = MinMaxScaler()
X_test_Nomalized = scaler.fit_transform(X_test)

# K-겹 교차 검증 수행
for train_index, val_index in kf.split(X_train):
    X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

    # 전처리
    # Min-Max 정규화 수행(o)
    X_fold_train_normalized = scaler.fit_transform(X_fold_train)
    X_fold_val_normalized = scaler.transform(X_fold_val)

    # SMOTE를 사용하여 학습 데이터 오버샘플링
    smote = SMOTE(random_state=42)
    X_fold_train_resampled, y_fold_train_resampled = smote.fit_resample(X_fold_train_normalized, y_fold_train)

    # Define cat_dims before using it
    cat_dims = []  # Initialize as an empty list

    for train_index, val_index in kf.split(X_train):
        X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

        # ...

        # Append values to cat_dims
        cat_dims = np.append(np.array(cat_dims), np.array([2])).astype(int)

        # ...

    cat_dims = np.append(np.array(cat_dims), np.array([2])).astype(
        int)  # unique values in cat column, with 2 appended in the end as the number of unique values of y. This is the case of binary classification

    # model = SAINT(
    #     num_continuous = [len(con_idxs), dim] =8,
    #
    #     dim_out = 1,
    #     depth=1,
    #     heads=4,
    #     attn_dropout=0
    #     ff_dropout=0.8,
    #     mlp_hidden_mults=(4, 2),
    #
    #     cont_embeddings='MLP',
    #     scalingfactor=10,
    #     attentiontype='col',
    #     final_mlp_style= 'common',
    #     y_dim=2
    #
    # )

    # Define and assign a value to con_idxs before using it
    con_idxs = [0, 1, 3, 5]  # Replace with the appropriate list of continuous feature indices

    pretrainmodel = SAINT(
        num_continuous=5,
        num_categories=10,
        dim=8,
        dim_out=1,
        depth=1,
        heads=4,
        attn_dropout=0,
        ff_dropout=0.8,
        mlp_hidden_mults=(4, 2),
        cont_embeddings='MLP',
        scalingfactor=10,
        attentiontype='col',
        final_mlp_style='common',
        y_dim=2,
        categories=10  # Make sure to replace this with the correct argument
    )
    # Assuming you are using a classifier model
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier  # You can replace this with your specific model import

    # ... Your data loading and preprocessing code ...

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train your model
    model = RandomForestClassifier()  # Replace with your specific model instantiation
    model.fit(X_train, y_train)

    # Get predictions
    y_pred = model.predict(X_test)
    # After getting predictions
print('Predictions:', y_pred)
from sklearn.metrics import confusion_matrix

# Ensure y_test and y_pred are lists containing labels or predictions
y_test = [1, 0, 1, 1, 0]  # Replace this with your actual data
y_pred = [1, 1, 1, 0, 0]  # Replace this with your actual data

# Use confusion_matrix with the correct inputs
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Before calculating metrics, check the lengths of y_test and y_pred
print('Length of y_test:', len(y_test))
print('Length of y_pred:', len(y_pred))

# Add print statements to check other relevant variables

# Calculate the average metrics across all folds
# Initialize pd_list,pf_list,bal_list,for_list before its usage
pd_list = []  # Initialize an empty list or with default values
pf_list = []
bal_list = []
fir_list = []

# Further in the code, populate or manipulate pd_list
# ...
# For instance, appending values to pd_list,pf_list,bal_list,for_list
pd_list.append(5)
pd_list.append(10)
pf_list.append(10)
pf_list.append(10)
bal_list.append(5)
bal_list.append(10)
bal_list.append(10)
fir_list.append(5)
fir_list.append(10)
# Finally, using pd_list
avg_PD = sum(pd_list) / len(pd_list) if len(pd_list) > 0 else 'No values in pd_list'
avg_PF = sum(pf_list) / len(pf_list) if len(pf_list) > 0 else 'No values in pf_list'
avg_balance = sum(bal_list) / len(bal_list) if len(bal_list) > 0 else 'No values in bal_list'
avg_FIR = sum(fir_list) / len(fir_list) if len(fir_list) > 0 else 'No values in fir_list'

# Print or use the average metrics as needed
print('avg_PD:', avg_PD)
print('avg_PF:', avg_PF)
print('avg_balance:', avg_balance)
print('avg_FIR:', avg_FIR)