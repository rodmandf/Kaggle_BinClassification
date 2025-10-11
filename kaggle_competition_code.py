from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline
import catboost

!gdown 1ERwQ5odiK1Zvi1LtjpkzCMUswYsAX8_K  # train.csv
!gdown 1fGw_-RFwvn_LEdt91Jq-7A-wzG6mmH8r  # test.csv
!gdown 199Mt4OYZNaelT83U-HGDsEYs2YcUGQ6y  # submission.csv

data = pd.read_csv('./train.csv')

num_cols = [
    'ClientPeriod',
    'MonthlySpending',
    'TotalSpent'
]

cat_cols = [
    'Sex',
    'IsSeniorCitizen',
    'HasPartner',
    'HasChild',
    'HasPhoneService',
    'HasMultiplePhoneNumbers',
    'HasInternetService',
    'HasOnlineSecurityService',
    'HasOnlineBackup',
    'HasDeviceProtection',
    'HasTechSupportAccess',
    'HasOnlineTV',
    'HasMovieSubscription',
    'HasContractPhone',
    'IsBillingPaperless',
    'PaymentMethod'
]

feature_cols = num_cols + cat_cols
target_col = 'Churn'

data.describe().T

data.head(10)

nan_counts = data.isna().sum()

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=[nan_counts.values],
                rowLabels=['Количество NaN'],
                colLabels=nan_counts.index,
                cellLoc='center',
                loc='center')

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

plt.title('Количество NaN значений по столбцам')
plt.show()

empty_string_counts = data.applymap(lambda x: isinstance(x, str) and x.strip() == '').sum()

print("Количество пустых строк:")
print(empty_string_counts)

mask = data['TotalSpent'].apply(lambda x: isinstance(x, str) and x.strip() == '')
data.loc[mask, 'TotalSpent'] = 0

data['TotalSpent'] = pd.to_numeric(data['TotalSpent'])

q_string_counts = data.applymap(lambda x: isinstance(x, str) and x.strip() == '?').sum()

print("Количество пустых строк:")
print(q_string_counts)

data_origin = data.copy()

value_counts = data['ClientPeriod'].value_counts()

plt.figure(figsize=(10, 6))
plt.bar(value_counts.index, value_counts.values, color='skyblue', alpha=0.7)
plt.title('Распределение Периода клиентов')
plt.xlabel('Период')
plt.ylabel('Количество клиентов')
plt.grid(axis='y', alpha=0.3)

plt.show()

data['Sex'] = data['Sex'].replace({'Male':1, 'Female':0})

n_cols = 2
n_rows = (len(cat_cols) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes = axes.flatten() if n_rows > 1 else [axes]

for i, col in enumerate(cat_cols):
    if i < len(axes):
        value_counts = data[col].value_counts()

        bars = axes[i].bar(value_counts.index, value_counts.values,
                          color=plt.cm.Set3(np.linspace(0, 1, len(value_counts))))

        axes[i].set_title(f'Распределение: {col}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Количество')

        if len(value_counts) > 5:
            axes[i].tick_params(axis='x', rotation=45)

        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

value_counts = data[target_col].value_counts()

# Строим график
plt.figure(figsize=(10, 6))
plt.bar(value_counts.index, value_counts.values, color='skyblue', alpha=0.7)
plt.title('Распределение признака: Churn')
plt.xlabel('Категории')
plt.ylabel('Количество')
plt.grid(axis='y', alpha=0.3)
plt.show()

to_num_columns = ['HasPartner', 'HasChild', 'HasPhoneService', 'IsBillingPaperless']

for col in to_num_columns:
    if col in data.columns:
        data[col] = data[col].replace({'Yes': 1, 'No': 0})

data.head()

data.info()

new_categ_cols = ["HasMultiplePhoneNumbers", "HasInternetService", "HasOnlineSecurityService", "HasOnlineBackup", "HasDeviceProtection", "HasTechSupportAccess", "HasOnlineTV", "HasMovieSubscription", "HasContractPhone", "PaymentMethod"]


ohe = OneHotEncoder(sparse_output=False, drop='first')

encoded_array = ohe.fit_transform(data[new_categ_cols])
encoded_columns = ohe.get_feature_names_out(new_categ_cols)
data_encoded = pd.DataFrame(encoded_array, columns=encoded_columns, index=data.index)

numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns[:-1].tolist()
data_final = pd.concat([data[numerical_columns], data_encoded], axis=1)

data_final = pd.concat([data_final, data[target_col]], axis=1)
data_final.info()


target_correlations = data_final.corr()[['Churn']].sort_values('Churn', ascending=False)

print("Корреляции с таргетной переменной:")
print(target_correlations)

plt.figure(figsize=(8, 10))
sns.heatmap(target_correlations, annot=True, cmap='RdYlBu', center=0, fmt='.2f')
plt.title('Корреляции признаков с таргетной переменной')
plt.tight_layout()
plt.show()

#Переходим к обучению модели
data_final.describe().T

cols_to_scale = ["ClientPeriod", "MonthlySpending", "TotalSpent"]

scaler = StandardScaler()

data_scaled = data_final.copy()
data_scaled[cols_to_scale] = scaler.fit_transform(data_final[cols_to_scale])

X_train, X_val, y_train, y_val = train_test_split(data_scaled.drop('Churn',axis=1), data_scaled['Churn'], test_size=0.2, random_state=42)

Cs = [100, 10, 1, 0.1, 0.01, 0.001]
model = LogisticRegressionCV(
    Cs=Cs,
    cv=5,
    scoring='roc_auc',
    random_state=42,
    max_iter=1000,
    refit=True,
    class_weight='balanced',
    penalty='l2',
)

model.fit(X_train, y_train)

print(f"Лучший параметр C: {model.C_[0]}")
print(f"Все tested C: {model.Cs_}")
print(f"Средние ROC-AUC для каждого C: {model.scores_[1].mean(axis=0)}")

best_log_model = LogisticRegressionCV(
    Cs=[10],
    cv=5,
    scoring='roc_auc',
    random_state=42,
    max_iter=1000,
    refit=True,
    class_weight='balanced',
    penalty='l2',
)

best_log_model.fit(X_train, y_train)
print(f"Средние ROC-AUC: {best_log_model.scores_[1]}")

#Теперь Catboost
X_train_origin, X_val_origin, y_train_origin, y_val_origin = train_test_split(data_origin.drop("Churn", axis=1), data_origin[target_col],
                                                       train_size=0.8,
                                                       random_state=42)



boosting_model = catboost.CatBoostClassifier(n_estimators=200,
                                             cat_features=cat_cols,
                                             eval_metric='AUC')

boosting_model.fit(X_train_origin, y_train_origin)

y_val_predicted = boosting_model.predict_proba(X_val_origin)[:, 1]

val_auc = roc_auc_score(y_val_origin, y_val_predicted)
print(val_auc)

boosting_model = catboost.CatBoostClassifier(n_estimators=300,
                                             depth=4,
                                             learning_rate = 0.03,
                                             cat_features=cat_cols,
                                             eval_metric='AUC')
boosting_model.fit(X_train_origin, y_train_origin)

y_val_predicted = boosting_model.predict_proba(X_val_origin)[:, 1]
val_auc = roc_auc_score(y_val_origin, y_val_predicted)
print(val_auc)

X_test = pd.read_csv('./test.csv')
X_test.info()

empty_string_counts_test = X_test.applymap(lambda x: isinstance(x, str) and x.strip() == '').sum()

print("Количество пустых строк:")
print(empty_string_counts_test)

mask_test = X_test['TotalSpent'].apply(lambda x: isinstance(x, str) and x.strip() == '')
X_test.loc[mask_test, 'TotalSpent'] = 0
X_test['TotalSpent'] = pd.to_numeric(X_test['TotalSpent'])

X_test.info()

X_new_test = X_test.copy()

X_new_test['Sex'] = X_new_test['Sex'].replace({'Male':1, 'Female':0})

for col in to_num_columns:
    if col in X_new_test.columns:
        X_new_test[col] = X_new_test[col].replace({'Yes': 1, 'No': 0})

encoded_array_test = ohe.transform(X_new_test[new_categ_cols])
data_encoded_test = pd.DataFrame(encoded_array_test, columns=encoded_columns, index=X_new_test.index)

test_numerical_columns = X_new_test.select_dtypes(include=['int64', 'float64']).columns.tolist()
data_final_test = pd.concat([X_new_test[test_numerical_columns], data_encoded_test], axis=1)

data_test_scaled = data_final_test.copy()

data_test_scaled[cols_to_scale] = scaler.transform(data_final_test[cols_to_scale])

best_catboost_model = boosting_model

submission = pd.read_csv('./submission.csv')

submission['Churn'] = best_catboost_model.predict_proba(X_test)[:, 1]
submission.to_csv('./my_submission.csv', index=False)

best_logistic_model = best_log_model

submission['Churn'] = best_logistic_model.predict_proba(data_test_scaled)[:, 1]
submission.to_csv('./my_submission2.csv', index=False)