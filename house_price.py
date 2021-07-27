############################################## Importing Libraries #####################################################

import pandas as pd
from helpers.eda import *
from helpers.data_prep import *
from datetime import datetime
from sklearn.neighbors import LocalOutlierFactor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.max_rows', None)

############################################## Importing Datasets ######################################################
train = pd.read_csv("house_prices/train.csv")
test = pd.read_csv("house_prices/test.csv")
df = train.append(test).reset_index(drop=True)

df.columns = [col.upper() for col in df.columns]


######################################## First Insight About Dataset ###################################################
cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in cat_cols:
    cat_summary(df, col)
for col in cat_but_car:
    cat_summary(df, col)

for col in num_cols:
    num_summary(df, col, plot=True)



######################################## Data Preprocessing ############################################################
del_list = ["ALLEY","UTILITIES","POOLQC","FENCE","MISCFEATURE"]
df.drop(del_list, axis=1, inplace=True)

df = df.apply(lambda x: x.fillna("No") if x.dtype == "O" else x, axis=0)

der_list = ["EXTERQUAL","EXTERCOND","BSMTQUAL","BSMTCOND","HEATINGQC","KITCHENQUAL","FIREPLACEQU","GARAGEQUAL","GARAGECOND"]
ext_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2 , 'Po':1, 'No':0}
df[der_list] = df[der_list].replace(ext_map).astype('int')


ext_map = {'Reg': 4, 'IR1': 3, 'IR2': 2, 'IR3': 1, 'No': 0}
df['LOTSHAPE'] = df['LOTSHAPE'].map(ext_map).astype('int')


df.loc[df['CONDITION2']=="Norm",'CONDITION2'] = 1
df.loc[ (df['CONDITION2'] !=1),'CONDITION2'] = 0

df.loc[df['ROOFMATL']=="CompShg",'ROOFMATL'] = 1
df.loc[ (df['ROOFMATL'] !=1),'ROOFMATL'] = 0

df.loc[df['BSMTFINTYPE1']=="Unf",'BSMTFINTYPE1'] = 1
df.loc[ (df['BSMTFINTYPE1'] == "No"),'BSMTFINTYPE1'] = 0
df.loc[ ((df['BSMTFINTYPE1'] != 1) & (df['BSMTFINTYPE1'] != 0)),'BSMTFINTYPE1'] = 2



df.loc[df['BSMTFINTYPE2']=="Unf",'BSMTFINTYPE2'] = 1
df.loc[ (df['BSMTFINTYPE2'] == "No"),'BSMTFINTYPE2'] = 0
df.loc[ ((df['BSMTFINTYPE2'] != 1) & (df['BSMTFINTYPE2'] != 0)),'BSMTFINTYPE2'] = 2


df.loc[df['HEATING']=="GasA",'HEATING'] = 1
df.loc[ (df['HEATING'] != 1),'HEATING'] = 0

df.loc[df['ELECTRICAL']=="FuseA",'ELECTRICAL'] = "Fuse_Mix"
df.loc[df['ELECTRICAL']=="FuseF",'ELECTRICAL'] = "Fuse_Mix"
df.loc[df['ELECTRICAL']=="FuseP",'ELECTRICAL'] = "Fuse_Mix"
df.loc[df['ELECTRICAL']=="Mix",'ELECTRICAL'] = "Fuse_Mix"


df.loc[df['FUNCTIONAL']=="Typ",'FUNCTIONAL'] = 1
df.loc[ (df['FUNCTIONAL'] !=1),'FUNCTIONAL'] = 0


ext_map = {'RFn': 2, 'Fin': 2, 'Unf': 1, 'No': 0}
df['GARAGEFINISH'] = df['GARAGEFINISH'].map(ext_map).astype('int')



ext_map = {'N': 1, 'Y': 1, 'P': 0}
df['PAVEDDRIVE'] = df['PAVEDDRIVE'].map(ext_map).astype('int')


df.loc[df['STREET']=="Pave",'STREET'] = 1
df.loc[ (df['STREET'] !=1),'STREET'] = 0

df.loc[df['CENTRALAIR']=="Y",'CENTRALAIR'] = 1
df.loc[ (df['CENTRALAIR'] !=1),'CENTRALAIR'] = 0

df.loc[( df["YRSOLD"] - df['YEARBUILT'] <= 10), 'NEW_BUILDING_AGE_CAT'] = 'new'
df.loc[( df["YRSOLD"] - df['YEARBUILT'] > 10) & ( df["YRSOLD"] - df['YEARBUILT'] <= 25), 'NEW_BUILDING_AGE_CAT'] = 'relatively_new'
df.loc[( df["YRSOLD"] - df['YEARBUILT'] > 25 ) & ( df["YRSOLD"] - df['YEARBUILT'] <= 50), 'NEW_BUILDING_AGE_CAT'] = 'fair_average'
df.loc[( df["YRSOLD"] - df['YEARBUILT'] > 50 ) & ( df["YRSOLD"] - df['YEARBUILT'] <= 75), 'NEW_BUILDING_AGE_CAT'] = 'old'
df.loc[( df["YRSOLD"] - df['YEARBUILT'] > 75 ) ,'NEW_BUILDING_AGE_CAT'] = 'anciently_old'


df["NEW_TOTAL_EXT"] = df["EXTERCOND"] + df["EXTERQUAL"]
df["NEW_TOTAL_BSMT"] = df["BSMTQUAL"] + df["BSMTCOND"]
df["NEW_TOTAL_GRG"] = df["GARAGEQUAL"] + df["GARAGEQUAL"]
df["NEW_TOTAL_BSMTBATH"] = df["BSMTFULLBATH"] + 0.5* df["BSMTHALFBATH"]
df["NEW_TOTAL_FULLBATH"] = df["FULLBATH"] + 0.5* df["HALFBATH"]
df["NEW_TOTAL_TOTALBATH"] = df["NEW_TOTAL_BSMTBATH"] + df["NEW_TOTAL_FULLBATH"]
df["NEW_HAVE_GARAGE"] = df["GARAGEYRBLT"].notnull().astype('int')
df["NEW_QUAL_COND"] = df['OVERALLQUAL'] + df['OVERALLCOND']

df["NEW_RENOVATED"] = df.apply(lambda row: 1 if (int(row.YEARREMODADD) - int(row.YEARBUILT)) > 0 else 0, axis=1)
df["NEW_QUALABOVEGRD"] = df["1STFLRSF"] + df["2NDFLRSF"]


df["NEW_HAVE_GARAGE"] = df["GARAGEYRBLT"].notnull().astype('int')
df["NEW_HOWRECENT_RENOVATED_BEFORESOLD"] = df.apply(lambda row: abs(int(row.YRSOLD)-int(row.YEARBUILT)) if int(row.NEW_RENOVATED)>= 1  else -1, axis=1)

df["NEW_CAT_LOTAREA"] = pd.qcut(df["LOTAREA"],5, labels=range(0,5))
df["LOTFRONTAGE"] = df["LOTFRONTAGE"].fillna(df.groupby("NEW_CAT_LOTAREA")["LOTFRONTAGE"].transform("mean"))

df.loc[df["GARAGEYRBLT"] == 2207, ["GARAGEYRBLT"]] = 2007
df["NEW_CAT_YEARBUILT"] = pd.qcut(df["YEARBUILT"],5, labels=range(0,5))
df["GARAGEYRBLT"] = df["GARAGEYRBLT"].fillna(df.groupby("NEW_CAT_YEARBUILT")["GARAGEYRBLT"].transform("median"))


df.loc[(df['FIREPLACES'] > 0) & (df['GARAGECARS'] >= 3), "NEW_LUX"] = 1
df["NEW_LUX"].fillna(0, inplace=True)
df["NEW_LUX"] = df["NEW_LUX"].astype(int)


cat_cols, num_cols, cat_but_car = grab_col_names(df)

na_cols = [col for col in num_cols if df[col].isnull().sum() > 0 and "SALEPRICE" not in col]

df[na_cols] = df[na_cols].apply(lambda x: x.fillna(x.median()), axis=0)

null_list = ["GARAGECARS","BSMTFULLBATH","BSMTHALFBATH","NEW_TOTAL_BSMTBATH"]
df[null_list] = df[null_list].fillna("mean")

ngb = df.groupby("NEIGHBORHOOD").SALEPRICE.mean().reset_index()
ngb["NEW_CLUSTER_NEIGHBORHOOD"] = pd.cut(df.groupby("NEIGHBORHOOD").SALEPRICE.mean().values, 5, labels=range(1, 6))
df = pd.merge(df, ngb.drop(["SALEPRICE"], axis=1), how="left", on="NEIGHBORHOOD")

df["NEW_TOTALPORCH"] = df["OPENPORCHSF"] + df["ENCLOSEDPORCH"] + df["3SSNPORCH"] + df["SCREENPORCH"]
df["NEW_TOTALSF"] = df["1STFLRSF"] + df["2NDFLRSF"] + df["TOTALBSMTSF"]
df["NEW_GARAGE_RATIO"] = df["GARAGEAREA"] / df["NEW_TOTALSF"]
df["NEW_PORCH_RATIO"] = df["NEW_TOTALPORCH"] / df["NEW_TOTALSF"]
df["NEW_HOUSE_RATIO"] = df["NEW_TOTALSF"] / df["LOTAREA"]
df["NEW_LIVING_AREA_RATIO"] = df["GRLIVAREA"] / df["NEW_TOTALSF"]


cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in cat_cols:
    cat_summary(df, col)

for col in cat_but_car:
    cat_summary(df, col)

for col in num_cols:
    num_summary(df, col, plot=True)


df = rare_encoder(df, 0.01)

rare_analyser(df, "SALEPRICE", cat_cols)

useless_cols = [col for col in cat_cols if df[col].nunique() == 1 or
                (df[col].nunique() == 2 and (df[col].value_counts() / len(df) <= 0.01).any(axis=None))]

for col in useless_cols:
    df.drop(col, axis=1, inplace=True)

cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols = cat_cols + cat_but_car

df = one_hot_encoder(df, cat_cols, drop_first=True)


cat_cols, num_cols, cat_but_car = grab_col_names(df)
df = rare_encoder(df, 0.01)

useless_cols_new = [col for col in cat_cols if (df[col].value_counts() / len(df) <= 0.01).any(axis=None)]

for col in useless_cols_new:
    df.drop(col, axis=1, inplace=True)

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
        replace_with_thresholds(df, col)

b = df.corr().sort_values(by="SALEPRICE", ascending=False)
b = b[(b["SALEPRICE"] > 0.7) | (b["SALEPRICE"] < -0.7)].index.tolist()

b = [i for i in b if "SALEPRICE" not in i]

for i in b:
    for j in b:
        df[f"NEW_{i[0:3]}_t_{j[0:3]}"] = df[i] + df[j]
        df[f"NEW_{i[0:3]}_d_{j[0:3]}"] = df[i] * df[j]
        df[f"NEW_{i[0:3]}_d_{j[0:3]}"] = df[i] / df[j]


a = df.corr().sort_values(by="SALEPRICE", ascending=False)

a = a[(a["SALEPRICE"] > 0.07) | (a["SALEPRICE"] < -0.07)].index.tolist()


df = df[a]

train_df = df[df['SALEPRICE'].notnull()]
test_df = df[df['SALEPRICE'].isnull()].drop("SALEPRICE", axis=1)

y = np.log1p(train_df['SALEPRICE'])
X = train_df.drop(["SALEPRICE"], axis=1)


######################################## Establish First Models ########################################################
models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

######################################## GBM Hyperparameter Tuning #####################################################
gbm_params = {"learning_rate": [0.01],
              "max_depth": [3],
              "n_estimators": [3000,4000],
              "subsample": [0.3],
              "loss": ['huber'],
              "max_features": [None,'sqrt']}

regressors = [('GBM', GradientBoostingRegressor(), gbm_params)]

best_models = {}

for name, regressor, params in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

    final_model = regressor.set_params(**gs_best.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model


############################################# Exporting Prediction #####################################################

final_model = final_model.fit(X, y)
submission_df = pd.DataFrame()
submission_df['ID'] = test["Id"]
y_pred_sub = final_model.predict(test_df)
y_pred_sub = np.expm1(y_pred_sub)
submission_df['SALEPRICE'] = y_pred_sub
submission_df.to_csv('submission1.csv', index=False)