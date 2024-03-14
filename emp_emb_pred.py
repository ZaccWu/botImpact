import numpy as np
import xgboost as xgb
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

check_model = 'cne-'
check_type = 't1_pos'

MSE_LASSO, MSE_RF, MSE_XGB, MAE_LASSO, MAE_RF, MAE_XGB = [], [], [], [], [], []

for seed in range(101,111):
    # Z_emb, treat_lb, y_tar
    emb = np.load('result/save_emb/'+check_type+'/'+check_model+str(seed)+'.npy')
    X, y = emb[:,:-1], emb[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

    lasso_model = Lasso(alpha=0.1, random_state=42)
    lasso_model.fit(X_train, y_train)
    y_pred_lasso = lasso_model.predict(X_test)

    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    print("MSE in seed ", seed)
    print("Lasso: {:.4f}, RF: {:.4f}, XGB: {:.4f}".format(mse_lasso, mse_rf, mse_xgb))
    MSE_LASSO.append(mse_lasso)
    MSE_RF.append(mse_rf)
    MSE_XGB.append(mse_xgb)

    mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    print("MAE in seed ", seed)
    print("Lasso: {:.4f}, RF: {:.4f}, XGB: {:.4f}".format(mae_lasso, mae_rf, mae_xgb))
    MAE_LASSO.append(mae_lasso)
    MAE_RF.append(mae_rf)
    MAE_XGB.append(mae_xgb)

print("-------------------------------------")
print("Result in ", check_type, " for model ", check_model)
print("10 Ave MSE | Lasso:  {:.4f},  RF:  {:.4f},  XGB:  {:.4f}".format(np.mean(MSE_LASSO), np.mean(MSE_RF), np.mean(MSE_XGB)))
print("Std        | Lasso: ({:.4f}), RF: ({:.4f}), XGB: ({:.4f})".format(np.std(MSE_LASSO), np.std(MSE_RF), np.std(MSE_XGB)))

print("10 Ave MAE | Lasso:  {:.4f},  RF:  {:.4f},  XGB:  {:.4f}".format(np.mean(MAE_LASSO), np.mean(MAE_RF), np.mean(MAE_XGB)))
print("Std        | Lasso: ({:.4f}), RF: ({:.4f}), XGB: ({:.4f})".format(np.std(MSE_LASSO), np.std(MSE_RF), np.std(MSE_XGB)))