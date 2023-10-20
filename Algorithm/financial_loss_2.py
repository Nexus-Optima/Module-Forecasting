import pandas as pd
from Models.XG_Boost.adaptive_xgboost import execute_evaluation
from Utils.process_data import process_data_lagged_rolling_stats

dailyConsumption = 10
maxDays = 40

data = pd.read_csv("../Data/ICAC multiple variables.csv", parse_dates=['Date'], dayfirst=True)
data = process_data_lagged_rolling_stats(data, 10)
data = data.last('5Y')
# prophet_pred = execute_prophet(data)
test, adaptive_xgb_pred  = execute_evaluation(data, hyperparams={})
# arima_pred, arima_forecast = execute_arima(data)
# print(prophet_pred)
# train_size = int(0.8 * len(data))
# test = data['Output'][train_size:]
print(test)
def optimum_purchase(pred):
    curStock = 0
    stockStatement = []

    for i in range(0, len(pred)):
        current = pred[i]
        for j in range(i, len(pred)):
            if pred[j] < current or j == len(pred) - 1:
                if maxDays < j - i:
                    curStock = maxDays
                else:
                    if curStock < j - i:
                        curStock = j - i
                stockStatement.append(curStock * dailyConsumption)
                break
        curStock -= 1
    return stockStatement


def compute_purchase_cost(stockStatement):
    stockToBuy = stockStatement[0]
    purchaseCost = [test[0] * stockToBuy]
    for i in range(0, len(stockStatement) - 1):
        # print(len(stockStatement)-1)
        if stockStatement[i] <= stockStatement[i + 1]:
            stockToBuy = stockStatement[i + 1] - stockStatement[i] + 1 * dailyConsumption
            purchaseCost.append(stockToBuy * test[i + 1])
            # print("Buy " + str(stockStatement[i + 1] - stockStatement[i] + 1 * dailyConsumption))
    return purchaseCost


predStockStatement = optimum_purchase(adaptive_xgb_pred)
print(predStockStatement)
print(len(test))
predPurchaseCost = compute_purchase_cost(predStockStatement)
actualStockStatement = optimum_purchase(test)
actualPurchaseCost = compute_purchase_cost(actualStockStatement)

print("error is " + str((sum(predPurchaseCost)-sum(actualPurchaseCost))/sum(actualPurchaseCost)))
