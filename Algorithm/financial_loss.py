"""
@Params
predicted_prices -> Predictions from the models
actual_prices -> Actual prices from the industries
daily_consumption -> Daily usage for the commodity
current_stock -> Current stock hold
max_stock -> daily_consumption * max_days (Total stock hold based on financial and warehouse constraints)
"""


def execute_purchase_strategy(predicted_prices, actual_prices, daily_consumption, current_stock, max_stock):
    def purchase_strategy(prices):
        n = len(prices)
        m = max_stock + 1
        DP = [[float('inf')] * m for _ in range(n)]
        previous_decision = [[0] * m for _ in range(n)]

        for stock in range(m):
            DP[0][stock] = (stock - current_stock + daily_consumption) * prices[0]

        for day in range(1, n):
            for stock_today in range(m):
                for stock_yesterday in range(m):
                    purchase_today = stock_today - stock_yesterday + daily_consumption
                    if 0 <= purchase_today <= max_stock:
                        cost = DP[day - 1][stock_yesterday] + purchase_today * prices[day]
                        if cost < DP[day][stock_today]:
                            DP[day][stock_today] = cost
                            previous_decision[day][stock_today] = stock_yesterday

        stock = 0
        days_to_buy = []
        purchase_quantities = []
        for day in range(n - 1, -1, -1):
            stock_yesterday = previous_decision[day][stock]
            purchase_today = stock - stock_yesterday + daily_consumption
            if purchase_today > 0:
                days_to_buy.append(day)
                purchase_quantities.append(purchase_today)
            stock = stock_yesterday

        return list(reversed(days_to_buy)), list(reversed(purchase_quantities))

    pred_days_to_buy, pred_purchase_quantities = purchase_strategy(predicted_prices)
    ori_days_to_buy, ori_purchase_quantities = purchase_strategy(actual_prices)

    total_predicted_price = sum(
        pred_purchase_quantities[i] * predicted_prices[pred_days_to_buy[i]] for i in range(len(pred_days_to_buy)))
    total_actual_price = sum(
        ori_purchase_quantities[i] * actual_prices[ori_days_to_buy[i]] for i in range(len(ori_days_to_buy)))
    result = (total_predicted_price - total_actual_price) / total_actual_price
    print(result)
