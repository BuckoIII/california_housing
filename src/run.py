# this script predicts house prices using several variations of linear regression models and saves the results

from helpers import *
from constants import *

def run():

    # load/prepare data
    df = load_data()
    train_df, test_df = train_test_split(df)

    # load configs
    # model_name, train_feat_func, learning_rate, num_iters
    configs = [['lin_1', set_train_vars, 5 * 10 ** -7, 1500],
               ['lin_2', set_train_vars, 5 * 10 ** -7, 15],
               ['lin_3', set_train_vars, 5 * 10 ** -7, 150]]

    # todo:
    #  [] put all these lists in a cache
    #  [] write if statements to skip models which have already been run

    model_names  = []
    preds = []
    costs = []
    mapes = []
    rmses = []
    train_times = []
    weights = []
    biases = []

    n_models = len(configs)


    last_updated, past_models = last_results_update()

    for i in range(n_models):
            # set model vars
            model_name = configs[i][0]
            train_feat_func = configs[i][1]
            learning_rate = configs[i][2]
            num_iters = configs[i][3]

            if model_name in past_models:
                continue

            # set train vars
            X, Y, m, n = train_feat_func(train_df)
            w, b = initial_zeros(X)

            # run gradient descent
            w_final, b_final, cost_history, train_time = gradient_descent(X, Y, w, b,
                                                        calculate_cost, calculate_grads,
                                                        learning_rate, num_iters)
            print(f"b,w found by gradient descent: {b_final},{w_final} ")

            # set test vars
            X, Y, m, n = set_test_vars(test_df)

            # predict
            Y_hat = predict(X, w_final, b_final)

            model_names.append(model_name)
            preds.append(Y_hat[0])

            costs.append(cost_history[-1])
            mapes.append(mape(Y, Y_hat))
            rmses.append(rmse(Y, Y_hat))
            train_times.append(train_time)
            weights.append(w_final)
            biases.append(b_final)
            last_updates = [datetime.now() for i in range(n_models)]

    if costs:
        preds_df = pd.DataFrame(dict(zip(model_names, preds)))
        print(preds_df)

        preds_path = Path(Path.cwd().parent.absolute(), 'data', 'predictions.csv')
        print(preds_path)
        preds_df.to_csv(preds_path, index=False)

        results_data = {'model_name': model_names,
                        'cost': costs,
                        'mape': mapes,
                        'rmse': rmses,
                        'train_time': train_times,
                        'weight':weights,
                        'bias': biases,
                        'last_updated': last_updates}
        results_df = pd.DataFrame(results_data)
        print(results_df)
        results_path = Path(Path.cwd().parent.absolute(), 'data', 'results.csv')
        results_df.to_csv(results_path, index=False)


if __name__ == '__main__':
    run()
