# this script predicts house prices using several variations of linear regression models and saves the results

from helpers import *
from constants import *

def run():

    # load/prepare data
    df = load_data()
    train_df, test_df = train_test_split(df)

    # load configs
    # model_name, train_feat_func, learning_rate, num_iters
    configs = [['lin_1', set_train_vars, set_test_vars, 5 * 10 ** -7, 1500],
               ['lin_2', set_train_vars, set_test_vars, 5 * 10 ** -7, 1500],
               ['lin_scaled_1', set_scaled_vars, set_scaled_test_vars, 0.01, 1500],
               ['lin_scaled_2', set_scaled_vars, set_scaled_test_vars, 0.05, 1500],
               ['lin_scaled_3', set_scaled_vars, set_scaled_test_vars, 0.1, 1500],
               ['lin_scaled_4', set_scaled_vars, set_scaled_test_vars, 0.5, 1500]]

    # todo:
    #  [] put all these lists in a cache
    #  [] turn config into .json/.yaml file
    #  [] fix feature scaling
    #  [] polynomial regression features

    model_names  = []
    preds = []
    costs = []
    mapes = []
    rmses = []
    train_times = []
    weights = []
    biases = []

    n_models = len(configs)

    if results_exists():
        last_updated, past_models = last_results_update()

    for i in range(n_models):
            # set model vars
            model_name = configs[i][0]
            train_feat_func = configs[i][1]
            test_feat_func = configs[i][2]
            learning_rate = configs[i][3]
            num_iters = configs[i][4]

            if results_exists():
                if model_name in past_models:
                    n_models -= 1
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
            X, Y, m, n = test_feat_func(test_df)

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
        preds_path = Path(Path.cwd().parent.absolute(), 'data', 'predictions.csv')

        if results_exists():
            old_preds_df = pd.read_csv(preds_path)
            updated_preds_df = old_preds_df.join(preds_df)
            updated_preds_df.to_csv(preds_path, index=False)

        else:
            preds_df.insert(0,'test', Y)
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
        results_path = Path(Path.cwd().parent.absolute(), 'data', 'results.csv')

        if results_exists():
            old_results_df = pd.read_csv(results_path)
            updated_results_df = pd.concat([old_results_df, results_df])
            updated_results_df.to_csv(results_path, index=False)

        else:
            results_df.to_csv(results_path, index=False)

    # print results / prediciotns for testing
    # print(pd.read_csv(Path(Path.cwd().parent.absolute(), 'data', 'results.csv')))
    # print(pd.read_csv(Path(Path.cwd().parent.absolute(), 'data', 'predictions.csv')).values.shape)

if __name__ == '__main__':
    run()
