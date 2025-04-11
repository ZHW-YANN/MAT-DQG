from bayes_opt import BayesianOptimization
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel, RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.neural_network import MLPRegressor
import numpy as np

def pre1(model,x_test,y_test):
    y_pred = model.predict(x_test)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    R2 = r2_score(y_test, y_pred)
    MAPE = np.mean(np.abs((y_pred - y_test) / y_test)) * 100
    return RMSE, MAPE, R2


def my_loss_func(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

score = make_scorer(my_loss_func, greater_is_better=False)

def lr_predictor(train_x, train_y, kf):
    tuned_parameters = {}
    lr = GridSearchCV(LinearRegression(), tuned_parameters, cv=kf, scoring=score)
    return 1, lr

def bayes_optimal_lasso(train_x, train_y, kf):
    def lasso_cv(alpha):
        val = cross_val_score(
            Lasso(alpha=alpha),
            train_x, train_y, cv=kf, scoring=score
        ).mean()
        return val
    lasso_pbounds = {'alpha': (1, 10)}  # defult
    lasso_optimizer = BayesianOptimization(
        f=lasso_cv,
        pbounds=lasso_pbounds,
        verbose=2,
        random_state=7,
        allow_duplicate_points=True,)
    lasso_optimizer.maximize(
        init_points=5,
        n_iter=30,
    )
    target = lasso_optimizer.max.get('target')
    alpha = lasso_optimizer.max.get('params').get('alpha')
    model = Lasso(alpha=alpha)

    return target, model


def bayes_optimal_ridge(train_x, train_y, kf):
    def ridge_cv(alpha, max_iter):
        val = cross_val_score(
            Ridge(alpha=alpha, max_iter=max_iter),
            train_x, train_y, cv=kf, scoring=score
        ).mean()
        return val
    ridge_pbounds = {'alpha': (0.01, 10),
                     'max_iter': (1, 5000)}
    ridge_optimizer = BayesianOptimization(
        f=ridge_cv,
        pbounds=ridge_pbounds,
        verbose=2,
        random_state=7,)
    ridge_optimizer.maximize(
        init_points=20,
        n_iter=30,
    )
    target = ridge_optimizer.max.get('target')
    alpha = ridge_optimizer.max.get('params').get('alpha')
    max_iter = ridge_optimizer.max.get('params').get('max_iter')
    model = Ridge(alpha=alpha, max_iter=max_iter)
    return target, model


def bayes_optimal_rf(train_x, train_y, kf):
    def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
        val = cross_val_score(
            RandomForestRegressor(n_estimators=int(n_estimators),
                                  min_samples_split=int(min_samples_split),
                                  max_features=min(max_features, 0.999),  # float
                                  max_depth=int(max_depth),
                                  random_state=7),
            train_x, train_y, cv=kf, scoring=score
        ).mean()
        return val

    rf_pbounds = {'n_estimators': (10, 300),
                  'min_samples_split': (2, 15),
                  'max_features': (0.01, 0.999),
                  'max_depth': (3, 20),
                  }
    rf_optimizer = BayesianOptimization(
        f=rf_cv,
        pbounds=rf_pbounds,
        verbose=2,
        random_state=7,
    )
    rf_optimizer.maximize(
        init_points=20,
        n_iter=30,
    )
    target = rf_optimizer.max.get('target')
    n_estimators = rf_optimizer.max.get('params').get('n_estimators')
    min_samples_split = rf_optimizer.max.get('params').get('min_samples_split')
    max_features = rf_optimizer.max.get('params').get('max_features')  # float
    max_depth = rf_optimizer.max.get('params').get('max_depth')
    model = RandomForestRegressor(n_estimators=int(n_estimators), min_samples_split=int(min_samples_split),
                                  max_features=max_features, max_depth=int(max_depth))
    return target, model


def bayes_optimal_svr(train_x, train_y, kf):
    def svr_cv(C, gamma):
        val = cross_val_score(
            SVR(kernel='rbf',
                C=C,
                gamma=gamma,
                ),
            train_x, train_y, cv=kf, scoring=score
        ).mean()
        return val

    svr_pbounds = {'gamma': (1e-5, 1), 'C': (1e-4, 500) }
    svr_optimizer = BayesianOptimization(
        f=svr_cv,
        pbounds=svr_pbounds,
        verbose=2,
        random_state=7,
        allow_duplicate_points=True
        )
    svr_optimizer.maximize(
        init_points=5,
        n_iter=20,
    )
    target = svr_optimizer.max.get('target')
    C = svr_optimizer.max.get('params').get('C')
    gamma = svr_optimizer.max.get('params').get('gamma')
    model = SVR(kernel='rbf', C=C, gamma=gamma)

    return target, model


def bayes_optimal_knn(train_x, train_y, kf):
    def knn_cv(n_neighbors, p):
        val = cross_val_score(
            KNeighborsRegressor(weights='distance',
                                n_neighbors=int(n_neighbors),
                                p=p
                                ),
            train_x, train_y, cv=kf, scoring=score
        ).mean()
        return val

    knn_pbounds = {'n_neighbors': (2, 30),
                   'p': (1, 8)}
    knn_optimizer = BayesianOptimization(
        f=knn_cv,
        pbounds=knn_pbounds,
        verbose=2,
        random_state=7,
        allow_duplicate_points=True)
    knn_optimizer.maximize(
        init_points=5,
        n_iter=25,
    )
    target = knn_optimizer.max.get('target')
    n_neighbors = knn_optimizer.max.get('params').get('n_neighbors')
    p = knn_optimizer.max.get('params').get('p')
    model = KNeighborsRegressor(weights='distance', n_neighbors=int(n_neighbors), p=p)
    return target, model


def bayes_optimal_gpr(train_x, train_y, kf):
    def gpr_cv(alpha, C):
        val = cross_val_score(
            GaussianProcessRegressor(kernel=C * RationalQuadratic(alpha=0.01, length_scale_bounds=(0.1, 1500)),
                                     alpha=alpha,
                                     n_restarts_optimizer=20),
            train_x, train_y, cv=kf, scoring=score
        ).mean()
        return val

    gpr_pbounds = {'alpha': (0.0001, 1.0),
                   'C': (1e-5, 1e5)}
    gpr_optimizer = BayesianOptimization(
        f=gpr_cv,
        pbounds=gpr_pbounds,
        verbose=2,
        random_state=7,)
    gpr_optimizer.maximize(
        init_points=5,
        n_iter=10,
    )
    target = gpr_optimizer.max.get('target')
    alpha = gpr_optimizer.max.get('params').get('alpha')
    C = gpr_optimizer.max.get('params').get('C')
    model = GaussianProcessRegressor(kernel = C * RationalQuadratic(alpha=0.01, length_scale_bounds=(0.1, 3000)), alpha=alpha, n_restarts_optimizer=20)
    return target, model


def predictors(train_x, train_y, kf, type='SVR'):
    if type == 'SVR':
        t, model = bayes_optimal_svr(train_x, train_y, kf)
    elif type == 'RF':
        t, model = bayes_optimal_rf(train_x, train_y, kf)
    elif type == 'LASSO':
        t, model = bayes_optimal_lasso(train_x, train_y, kf)
    elif type == 'Ridge':
        t, model = bayes_optimal_ridge(train_x, train_y, kf)
    elif type == 'KNN':
        t, model = bayes_optimal_knn(train_x, train_y, kf)
    elif type == 'GPR':
        t, model = bayes_optimal_gpr(train_x, train_y, kf)
    else:
        t, model = lr_predictor(train_x, train_y, kf)
    return t, model


