import numpy as np
from sklearn import preprocessing
from FeatureSelection.ViolationDegreeCal import VioCal as VC
from ModelsCV import predictors
from FeatureSelection.NondominatedSolutionCal import NonSolu
import copy


def flatten(data):
    flatten_data = []
    for i in range(len(data)):
        flatten_data.extend(data[i])
    return flatten_data


# 控制粒子的目标函数值不超过最低值，设为0相当于无效
min_value1, min_value2, min_kncor_value = 0, 0, 0


class BPSO(object):
    def __init__(self, data, p, max, num, best, kernel, rules, model):
        self.train_data = data
        self.model_type = model
        self.w_max = 0.9  # 最大权重
        self.w_min = 0.4  # 最小权重
        self.w = 0.9  # 初始权重
        self.v_max = 6  # 最大速度
        self.c1 = self.c2 = 2  # 学习因子
        self.population_size = p  # 种群大小
        self.dim = self.train_data.dim - len(kernel)  # 搜索空间的维度
        self.max_steps = max  # 最大迭代次数
        self.num_of_funcs = num  # 目标函数的数量
        self.best_values = best  # 迭代停止阈值
        self.feature_kernel = kernel  # 核特征
        self.rules = rules  # NCORs
        self.feature_space = np.delete(range(self.train_data.dim), self.feature_kernel)  # 特征空间
        self.p_archive = 20
        self.g_archive = 20
        self.x, self.v = self.init_swarm()  # 粒子群的位置和速度向量
        self.values = self.cal_for_all()  # 粒子群的目标函数值
        self.order = self.sort('x')  # 多目标排序结果（好→坏）
        self.p_best_archive, self.p_best = self.init_p_best()  # p_best及其外部档案
        self.g_best_archive = []  # g_best的外部档案
        self.g_best = self.update_g_best(0)  # g_best

    # 初始化所有粒子的位置和速度向量
    def init_swarm(self):
        x = np.random.randint(0, 2, (1, self.dim))
        v = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size - 1):
            x = np.vstack((x, np.random.randint(0, 2, (1, self.dim))))
        return x, v

    # 初始化p_best及其外部存储
    def init_p_best(self):
        p_best, p_best_archive = [], []
        for i in range(self.population_size):
            temp = {'x': copy.deepcopy(self.x[i]),
                    'value1': copy.deepcopy(self.values[i]['value1']),
                    'value2': copy.deepcopy(self.values[i]['value2'])}
            p_best.append(temp)
            p_best_archive.append([copy.deepcopy(temp)])
        return p_best_archive, p_best

    # 计算单个粒子的NCOR违反度
    def cal_objective_func1(self, x):
        if len(self.rules['dncors']) == 0 and len(self.rules['kncors']) == 0:
            value = 0
        else:
            vc = VC(self.rules['dncors'], self.rules['kncors'], x)
            degrees = vc.cal_all_degree()  # 计算NCOR违反度
            if len(self.rules['dncors']) == 0 and len(self.rules['kncors']) != 0:
                temp_knor_degree = degrees[1] / vc.total_complexity[1]  # 计算KNCOR违反度
                if temp_knor_degree < min_kncor_value:
                    knor_degree = 1
                else:
                    knor_degree = temp_knor_degree - min_kncor_value
                value = knor_degree
            elif len(self.rules['dncors']) != 0 and len(self.rules['kncors']) == 0:
                dnor_degree = degrees[0] / vc.total_complexity[0]  # 计算DNCOR违反度
                value = dnor_degree
            else:
                temp_knor_degree = degrees[1] / vc.total_complexity[1]
                if temp_knor_degree < min_kncor_value:
                    knor_degree = 1
                else:
                    knor_degree = temp_knor_degree - min_kncor_value
                dnor_degree = degrees[0] / vc.total_complexity[0]
                value = dnor_degree + knor_degree  # 计算总NCOR违反度
        return value

    # 计算单个粒子的CV RMSE
    def cal_objective_func2(self, data_x, data_y):
        if len(data_x) == 0 or np.array(data_x).shape[1] == 0:
            avg_rmse = np.inf
        else:
            data_x = np.array(data_x)
            data_y = np.array(data_y)

            scaler = preprocessing.StandardScaler()
            x_train_pro = scaler.fit_transform(data_x)
            model = predictors(x_train_pro, data_y, type=self.model_type)
            avg_rmse = np.abs(model.best_score_)
        return avg_rmse

    # 计算单个粒子的全部目标函数值
    def cal_all_funcs(self, x, feature_set):
        if len(feature_set) == 1:
            train_data_x = np.array(self.train_data.X[:, feature_set]).reshape(-1, 1)
        else:
            train_data_x = np.array(self.train_data.X[:, feature_set])
        train_data_y = np.array(self.train_data.Y)
        value1 = self.cal_objective_func1(x)
        value2 = self.cal_objective_func2(train_data_x, train_data_y)
        values = {'value1': value1, "value2": value2}
        return values

    # 计算全部粒子的全部目标函数值
    def cal_for_all(self):
        all_values = []
        for i in range(self.population_size):
            selected_features = self.feature_space[[index for index, values in enumerate(self.x[i]) if values == 1]]
            feature_set = np.append(selected_features, self.feature_kernel).astype('int32')
            values = self.cal_all_funcs(self.x[i], feature_set)
            all_values.append(values)
        return all_values

    # 计算非支配解
    def cal_non_dominated_solutions(self, obj):
        top = []  # Pareto等级最高的解
        num_of_dominated = None
        sets_of_domination = None

        # 计算每个粒子的支配和非支配解
        if obj == 'x':
            ns = NonSolu(self.population_size, self.values, [min_value1, min_value2])
            num_of_dominated, sets_of_domination = ns.find_nondomination()
        if obj == 'pbest':
            ns = NonSolu(self.population_size, self.p_best, [min_value1, min_value2])
            num_of_dominated, sets_of_domination = ns.find_nondomination()

        for i in range(self.population_size):
            if num_of_dominated[i] == 0:
                top.append(i)

        return num_of_dominated, sets_of_domination, top

    # 非支配排序
    def sort_by_pareto(self, obj):
        num_of_dominated, sets_of_domination, top1 = self.cal_non_dominated_solutions(obj)
        temp_top1 = copy.deepcopy(top1)
        temp_top2, all_order = [], []
        all_order.append(temp_top1)

        # 根据Pareto等级划分粒子群
        while len(flatten(all_order)) != self.population_size:
            for t1 in temp_top1:
                for item in sets_of_domination[t1]:
                    num_of_dominated[item] = num_of_dominated[item] - 1
                    if num_of_dominated[item] == 0:
                        temp_top2.append(item)
            if len(temp_top2):
                all_order.append(temp_top2)
                temp_top1 = temp_top2
                temp_top2 = []
        return all_order

    # 计算拥挤度
    def cal_congestion_degree(self, some_level_x, object):
        value1s, value2s = [], []
        congestion_degree1, congestion_degree2 = np.zeros(len(some_level_x)), np.zeros(len(some_level_x))
        congestion_degree = [congestion_degree1, congestion_degree2]

        if object == 'x':
            for i in range(len(some_level_x)):
                value1s.append(self.values[some_level_x[i]]['value1'])
                value2s.append(self.values[some_level_x[i]]['value2'])
        elif object == 'pbest':
            for i in range(len(some_level_x)):
                value1s.append(self.p_best[some_level_x[i]]['value1'])
                value2s.append(self.p_best[some_level_x[i]]['value2'])
        elif object == 'gbest':
            for i in range(len(some_level_x)):
                value1s.append(self.g_best_archive[some_level_x[i]]['value1'])
                value2s.append(self.g_best_archive[some_level_x[i]]['value2'])
        else:
            for i in range(len(some_level_x)):
                value1s.append(object[some_level_x[i]]['value1'])
                value2s.append(object[some_level_x[i]]['value2'])

        values = [value1s, value2s]
        index_of_value1s, index_of_value2s = np.array(value1s).argsort(), np.array(value2s).argsort()
        index_of_values = [index_of_value1s, index_of_value2s]

        for i in range(self.num_of_funcs):
            index_of_best = index_of_values[i][0]
            index_of_worst = index_of_values[i][len(some_level_x) - 1]
            congestion_degree[i][index_of_best], congestion_degree[i][index_of_worst] = 1, 1
            if len(some_level_x) > 2:
                for j in range(len(some_level_x)):
                    if j != index_of_best and j != index_of_worst:
                        index_front = index_of_values[i][list(index_of_values[i]).index(j) - 1]
                        index_back = index_of_values[i][list(index_of_values[i]).index(j) + 1]
                        if values[i][index_of_worst] - values[i][index_of_best] == 0:
                            congestion_degree[i][j] = np.inf
                        else:
                            back_diff_front = values[i][index_back] - values[i][index_front]
                            if i == 1:
                                if values[i][index_back] < min_value2:
                                    back_diff_front = 0
                                elif values[i][index_front] < min_value2:
                                    back_diff_front = values[i][index_back] - min_value2
                            if i == 0:
                                if values[i][index_back] < min_value1:
                                    back_diff_front = 0
                                elif values[i][index_front] < min_value1:
                                    back_diff_front = values[i][index_back] - min_value1

                            congestion_degree[i][j] = back_diff_front / (
                                    values[i][index_of_worst] - values[i][index_of_best])
        final_congestion_degree = congestion_degree[0] + congestion_degree[1]
        return final_congestion_degree

    # 根据Pareto等级和拥挤度进行排序
    def sort(self, obj):
        order_by_pareto = self.sort_by_pareto(obj)
        order_by_congestion_degree = []
        for i in range(len(order_by_pareto)):
            congestion_degrees = self.cal_congestion_degree(order_by_pareto[i], obj)
            index_of_sort = np.array(-np.array(congestion_degrees)).argsort()
            temp = []
            for j in range(len(index_of_sort)):
                temp.append(order_by_pareto[i][index_of_sort[j]])
            order_by_congestion_degree.append(temp)
        final_order = flatten(order_by_congestion_degree)
        return final_order

    # 更新p_best及其外部存档
    def update_p_best(self):
        new_values = self.cal_for_all()
        self.values = copy.deepcopy(new_values)
        for i in range(self.population_size):
            new_particle = {'x': copy.deepcopy(self.x[i]),
                            'value1': copy.copy(new_values[i]['value1']),
                            'value2': copy.copy(new_values[i]['value2'])}
            self.p_best_archive[i].append(new_particle)

            top = []
            ns = NonSolu(len(self.p_best_archive[i]), self.p_best_archive[i], [min_value1, min_value2])
            num_of_dominated, sets_of_domination = ns.find_nondomination()

            for j in range(len(self.p_best_archive[i])):
                if num_of_dominated[j] == 0:
                    top.append(j)

            order_by_congestion_degree = []
            congestion_degrees = self.cal_congestion_degree(top, self.p_best_archive[i])
            index_of_sort = np.array(-np.array(congestion_degrees)).argsort()
            for j in range(len(index_of_sort)):
                order_by_congestion_degree.append(top[index_of_sort[j]])

            self.p_best_archive[i] = list(np.array(self.p_best_archive[i])[order_by_congestion_degree])
            self.p_best[i] = copy.deepcopy(self.p_best_archive[i][0])
            if len(self.p_best_archive[i]) > self.p_archive:
                self.p_best_archive[i] = self.p_best_archive[i][: self.p_archive]

    # 更新g_best及其外部存档
    def update_g_best(self, step):
        all_p_best = []
        for p in self.p_best:
            all_p_best.append(p)

        p_top = []
        ns = NonSolu(len(all_p_best), all_p_best, [min_value1, min_value2])
        num_of_dominated, sets_of_domination = ns.find_nondomination()
        for i in range(len(all_p_best)):
            if num_of_dominated[i] == 0:
                p_top.append(i)

        is_exist = False
        for index in p_top:
            temp = all_p_best[index]
            for tg in self.g_best_archive:
                if (np.array(temp['x']) == np.array(tg['x'])).all():
                    is_exist = True
            if not is_exist:
                self.g_best_archive.append(temp)

        g_top = []
        ns = NonSolu(len(self.g_best_archive), self.g_best_archive, [min_value1, min_value2])
        num_of_dominated, sets_of_domination = ns.find_nondomination()
        for i in range(len(self.g_best_archive)):
            if num_of_dominated[i] == 0:
                g_top.append(i)

        order_by_congestion_degree = []
        congestion_degrees = self.cal_congestion_degree(g_top, 'gbest')
        index_of_sort = np.array(-np.array(congestion_degrees)).argsort()
        for j in range(len(index_of_sort)):
            order_by_congestion_degree.append(g_top[index_of_sort[j]])
        if len(order_by_congestion_degree) > self.g_archive:
            order_by_congestion_degree = order_by_congestion_degree[: self.g_archive]
        self.g_best_archive = list(np.array(self.g_best_archive)[order_by_congestion_degree])

        if step / self.max_steps <= 0.6:
            new_g_best_x = self.g_best_archive[0]
        else:
            min = np.inf
            best = None
            for g in self.g_best_archive:
                if min_value2 <= g['value2'] < min and g['value1'] >= min_value1:
                    best = copy.deepcopy(g)
                    min = g['value2']
            if not best:
                for g in self.g_best_archive:
                    if 0.1 ** np.sign(g['value2'] - min_value2) * np.abs(g['value2'] - min_value2) < min:
                        best = copy.deepcopy(g)
                        min = g['value2']
            new_g_best_x = best
        return new_g_best_x

    # 进化
    def evolve(self):
        n1 = 0  # 记录g_best不变的次数
        for step in range(self.max_steps):

            r1 = np.random.uniform(0, 1, (self.population_size, self.dim))
            r2 = np.random.uniform(0, 1, (self.population_size, self.dim))
            self.w = self.w_max - (self.w_max - self.w_min) * step / self.max_steps

            # 更新速度向量
            p_best_x = []
            for i in range(self.population_size):
                p_best_x.append(self.p_best[i]['x'])
            self.v = self.w * self.v + self.c1 * r1 * (np.array(p_best_x) - self.x) + self.c2 * r2 * (
                    self.g_best['x'] - self.x)
            self.v[self.v < -self.v_max] = -self.v_max
            self.v[self.v > self.v_max] = self.v_max

            # 更新x, p_best, p_best_archive, gbest和g_best_archive
            sig_x = 1 / (1 + np.exp(- self.v))
            self.x = np.array([[1 if np.random.rand() < v else 0 for v in vs] for vs in sig_x])
            self.update_p_best()
            before_g_best = copy.deepcopy(self.g_best)
            self.g_best = self.update_g_best(step)
            selected_features = self.feature_space[
                [index for index, values in enumerate(self.g_best['x']) if values == 1]]
            feature_set = sorted(np.append(self.feature_kernel, selected_features).astype('int32'))
            print('第%d次迭代：' % step)
            print('最优的特征子集是：', feature_set)
            print('特征数：', len(feature_set))
            print('目标函数1：%.5f' % self.g_best['value1'])
            print('目标函数2：%.5f' % self.g_best['value2'])

            if (before_g_best['x'] == self.g_best['x']).all():
                n1 += 1
            else:
                n1 = 0
            if step / self.max_steps > 0.6 and n1 > 5:
                if min_value1 <= self.g_best['value1'] <= self.best_values['value1'] \
                        and min_value2 <= self.g_best['value2'] <= self.best_values['value2']:
                    break
                else:
                    n1 = 0
                    num_of_change = int(self.population_size * 0.3)
                    x_index = self.order[-num_of_change:]

                    for k in range(num_of_change):
                        if np.random.rand() > 0.7:
                            self.x[x_index[k]] = np.ones(self.dim)
                            options = np.arange(self.dim + len(self.feature_kernel))
                            values = self.cal_all_funcs(self.x[x_index[k]], options)
                            self.values[x_index[k]]['value1'] = copy.copy(values['value1'])
                            self.values[x_index[k]]['value2'] = copy.copy(values['value2'])
                            self.p_best_archive[x_index[k]] = [
                                {'x': np.ones(self.dim), 'value1': copy.copy(values['value1']),
                                 'value2': copy.copy(values['value2'])}]
                        else:
                            self.x[x_index[k]] = np.zeros(self.dim)
                            self.values[x_index[k]]['value1'] = 1
                            self.values[x_index[k]]['value2'] = 1
                            self.p_best_archive[x_index[k]] = [{'x': np.zeros(self.dim), 'value1': 1, 'value2': np.inf}]

        min = np.inf
        best = None
        for x in self.g_best_archive:
            if min_value2 <= x['value2'] < min and x['value1'] > min_value1:
                min = x['value2']
                best = copy.deepcopy(x)
        if not best:
            for x in self.g_best_archive:
                if min_value2 <= x['value2'] < min:
                    min = x['value2']
                    best = copy.deepcopy(x)
        final_selected_features = self.feature_space[[index for index, values in enumerate(best['x']) if values == 1]]
        final_feature_set = sorted(np.append(self.feature_kernel, final_selected_features).astype('int32'))
        return best, final_feature_set
