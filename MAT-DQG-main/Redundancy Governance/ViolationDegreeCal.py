class VioCal(object):
    def __init__(self, dncors, kncors, x):
        self.ncors = [dncors, kncors]
        self.x = x
        self.complexities, self.total_complexity = self.assign_complexity()

    # 为每个NCOR分配复杂度，并计算DNCOR和KNCOR的条数
    def assign_complexity(self):
        comlpexities = [[], []]  # 分别存储DNCOR和KNCOR的复杂度
        for i, ncor in enumerate(self.ncors):
            for r in ncor:
                comlpexities[i].append(len(r) - 1)  # 计算每条NCOR的复杂度

        total_complexities = []
        for x in self.ncors:
            total_complexities.append(len(x))   # DNCOR或KNCOR的总数
        return comlpexities, total_complexities

    # 单独计算DNCOR或KNCOR的违反度
    def cal_one_degree(self, ncor, complexity):
        '''
        :param ncor: 全部DNCOR或KNCOR
        :param complexity: ncor的复杂度
        :return:
        '''
        total_degree = 0
        for index, r in enumerate(ncor):
            sum = 0
            for f in r:
                flag = 0
                for item in f:
                    if self.x[item] == 1:
                        flag = 1
                        break
                    else:
                        flag = 0
                sum += flag
            if sum == 0:
                degree = 0
            else:
                degree = (sum - 1) / complexity[index]
            total_degree += degree
        return total_degree

    # 计算总NCOR的违反度
    def cal_all_degree(self):
        degrees = []
        for ncor, complexity in zip(self.ncors, self.complexities):
            degrees.append(self.cal_one_degree(ncor, complexity))
        return degrees
