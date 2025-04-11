class ConversionRule(object):
    def __init__(self, rules, x):
        self.rules = rules
        self.x = x
        self.complexity = self.assign_complexity()
        self.total_complexity = sum(self.complexity)
        self.total_degree = self.cal_degree()

    # 分配复杂度
    def assign_complexity(self):
        comlpexities = []
        for r in self.rules:
            comlpexities.append(len(r)-1)
        return comlpexities

    # 计算NCOR违反度
    def cal_degree(self):
        total_degree = 0
        for index, r in enumerate(self.rules):
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
                degree = (sum - 1) / self.complexity[index]
            total_degree += degree
        return total_degree