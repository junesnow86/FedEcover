import torch
import torch.nn as nn


class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = nn.ModuleList(models)

    def add_model(self, model):
        self.models.append(model)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        avg_output = torch.mean(torch.stack(outputs), dim=0)
        return avg_output


# 示例用法
if __name__ == "__main__":
    # 假设我们有两个简单的线性模型
    model1 = nn.Linear(10, 2)
    model2 = nn.Linear(10, 2)

    # 创建Ensemble对象
    ensemble_model = Ensemble([model1, model2])

    # 添加另一个模型
    model3 = nn.Linear(10, 2)
    ensemble_model.add_model(model3)

    # 输入数据
    input_data = torch.randn(5, 10)

    # 获取Ensemble模型的预测结果
    output = ensemble_model(input_data)
    print(output)
