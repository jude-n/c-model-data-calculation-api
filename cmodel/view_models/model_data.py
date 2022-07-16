from cmodel.view_models.model_type import ModelType
from cmodel.view_models.revenue_type import RevenueType


class ModelData:
    def __init__(self, median=0, mean=0, forecast=0, totalRevenue=0, revenueType=RevenueType.RECURRING_REVENUE, modelType=ModelType.CMODEL):
        self.median = median
        self.mean = mean
        self.forecast = forecast
        self.totalRevenue = totalRevenue
        self.revenueType = revenueType
        self.modelType = modelType
