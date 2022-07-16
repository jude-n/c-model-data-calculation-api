class TimePeriod:
    def __init__(self, quarter, month, monthName, year):
        self.quarter = quarter
        self.month = month
        self.monthName = monthName
        self.year = year

    def __str__(self):
        return self.__dict__

    def __repr__(self):
        return self.__dict__
