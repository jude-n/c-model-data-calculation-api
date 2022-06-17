from enum import Enum

class RevenueType(Enum):
        recurringRevenue = "Recurring Revenue"
        nonRecurringRevenue = "Non Recurring Revenue"
        usage = "Usage"
        servicesBillable = "Services Billable"
        servicesFixed = "Services Fixed"