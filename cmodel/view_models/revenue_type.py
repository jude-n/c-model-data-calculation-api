from enum import Enum


# https://py.watch/convert-a-python-enum-to-json-5eb5e94ecc9
class RevenueType(str, Enum):
    RECURRING_REVENUE = "Recurring Revenue"
    NON_RECURRING_REVENUE = "Non Recurring Revenue"
    USAGE = "Usage"
    SERVICE_BILLABLE = "Services Billable"
    SERVICES_FIXED = "Services Fixed"
