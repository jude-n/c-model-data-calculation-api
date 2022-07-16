import json
import pandas as pd

from cmodel.view_models.booking import Booking
from cmodel.view_models.model_data import ModelData
from cmodel.view_models.product_data import ProductData
from cmodel.view_models.time_period import TimePeriod

df_joined_data = pd.read_pickle('join_data_df.pkl')

bookings = []

quarter_needs_to_be_removed = 0
for i, row in df_joined_data.iterrows():

    if i % 3 == 0:
        quarter_needs_to_be_removed = quarter_needs_to_be_removed + 1

    timePeriod = TimePeriod(quarter=quarter_needs_to_be_removed, month=row['month'], monthName=row['monthName'], year=row['year']).__dict__
    productData = ProductData(groupId=i, groupName=row['Group Name'], totalRevenue=row['c-model']).__dict__
    modelData = ModelData().__dict__
    booking = Booking(timePeriod=timePeriod, productData=productData, modelData=modelData).__dict__
    bookings.append(booking)

serialized = json.dumps(bookings)

print(serialized)
