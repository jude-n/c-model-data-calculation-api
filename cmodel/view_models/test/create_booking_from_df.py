import json
import pandas as pd

from django.core import serializers

from cmodel.view_models.booking_class import Booking
from cmodel.view_models.model_data_class import ModelData
from cmodel.view_models.product_data_class import ProductData
from cmodel.view_models.time_period_class import TimePeriod


df_joined_data = pd.read_pickle('join_data_df.pkl')

bookings = []

for i, row in df_joined_data.iterrows():
    timePeriod = TimePeriod(month=row['month'], monthName=row['monthName'], year=row['year']).__dict__
    productData = ProductData(name=row['Group Name'], totalRevenue=row['c-model']).__dict__
    modelData = ModelData().__dict__
    booking = Booking(timePeriod=timePeriod, productData=productData, modelData=modelData).__dict__
    bookings.append(booking)

serialized = json.dumps(bookings)

print(serialized)
