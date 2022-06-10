# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class Address(models.Model):
    id = models.BigIntegerField(primary_key=True)
    companyid = models.ForeignKey('Companies', models.DO_NOTHING, db_column='companyid', blank=True, null=True)
    countryid = models.ForeignKey('Regions', models.DO_NOTHING, db_column='countryid', blank=True, null=True)
    stateid = models.ForeignKey('Regions', models.DO_NOTHING, db_column='stateid', blank=True, null=True)
    cityid = models.ForeignKey('Regions', models.DO_NOTHING, db_column='cityid', blank=True, null=True)
    addressline1 = models.CharField(max_length=255, blank=True, null=True)
    addressline2 = models.CharField(max_length=255, blank=True, null=True)
    postcode = models.CharField(max_length=255, blank=True, null=True)
    addresstype = models.CharField(max_length=8, blank=True, null=True)
    createdat = models.DateTimeField()
    updatedat = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'address'


class Anomalies(models.Model):
    id = models.BigIntegerField(primary_key=True)
    name = models.CharField(max_length=255, blank=True, null=True)
    description = models.CharField(max_length=255, blank=True, null=True)
    qualifier = models.CharField(max_length=255, blank=True, null=True)
    message = models.CharField(max_length=255, blank=True, null=True)
    createdat = models.DateTimeField()
    updatedat = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'anomalies'


class AnomalyResolutions(models.Model):
    id = models.BigIntegerField(primary_key=True)
    companyid = models.ForeignKey('Companies', models.DO_NOTHING, db_column='companyid', blank=True, null=True)
    anomalyid = models.ForeignKey(Anomalies, models.DO_NOTHING, db_column='anomalyid', blank=True, null=True)
    resolutionid = models.ForeignKey('Resolutions', models.DO_NOTHING, db_column='resolutionid', blank=True, null=True)
    alwaysask = models.BooleanField()
    createdat = models.DateTimeField()
    updatedat = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'anomaly_resolutions'


class CModelVariables(models.Model):
    id = models.BigIntegerField(primary_key=True)
    name = models.CharField(max_length=255, blank=True, null=True)
    description = models.CharField(max_length=255, blank=True, null=True)
    company_model_variable_type = models.CharField(max_length=5, blank=True, null=True)
    createdat = models.DateTimeField()
    updatedat = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'c_model_variables'


class CmodelResults(models.Model):
    id = models.BigIntegerField(primary_key=True)
    savepath = models.CharField(max_length=255, blank=True, null=True)
    companydatasourceid = models.ForeignKey('CompanyDataSources', models.DO_NOTHING, db_column='companydatasourceid', blank=True, null=True)
    modeltypeid = models.ForeignKey('ModelTypes', models.DO_NOTHING, db_column='modeltypeid', blank=True, null=True)
    createdat = models.DateTimeField()
    updatedat = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'cmodel_results'


class Companies(models.Model):
    id = models.BigIntegerField(primary_key=True)
    name = models.CharField(max_length=255, blank=True, null=True)
    email = models.CharField(max_length=255, blank=True, null=True)
    website = models.CharField(max_length=255, blank=True, null=True)
    company_type = models.CharField(max_length=8, blank=True, null=True)
    industryid = models.ForeignKey('Industries', models.DO_NOTHING, db_column='industryid', blank=True, null=True)
    revenue_types = models.JSONField(blank=True, null=True)
    projectiontypes = models.JSONField(blank=True, null=True)
    currencyid = models.ForeignKey('Currencies', models.DO_NOTHING, db_column='currencyid', blank=True, null=True)
    fiscal_start_month = models.CharField(max_length=9, blank=True, null=True)
    status = models.CharField(max_length=8, blank=True, null=True)
    createdat = models.DateTimeField()
    updatedat = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'companies'


class CompanyDataSources(models.Model):
    id = models.BigIntegerField(primary_key=True)
    companyid = models.ForeignKey(Companies, models.DO_NOTHING, db_column='companyid', blank=True, null=True)
    datasourceid = models.ForeignKey('DataSources', models.DO_NOTHING, db_column='datasourceid', blank=True, null=True)
    providertoken = models.CharField(max_length=255, blank=True, null=True)
    client_mapped_variables = models.JSONField(blank=True, null=True)
    createdat = models.DateTimeField()
    updatedat = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'company_data_sources'


class Currencies(models.Model):
    id = models.BigIntegerField(primary_key=True)
    name = models.CharField(max_length=255, blank=True, null=True)
    status = models.CharField(max_length=8, blank=True, null=True)
    description = models.CharField(max_length=255, blank=True, null=True)
    createdat = models.DateTimeField()
    updatedat = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'currencies'


class DataSources(models.Model):
    id = models.BigIntegerField(primary_key=True)
    name = models.CharField(max_length=255, blank=True, null=True)
    link = models.CharField(max_length=255, blank=True, null=True)
    status = models.CharField(max_length=8, blank=True, null=True)
    description = models.CharField(max_length=255, blank=True, null=True)
    createdat = models.DateTimeField()
    updatedat = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'data_sources'


class DealOwners(models.Model):
    id = models.BigIntegerField(primary_key=True)
    companyid = models.ForeignKey(Companies, models.DO_NOTHING, db_column='companyid')
    name = models.CharField(max_length=255)
    email = models.CharField(max_length=255, blank=True, null=True)
    company_identifier = models.CharField(max_length=255, blank=True, null=True)
    on_target_earnings = models.FloatField()
    ramp_status = models.CharField(max_length=10, blank=True, null=True)
    ramp_date = models.DateTimeField(blank=True, null=True)
    number_of_lead_processors = models.IntegerField()
    createdat = models.DateTimeField()
    updatedat = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'deal_owners'


class Industries(models.Model):
    id = models.BigIntegerField(primary_key=True)
    name = models.CharField(max_length=255, blank=True, null=True)
    status = models.CharField(max_length=8, blank=True, null=True)
    description = models.CharField(max_length=255, blank=True, null=True)
    createdat = models.DateTimeField()
    updatedat = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'industries'


class Languages(models.Model):
    id = models.BigIntegerField(primary_key=True)
    locale = models.CharField(max_length=20, blank=True, null=True)
    createdat = models.DateTimeField()
    updatedat = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'languages'


class ModelTypes(models.Model):
    id = models.BigIntegerField(primary_key=True)
    name = models.CharField(max_length=255, blank=True, null=True)
    description = models.CharField(max_length=255, blank=True, null=True)
    createdat = models.DateTimeField()
    updatedat = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'model_types'


class ProjectionTypes(models.Model):
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=255, blank=True, null=True)
    description = models.CharField(max_length=255, blank=True, null=True)
    status = models.CharField(max_length=8, blank=True, null=True)
    createdat = models.DateTimeField()
    updatedat = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'projection_types'


class Regionedges(models.Model):
    id = models.BigIntegerField(primary_key=True)
    parentregionid = models.ForeignKey('Regions', models.DO_NOTHING, db_column='parentregionid', blank=True, null=True)
    childregionid = models.ForeignKey('Regions', models.DO_NOTHING, db_column='childregionid', blank=True, null=True)
    createdat = models.DateTimeField()
    updatedat = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'regionedges'


class Regionmetas(models.Model):
    id = models.BigIntegerField(primary_key=True)
    languageid = models.ForeignKey(Languages, models.DO_NOTHING, db_column='languageid', blank=True, null=True)
    regionid = models.ForeignKey('Regions', models.DO_NOTHING, db_column='regionid', blank=True, null=True)
    name = models.CharField(max_length=255, blank=True, null=True)
    preposition = models.TextField(blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    createdat = models.DateTimeField()
    updatedat = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'regionmetas'


class Regions(models.Model):
    id = models.BigIntegerField(primary_key=True)
    name = models.CharField(max_length=255, blank=True, null=True)
    isocode2l = models.CharField(max_length=255, blank=True, null=True)
    isocode3l = models.CharField(max_length=255, blank=True, null=True)
    regiontype = models.IntegerField(blank=True, null=True)
    currency = models.CharField(max_length=255, blank=True, null=True)
    statistacode = models.CharField(max_length=255, blank=True, null=True)
    createdat = models.DateTimeField()
    updatedat = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'regions'


class Regionslugs(models.Model):
    id = models.BigIntegerField(primary_key=True)
    slug = models.CharField(max_length=255, blank=True, null=True)
    languageid = models.ForeignKey(Languages, models.DO_NOTHING, db_column='languageid', blank=True, null=True)
    regionid = models.ForeignKey(Regions, models.DO_NOTHING, db_column='regionid', blank=True, null=True)
    createdat = models.DateTimeField()
    updatedat = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'regionslugs'


class Resolutions(models.Model):
    id = models.BigIntegerField(primary_key=True)
    name = models.CharField(max_length=255, blank=True, null=True)
    description = models.CharField(max_length=255, blank=True, null=True)
    createdat = models.DateTimeField()
    updatedat = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'resolutions'


class RevenueTypes(models.Model):
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=255, blank=True, null=True)
    description = models.CharField(max_length=255, blank=True, null=True)
    status = models.CharField(max_length=8, blank=True, null=True)
    createdat = models.DateTimeField()
    updatedat = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'revenue_types'
