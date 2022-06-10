from rest_framework import serializers

class TargetSerializer(serializers.Serializer):
    group = serializers.CharField()
    percentage_growth = serializers.IntegerField()
    model_type = serializers.CharField()
