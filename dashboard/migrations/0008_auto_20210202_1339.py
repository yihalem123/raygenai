# Generated by Django 3.1.3 on 2021-02-02 13:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0007_stockholding_average_cost'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='stockholding',
            name='average_cost',
        ),
        migrations.AddField(
            model_name='stockholding',
            name='last_trading_price',
            field=models.FloatField(default=0),
        ),
        migrations.AddField(
            model_name='stockholding',
            name='net_change',
            field=models.FloatField(default=0),
        ),
        migrations.AddField(
            model_name='stockholding',
            name='p_n_l',
            field=models.FloatField(default=0),
        ),
    ]
