# Generated by Django 3.1.3 on 2021-02-02 14:00

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0010_auto_20210202_1358'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='stockholding',
            name='_last_trading_price',
        ),
        migrations.RemoveField(
            model_name='stockholding',
            name='net_change',
        ),
        migrations.RemoveField(
            model_name='stockholding',
            name='p_n_l',
        ),
    ]
