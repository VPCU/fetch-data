import os
os.environ['kdconfig']='../../database.yaml'
from qsdata import kddb
import argparse

def gen_sql(y, m):
    yy = y
    mm = m
    if m == 12:
        yy = y + 1
        mm = 1
    else:
        mm = m + 1
    return f"select * from kd_news_for_em where date_time >= '{y}-{m}-01' and date_time < '{yy}-{mm}-01'"

# sql = "select * from kd_news_for_em where date_time >= '2021-01-01' and date_time < '2021-02-01'"
# news = kddb.read_db(sql,db_from="to_pull_news")
# news

parser = argparse.ArgumentParser()
parser.add_argument('-y', help='year')
parser.add_argument('-m', help='month')
args = parser.parse_args()

y = int(args.y)
m = int(args.m)

news = kddb.read_db(gen_sql(y, m),db_from="to_pull_news")

news.to_pickle(f"{y}-{m}.pkl")