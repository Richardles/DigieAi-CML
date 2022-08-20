# import mariadb
import sys
import csv
import MySQLdb
from datetime import datetime
from datetime import timedelta

# Instantiate mariadb Connection
# try:
#    conn = mariadb.connect(
#       host="localhost",
#       port=3306,
#       user="root",
#       password="",
#       database="ai_digi_test"
#     )
# except mariadb.Error as e:
#    print(f"Error connecting to the database: {e}")
#    sys.exit(1)
 
connection = MySQLdb.connect(
    host="localhost",
    user="root",
    passwd="",
    db="ai_digi_test"
)
 
cur = connection.cursor()
cur.execute("select database();")
db = cur.fetchone()
 
if db:
    print("You're connected to database: ", db)
else:
    print('Not connected.')

# Get Cursor
# cur = conn.cursor()

# Use Connection
# ...
# cur.execute(
#     "SELECT first_name,last_name FROM employees WHERE first_name=?", 
#     (some_name,))

try:
   # cur.execute("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'udd_metrics'")
   cur.execute("SELECT * FROM udd_metrics ORDER BY metric_datetime")
except e:
   print(e)

rows = cur.fetchall()
i = 0
# for row in rows:
#     print(f'{i} - {row}')
#     i+=1

# today = datetime.now()
# start = today - timedelta(days=15)
# print(today)
# print()
# print(start)

# try:
#    # cur.execute("SELECT metric_datetime, metric_value FROM udd_metrics ORDER BY metric_datetime")
#    cur.execute(f"SELECT * FROM udd_metrics WHERE metric_name = 'Disk/File System/[/dev/mapper/rhel-home (/home)]/percent full' AND metric_datetime < '{today}' AND metric_datetime > '{start}' ORDER BY metric_datetime")
#    # cur.execute(
#    #    "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'udd_metrics'")
# except e:
#    print(e)

# rows = cur.fetchall()

with open('test_data2.csv', 'w', newline='') as out:
   csv_out=csv.writer(out)
   csv_out.writerows(rows)

# Print Result-set
# for (first_name, last_name) in cur:
#     print(f"First Name: {first_name}, Last Name: {last_name}")

# Close Connection
connection.close()