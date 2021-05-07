import sqlite3

conn = sqlite3.connect('music.sqlite')
cur = conn.cursor()

cur.execute('DROP TABLE IF EXISTS tracks')
cur.execute('CREATE TABLE tracks (title TEXT, plays INTEGER)')

cur.execute('INSERT INTO tracks (title, plays) VALUES (?,?)', ('Happy Birthday', 500))
cur.execute('INSERT INTO tracks (title, plays) VALUES (?,?)', ('Twinkle Twinkle Little Star', 691))

conn.commit()

cur.execute('SELECT title, plays FROM tracks WHERE title LIKE \'%Star\'')

for row in cur:
    print(row)

conn.close()