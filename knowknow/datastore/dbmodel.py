from peewee import *

from pathlib import Path

db = SqliteDatabase( Path(__file__).parent.joinpath( '../..', 'wos.db') )
#db = PostgresqlDatabase('wos', user='postgres', password='postgres',
#                           host='0.0.0.0', port=5432)

class JSONField(TextField):
    """Store JSON data in a TextField."""
    def python_value(self, value):
        if value is not None:
            return json.loads(value)

    def db_value(self, value):
        if value is not None:
            return json.dumps(value)

class Doc(Model):
    title = TextField()
    first_author = CharField(index=True)
    journal = CharField(index=True)
    year = IntegerField(index=True)
    citcnt_external = IntegerField()

    class Meta:
        database = db

class Cit(Model):
    doc = ForeignKeyField(Doc, backref='citations')
    full_ref = CharField(index=True)
    author = CharField(index=True)
    year = IntegerField(index=True)
    
    class Meta:
        database = db

class Keyword(Model):
    doc = ForeignKeyField(Doc, backref='keywords')
    text = TextField(index=True)

    class Meta:
        database = db