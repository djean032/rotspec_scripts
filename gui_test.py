import os
import sys
import sqlite3
import sqlalchemy as db
import timeit

from io import StringIO
from PySide6.QtWidgets import QApplication, QTableView, QMainWindow
from PySide6.QtCore import QSize, Qt
from PySide6.QtSql import QSqlDatabase, QSqlQueryModel, QSqlTableModel
import pyqtgraph as pg


basedir = os.path.dirname(__file__)


def create_connection(db_path: str, file_name: str):
    engine = db.create_engine(f"sqlite:///{basedir}{db_path}")
    connection = engine.connect()
    tables = ["cat", "lin", "res", "spe"]
    table_names = [f"{file_name}_{table}" for table in tables]
    metadata = db.MetaData()
    cat = db.Table(table_names[0], metadata, autoload_with=engine)
    lin = db.Table(table_names[1], metadata, autoload_with=engine)
    res = db.Table(table_names[2], metadata, autoload_with=engine)
    spe = db.Table(table_names[3], metadata, autoload_with=engine)
    return cat, lin, res, spe, engine, connection


begin = timeit.default_timer()
cat, lin, res, spe, engine, connection = create_connection(
    "/z24pdn_dyad.sqlite", "z24pdn_dyad"
)
stmt = db.select(spe)
result = connection.execute(stmt).fetchall()
new_result = list(zip(*result))
meas_freq = new_result[0]
meas_int = new_result[1]
end = timeit.default_timer()
print(f"Time: {end - begin}")

app = pg.mkQApp("Spectra Viewer")

win = pg.GraphicsLayoutWidget(show=True, title="Spectra")
win.resize(800,600)
win.setWindowTitle('Spectra Viewer')

p1 = win.addPlot(title="Exp Spectrum", x=meas_freq, y=meas_int)
p1.enableAutoRange('y', 0.95)
p1.setAutoVisible(y=True)

if __name__ == '__main__':
    pg.exec()
