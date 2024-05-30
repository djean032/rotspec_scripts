from fileops import Experiment as Experiment
from pathlib import Path
import sqlite3
import numpy as np
import matplotlib.pyplot as plt


def create_db(databases, table_names, db_filepath):
    con = sqlite3.connect(db_filepath)
    for db, table_name in zip(databases, table_names):
        db.to_sql(table_name, con, if_exists="replace", index=False)
    con.close()


def init_experiment(file: str, spectra_file: Path):
    exp = Experiment(file, spectra_file)
    cat_df = exp.cat_dataframe
    res_df = exp.res_dataframe
    lin_df = exp.lin_dataframe
    spec_df = exp.spec_dataframe
    databases = [cat_df, res_df, lin_df, spec_df]
    table_names = [file + "_cat", file + "_res", file + "_lin", file + "_spe"]
    db_filepath = file + ".sqlite"
    create_db(databases, table_names, db_filepath)
    return db_filepath


def peak_regions(cfrq):
    def get_peak_region(frq):
        idx = (np.abs(frq - cfrq)).argmin()
        return freq2[idx - 75 : idx + 75], intensity[idx - 75 : idx + 75]

    return get_peak_region


def load_spectrum(path):
    x, y = np.loadtxt(
        path,
        usecols=(0, 1),
        skiprows=3,
        unpack=True,
    )
    return x, y


def file_stitcher(x_lower, y_lower, x_upper, y_upper, stitch_freq):
    l_ind = np.where(x_lower < stitch_freq)[0]
    u_ind = np.where(x_upper >= stitch_freq)[0]
    x_l = x_lower[l_ind]
    y_l = y_lower[l_ind]
    x_u = x_upper[u_ind]
    y_u = y_upper[u_ind]
    x = np.concatenate((x_l, x_u))
    y = np.concatenate((y_l, y_u))
    return x, y


path1 = "./2022-09-30-12_08_(3-cyano)methylenecyclopropane_235-345GHz_12mtorr.spe"
path2 = "./2022-09-27-10_10_(3-cyano)methylenecyclopropane_340-500GHz_12mtorr.spe"
x_low, y_low = load_spectrum(path1)
x_up, y_up = load_spectrum(path2)
x_stitched, y_stitched = file_stitcher(x_low, y_low, x_up, y_up, 345000)

init_experiment(
    "cyanomethcycloprop_gs", Path("./cyanomethylenecyclopropane_235-500GHz_bin.spe")
)
con2 = sqlite3.connect("z24pdn_dyad.sqlite")
spe_query = f"SELECT * FROM z24pdn_dyad_spe "
cursor = con2.cursor()
spec = cursor.execute(spe_query).fetchall()
freq2 = np.array([line[0] for line in spec])
intensity2 = np.array([line[1] for line in spec])
freq = x_stitched
intensity = y_stitched
min_freq = max(np.min(freq2), np.min(freq))
max_freq = min(np.max(freq2), np.max(freq))
cat_query = f"SELECT * FROM z24pdn_dyad_cat  WHERE Ka = 0 AND `Ka'` = 0 AND Frequency > {min_freq} AND Frequency < {max_freq} and V = `V'`"
cursor = con2.cursor()
calc_lines2 = cursor.execute(cat_query).fetchall()
calc_freq2 = np.array([line[0] for line in calc_lines2])
con2.close()

con = sqlite3.connect("cyanomethcycloprop_gs.sqlite")
cat_query = f"SELECT * FROM cyanomethcycloprop_gs_cat WHERE Ka = 0 AND `Ka'` = 0 AND Frequency > {min_freq} AND Frequency < {max_freq} and V = `V'`"
cursor = con.cursor()
calc_lines = cursor.execute(cat_query).fetchall()
calc_freq = np.array([line[0] for line in calc_lines])
con.close()


left_limit = 2668610
right_limit = 2668772

# Extract peak so we can match filter later
# -1 due to Fortran indexing
reference_peak = freq[left_limit - 1 : right_limit + 1]

freq_regions = []
intensity_regions = []

for i in calc_freq2:
    idx = (np.abs(freq - i)).argmin()
    freq_regions.append(freq[idx - 75 : idx + 75])
    intensity_regions.append(intensity[idx - 75 : idx + 75])
print(intensity_regions)
stacked_intensity = np.sum(np.vstack(intensity_regions), axis=0)
# stacked_intensity2 = np.sum(np.vstack(intensity_regions2), axis=0)
x = np.arange(-75, 75, 1)
print(freq_regions[30])
plt.plot(x, stacked_intensity)
plt.show()
"""
file = "cyanomethcycloprop_gs"
spectra_file = "./Z24PDN_135-375GHz_bin.spe"
exp = Experiment(file, spectra_file)
cat_df = exp.cat_dataframe
res_df = exp.res_dataframe
lin_df = exp.lin_dataframe
spec_df = exp.spec_dataframe
databases = [cat_df, res_df, lin_df, spec_df]
table_names = [file + "_cat", file + "_res", file + "_lin", file + "_spe"]
db_filepath = file + ".sqlite"
create_db(databases, table_names, db_filepath)

con = sqlite3.connect(db_filepath)
cursor = con.cursor()
sql = "SELECT * FROM cyanomethcycloprop_gs_lin"
cursor.execute(sql)
con.close()
print("Time to query: ", end - begin)

"""
