import math
import sys
from pathlib import Path
from struct import unpack
from typing import Tuple

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from matplotlib.legend_handler import HandlerPatch
from numpy import linspace
from pandas import DataFrame

mpl.rc("text", usetex=True)
plt.style.use(["science", "ieee"])


class Experiment:
    def __init__(
        self,
        file_name: str,
        spectra_filepath: Path,
    ):
        self.file_name = file_name
        self.spectra_filepath = spectra_filepath
        (
            self.file_path_res,
            self.file_path_lin,
            self.file_path_cat,
            self.file_path_par,
        ) = self.rel_file_path()
        self.num_lines = self.get_line_num()
        self.read_lin()
        self.read_res()
        self.read_cat()
        self.map_strings_to_numeric()
        self.read_spectra()
        self.df_error = self.res_dataframe.copy(deep=True)

    def get_line_num(self):
        with open(self.file_path_par) as f:
            lines = f.readlines()
        self.num_lines = lines[1].split()[1]

    def rel_file_path(self):
        base_path = Path(sys.argv[0]).resolve().parent
        file_name_res = self.file_name + ".res"
        file_name_lin = self.file_name + ".lin"
        file_name_cat = self.file_name + ".cat"
        file_name_par = self.file_name + ".par"
        file_path_res = (base_path / file_name_res).resolve()
        file_path_lin = (base_path / file_name_lin).resolve()
        file_path_cat = (base_path / file_name_cat).resolve()
        file_path_par = (base_path / file_name_par).resolve()
        return file_path_res, file_path_lin, file_path_cat, file_path_par

    def read_lin(self):
        self.lin_dataframe = pd.read_fwf(
            self.file_path_lin,
            header=None,
            names=[
                "J",
                "Ka",
                "Kc",
                "v",
                "J'",
                "Ka'",
                "Kc'",
                "v'",
                "01",
                "02",
                "03",
                "04",
                "Measured Frequency",
                "Max Error",
                "Relative Weight",
            ],
            widths=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 16, 13, 10],
        )

    def read_spectra(self):
        data = Path(self.spectra_filepath).read_bytes()
        c = unpack("72s", data[0:72])[0]
        iday, imon, iyear = unpack("3h", data[72:78])
        ihour, imin, isec = unpack("3h", data[78:84])
        lamp = unpack("6s", data[84:90])[0]
        vkmst, vkmend = unpack("2f", data[90:98])
        grid = unpack("f", data[98:102])[0]
        sample = unpack("20s", data[102:122])[0]
        sampre = unpack("f", data[122:126])[0]
        gain, timec, phase = unpack("3f", data[126:138])
        scansp = unpack("6s", data[138:144])[0]
        pps = unpack("f", data[144:148])[0]
        frmod, frampl = unpack("2f", data[148:156])
        neg1 = unpack("h", data[156:158])[0]
        npts = abs(unpack("i", data[158:162])[0])
        ipsec_string = str(npts) + "i"
        ipsec = unpack(ipsec_string, data[162 : 162 + npts * 4])
        fstart, fend, fincr = unpack("3d", data[162 + npts * 4 : 162 + npts * 4 + 24])
        freq_list = linspace(fstart, fend, npts)
        spectra = dict(zip(freq_list, ipsec))
        self.spec_dataframe = pd.DataFrame(
            spectra.items(), columns=["Frequency", "Intensity"]
        )

    def clean_lin(self, threshold: float):
        self.df_error.drop(["Line Number"], axis=1, inplace=True)
        self.df_error[self.df_error["Error"] != "UNFITTD"]
        self.lin_dataframe.apply(pd.to_numeric)
        self.lin_dataframe.drop_duplicates(
            ["J", "Ka", "Kc", "J'", "Ka'", "Kc'"], keep="last", inplace=True
        )
        self.df_error["blend O-C"] = self.df_error["blend O-C"].fillna(
            self.df_error["O-C"]
        )
        self.df_error["blend weight"] = self.df_error["blend weight"].fillna(
            self.df_error["Error"]
        )
        self.df_error["Relative Error Blend"] = abs(self.df_error["blend O-C"]) / abs(
            self.df_error["Error"]
        )
        df_lin_res = self.lin_dataframe.join(
            self.df_error["Relative Error Blend"], how="right"
        )
        df_remove = df_lin_res[df_lin_res["Relative Error Blend"] > threshold]
        removal_list = df_remove.index.tolist()
        with open(self.file_path_lin) as f:
            lines = f.readlines()
        keys = np.arange(0, len(lines), 1)
        line_dict = dict(zip(keys, lines))
        for i in removal_list:
            line_dict.pop(i)
        with open(self.file_path_lin, "w") as output:
            for i in line_dict.values():
                output.write(str(i))

    def read_res(self):
        line_width = len(str(self.num_lines)) + 1
        self.res_dataframe = pd.read_fwf(
            self.file_path_res,
            header=None,
            skiprows=7,
            names=[
                "Line Number",
                "J",
                "Ka",
                "Kc",
                "v",
                "J'",
                "Ka'",
                "Kc'",
                "v'",
                "Obs-Freq",
                "O-C",
                "Error",
                "blend O-C",
                "blend weight",
            ],
            widths=[
                line_width,
                3,
                3,
                3,
                3,
                5,
                3,
                3,
                3,
                24,
                9,
                7,
                9,
                5,
            ],
        )
        last_row = self.res_dataframe[
            self.res_dataframe["Line Number"].str.contains("---", na=False)
        ].index[0]
        self.res_dataframe = self.res_dataframe.iloc[:last_row]
        self.res_dataframe = self.res_dataframe[
            self.res_dataframe["Error"] != "UNFITTD"
        ]
        self.delta_values()
        self.categorize_trans_type("Delta Ka", "Delta Kc")
        self.categorize_branch("Delta J")

    def categorize_trans_type(self, column_name1: str, column_name2: str):
        column1 = self.res_dataframe[column_name1].astype(int)
        column2 = self.res_dataframe[column_name2].astype(int)
        transition_types = []

        for Ka, Kc in zip(column1, column2):
            if Kc % 2 == 0:
                transition_types.append("c-type")
            elif Ka % 2 == 0:
                transition_types.append("a-type")
            else:
                transition_types.append("b-type")
        self.res_dataframe["Transition Type"] = transition_types

    def categorize_branch(self, column_name: str):
        column = self.res_dataframe[column_name]
        branches = []

        for value in column:
            if value == 1:
                branches.append("R-Branch")
            elif value == -1:
                branches.append("P-Branch")
            else:
                branches.append("Q-Branch")
        self.res_dataframe["Branch"] = branches

    def delta_values(self):
        self.res_dataframe = self.res_dataframe.apply(
            pd.to_numeric, errors="coerce"
        ).fillna(self.res_dataframe)
        self.res_dataframe["Delta J"] = (
            self.res_dataframe["J"] - self.res_dataframe["J'"]
        )
        self.res_dataframe["Delta Ka"] = (
            self.res_dataframe["Ka"] - self.res_dataframe["Ka'"]
        )
        self.res_dataframe["Delta Kc"] = (
            self.res_dataframe["Kc"] - self.res_dataframe["Kc'"]
        )
        self.res_dataframe["Delta_v"] = (
            self.res_dataframe["v"] - self.res_dataframe["v'"]
        )

    def categorize_error(self):
        self.res_dataframe["blend O-C"].fillna(
            value=self.res_dataframe["O-C"], inplace=True
        )
        self.res_dataframe["blend O-C"] = self.res_dataframe["blend O-C"].abs()
        self.res_dataframe["blend O-C/error"] = (
            self.res_dataframe["blend O-C"].abs() / self.res_dataframe["Error"]
        )
        self.res_dataframe["blend O-C/error"] = self.res_dataframe[
            "blend O-C/error"
        ].apply(lambda x: math.ceil(x))
        self.res_dataframe["blend O-C/error"].replace(0, 1, inplace=True)
        self.res_dataframe = self.res_dataframe.sort_values(
            by=["blend O-C/error", "Obs-Freq"], ascending=[True, False]
        )

    # Do I need this???
    def split_branches(self) -> Tuple[DataFrame, DataFrame, DataFrame]:
        df_r = self.res_dataframe.loc[self.res_dataframe["Branch"] == "R-Branch"]
        df_p = self.res_dataframe.loc[self.res_dataframe["Branch"] == "P-Branch"]
        df_q = self.res_dataframe.loc[self.res_dataframe["Branch"] == "Q-Branch"]
        return (df_r, df_p, df_q)

    def split_lines(
        self,
    ) -> Tuple[
        DataFrame,
        DataFrame,
        DataFrame,
        DataFrame,
        DataFrame,
        DataFrame,
        DataFrame,
        DataFrame,
        DataFrame,
    ]:
        df_ar = self.res_dataframe.loc[
            (self.res_dataframe["Transition Type"] == "a-type")
            & (self.res_dataframe["Branch"] == "R-Branch")
        ]
        df_br = self.res_dataframe.loc[
            (self.res_dataframe["Transition Type"] == "b-type")
            & (self.res_dataframe["Branch"] == "R-Branch")
        ]
        df_cr = self.res_dataframe.loc[
            (self.res_dataframe["Transition Type"] == "c-type")
            & (self.res_dataframe["Branch"] == "R-Branch")
        ]

        df_ap = self.res_dataframe.loc[
            (self.res_dataframe["Transition Type"] == "a-type")
            & (self.res_dataframe["Branch"] == "P-Branch")
        ]
        df_bp = self.res_dataframe.loc[
            (self.res_dataframe["Transition Type"] == "b-type")
            & (self.res_dataframe["Branch"] == "P-Branch")
        ]
        df_cp = self.res_dataframe.loc[
            (self.res_dataframe["Transition Type"] == "c-type")
            & (self.res_dataframe["Branch"] == "P-Branch")
        ]

        df_aq = self.res_dataframe.loc[
            (self.res_dataframe["Transition Type"] == "a-type")
            & (self.res_dataframe["Branch"] == "Q-Branch")
        ]
        df_bq = self.res_dataframe.loc[
            (self.res_dataframe["Transition Type"] == "b-type")
            & (self.res_dataframe["Branch"] == "Q-Branch")
        ]
        df_cq = self.res_dataframe.loc[
            (self.res_dataframe["Transition Type"] == "c-type")
            & (self.res_dataframe["Branch"] == "Q-Branch")
        ]

        return (df_ar, df_br, df_cr, df_ap, df_bp, df_cp, df_aq, df_bq, df_cq)

    def get_IR(self, df: DataFrame) -> DataFrame:
        df_IR = df.loc[df["Delta_v"] != 0]
        return df_IR

    def get_rot(self, df: DataFrame) -> DataFrame:
        df_rot = df.loc[df["Delta_v"] == 0]
        return df_rot

    def read_cat(self):
        self.cat_dataframe: DataFrame = pd.read_fwf(  # type: ignore
            self.file_path_cat,
            header=None,
            names=[
                "Frequency",
                "Error",
                "Integrated Intensity",
                "Deg of Freedom",
                "Lower State Energy (cm-1)",
                "Upper State Deg",
                "Tag",
                "QNFMT",
                "N",
                "Ka",
                "Kc",
                "V",
                "J",
                "F",
                "N'",
                "Ka'",
                "Kc'",
                "V'",
                "J'",
                "F'",
            ],
            widths=[13, 8, 8, 2, 10, 3, 7, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        )

    def map_strings_to_numeric(self):
        self.cat_dataframe.astype(str)
        self.cat_dataframe.drop(
            columns=["J", "F", "J'", "F'"],
            inplace=True,
        )
        quantum_list = [
            "N",
            "Ka",
            "Kc",
            "V",
            "N'",
            "Ka'",
            "Kc'",
            "V'",
        ]
        for column_name in quantum_list:
            column = self.cat_dataframe[column_name]
            numeric_values = []

            for value in column:
                if str(value).isdigit():
                    numeric_values.append(int(value))
                    continue

                if pd.isnull(value):  # Skip empty values
                    numeric_values.append(value)
                    continue

                value = str(value)
                letter = value[0]  # Extract the first character (letter)
                number = int(
                    value[1:]
                )  # Extract the remaining characters as the number
                letter_value = int(
                    ord(letter.upper()) - 65
                )  # Convert letter to value (A=0, B=1, ...)
                if letter.islower():  # Check if the letter is lowercase
                    numeric_value = int(
                        -10 * (letter_value + 1) - number
                    )  # Calculate the final numeric value for lowercase letter
                else:
                    numeric_value = int(
                        100 + letter_value * 10 + number
                    )  # Calculate the final numeric value for uppercase letter
                numeric_values.append(numeric_value)

            self.cat_dataframe[column_name] = numeric_values

    # TODO: Adjust plotting to be more modular
    def plot_data_dist_rot_color(self, max_Ka: int, max_J: int):
        mpl.rc("text", usetex=True)
        plt.style.use(["science", "ieee"])
        text_x_pos = max_Ka - 22
        text_y_pos = max_J - 10
        self.categorize_error()
        self.res_dataframe = self.res_dataframe.sort_values(
            by=["Transition Type"], ascending=True
        )
        dataframes = self.split_branches()
        dataframes = tuple(map(self.get_rot, dataframes))
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        for i in range(0, 2):
            condition = dataframes[i]["blend O-C/error"] >= 4
            df_high_error = dataframes[i][condition]
            df_normal = dataframes[i][~condition]
            ax[0].scatter(
                df_normal["Ka'"],
                df_normal["J'"],
                s=4 * df_normal["blend O-C/error"],
                facecolors=df_normal["Transition Type"].map(
                    {
                        "a-type": "#0000FF65",
                        "b-type": "#00800045",
                        "c-type": "#FFA50020",
                    }
                ),
                edgecolors=df_normal["Transition Type"].map(
                    {"a-type": "blue", "b-type": "green", "c-type": "orange"}
                ),
                linewidth=0.8,
            )
            ax[0].scatter(
                df_high_error["Ka'"],
                df_high_error["J'"],
                s=4 * df_high_error["blend O-C/error"],
                facecolors="#FF000065",
                edgecolors="red",
                linewidth=0.8,
            )
        condition_q = dataframes[2]["blend O-C/error"] >= 4
        df_high_error_q = dataframes[2][condition_q]
        df_normal_q = dataframes[2][~condition_q]
        ax[1].scatter(
            df_high_error_q["Ka'"],
            df_high_error_q["J'"],
            s=4 * df_high_error_q["blend O-C/error"],
            facecolors="#FF000065",
            edgecolors="red",
            linewidth=0.8,
        )
        ax[1].scatter(
            df_normal_q["Ka'"],
            df_normal_q["J'"],
            s=4 * df_normal_q["blend O-C/error"],
            facecolors=df_normal_q["Transition Type"].map(
                {"a-type": "#0000FF65", "b-type": "#00800045", "c-type": "#FFA50020"}
            ),
            edgecolors=df_normal_q["Transition Type"].map(
                {"a-type": "blue", "b-type": "green", "c-type": "orange"}
            ),
            linewidth=0.8,
        )

        ax[0].axline((0, 0), slope=1, color="lightgrey", linewidth=1.5)
        ax[0].set_xlabel("$K_a''$", fontsize=14)
        ax[0].set_ylabel("$J''$", fontsize=14)
        ax[0].set_ylim(0, max_J)
        ax[0].set_xlim(-0.5, max_Ka)
        ax[0].set_xticks((0, 10, 20, 30, 40, 50, 60))
        ax[0].set_yticks((0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120))
        ax[0].text(text_x_pos, text_y_pos, "R-Branch", fontsize=18)

        ax[1].axline((0, 0), slope=1, color="lightgrey", linewidth=1.5)
        ax[1].set_xlabel("$K_a''$", fontsize=14)
        ax[1].set_ylabel("$J''$", fontsize=14)
        ax[1].set_ylim(0, max_J)
        ax[1].set_xlim(-0.5, max_Ka)
        ax[1].set_xticks((0, 10, 20, 30, 40, 50, 60))
        ax[1].set_yticks((0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120))
        ax[1].text(text_x_pos, text_y_pos, "Q-Branch", fontsize=18)

        c = [
            mpatches.Circle(
                (0.5, 0.5),
                radius=0.25,
                facecolor="#0000FF65",
                edgecolor="blue",
                label="a-type",
            ),
            mpatches.Circle(
                (0.5, 0.5),
                radius=0.25,
                facecolor="#00800045",
                edgecolor="green",
                label="b-type",
            ),
            mpatches.Circle(
                (0.5, 0.5),
                radius=0.25,
                facecolor="#FFA50065",
                edgecolor="orange",
                label="c-type",
            ),
            mpatches.Circle(
                (0.5, 0.5),
                radius=0.25,
                facecolor="#FF000065",
                edgecolor="red",
                label=r"$>$ $3\sigma$",
            ),
        ]
        texts = ["a-type", "b-type", "c-type", r"$>$ $3\sigma$"]
        legend = ax[0].legend(
            c,
            texts,
            loc="lower right",
            fontsize=8,
            frameon=True,
            edgecolor="black",
            framealpha=1,
            borderaxespad=0,
            fancybox=False,
            handler_map={mpatches.Circle: HandlerEllipse()},
        )
        frame = legend.get_frame()
        frame.set_linewidth(0.5)

        legend1 = ax[1].legend(
            c,
            texts,
            loc="lower right",
            fontsize=8,
            frameon=True,
            edgecolor="black",
            framealpha=1,
            borderaxespad=0,
            fancybox=False,
            handler_map={mpatches.Circle: HandlerEllipse()},
        )
        frame1 = legend1.get_frame()
        frame1.set_linewidth(0.5)
        file_path = self.file_name + ".jpg"
        fig.savefig(file_path, dpi=800)

    def plot_data_dist_IR_color(self, max_Ka: int, max_J: int):
        mpl.rc("text", usetex=True)
        plt.style.use(["science", "ieee"])
        text_x_pos = max_Ka - 22
        text_y_pos = max_J - 10
        self.categorize_error()
        self.res_dataframe = self.res_dataframe.sort_values(
            by=["Transition Type"], ascending=True
        )
        dataframes = self.split_branches()
        dataframes = tuple(map(self.get_IR, dataframes))
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        for i in range(0, 2):
            condition = dataframes[i]["blend O-C/error"] >= 4
            df_high_error = dataframes[i][condition]
            df_normal = dataframes[i][~condition]
            ax[0].scatter(
                df_normal["Ka'"],
                df_normal["J'"],
                s=4 * df_normal["blend O-C/error"],
                facecolors=df_normal["Transition Type"].map(
                    {
                        "a-type": "#0000FF65",
                        "b-type": "#00800045",
                        "c-type": "#FFA50020",
                    }
                ),
                edgecolors=df_normal["Transition Type"].map(
                    {"a-type": "blue", "b-type": "green", "c-type": "orange"}
                ),
                linewidth=0.8,
            )
            ax[0].scatter(
                df_high_error["Ka'"],
                df_high_error["J'"],
                s=4 * df_high_error["blend O-C/error"],
                facecolors="#FF000065",
                edgecolors="red",
                linewidth=0.8,
            )
        condition_q = dataframes[2]["blend O-C/error"] >= 4
        df_high_error_q = dataframes[2][condition_q]
        df_normal_q = dataframes[2][~condition_q]
        ax[1].scatter(
            df_high_error_q["Ka'"],
            df_high_error_q["J'"],
            s=4 * df_high_error_q["blend O-C/error"],
            facecolors="#FF000065",
            edgecolors="red",
            linewidth=0.8,
        )
        ax[1].scatter(
            df_normal_q["Ka'"],
            df_normal_q["J'"],
            s=4 * df_normal_q["blend O-C/error"],
            facecolors=df_normal_q["Transition Type"].map(
                {"a-type": "#0000FF65", "b-type": "#00800045", "c-type": "#FFA50020"}
            ),
            edgecolors=df_normal_q["Transition Type"].map(
                {"a-type": "blue", "b-type": "green", "c-type": "orange"}
            ),
            linewidth=0.8,
        )

        ax[0].axline((0, 0), slope=1, color="lightgrey", linewidth=1.5)
        ax[0].set_xlabel("$K_a''$", fontsize=14)
        ax[0].set_ylabel("$J''$", fontsize=14)
        ax[0].set_ylim(0, max_J)
        ax[0].set_xlim(-0.5, max_Ka)
        ax[0].set_xticks((0, 10, 20, 30, 40, 50, 60))
        ax[0].set_yticks((0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120))
        ax[0].text(text_x_pos, text_y_pos, "R-Branch", fontsize=18)

        ax[1].axline((0, 0), slope=1, color="lightgrey", linewidth=1.5)
        ax[1].set_xlabel("$K_a''$", fontsize=14)
        ax[1].set_ylabel("$J''$", fontsize=14)
        ax[1].set_ylim(0, max_J)
        ax[1].set_xlim(-0.5, max_Ka)
        ax[1].set_xticks((0, 10, 20, 30, 40, 50, 60))
        ax[1].set_yticks((0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120))
        ax[1].text(text_x_pos, text_y_pos, "Q-Branch", fontsize=18)

        c = [
            mpatches.Circle(
                (0.5, 0.5),
                radius=0.25,
                facecolor="#0000FF65",
                edgecolor="blue",
                label="a-type",
            ),
            mpatches.Circle(
                (0.5, 0.5),
                radius=0.25,
                facecolor="#00800045",
                edgecolor="green",
                label="b-type",
            ),
            mpatches.Circle(
                (0.5, 0.5),
                radius=0.25,
                facecolor="#FFA50065",
                edgecolor="orange",
                label="c-type",
            ),
            mpatches.Circle(
                (0.5, 0.5),
                radius=0.25,
                facecolor="#FF000065",
                edgecolor="red",
                label=r"$>$ $3\sigma$",
            ),
        ]
        texts = ["a-type", "b-type", "c-type", r"$>$ $3\sigma$"]
        legend = ax[0].legend(
            c,
            texts,
            loc="lower right",
            fontsize=8,
            frameon=True,
            edgecolor="black",
            framealpha=1,
            borderaxespad=0,
            fancybox=False,
            handler_map={mpatches.Circle: HandlerEllipse()},
        )
        frame = legend.get_frame()
        frame.set_linewidth(0.5)

        legend1 = ax[1].legend(
            c,
            texts,
            loc="lower right",
            fontsize=8,
            frameon=True,
            edgecolor="black",
            framealpha=1,
            borderaxespad=0,
            fancybox=False,
            handler_map={mpatches.Circle: HandlerEllipse()},
        )
        frame1 = legend1.get_frame()
        frame1.set_linewidth(0.5)
        file_path = self.file_name + ".jpg"
        fig.savefig(file_path, dpi=800)

    def plot_data_dist_rot_pub(self, max_Ka: int, max_J: int):
        mpl.rc("text", usetex=True)
        plt.style.use(["science", "ieee"])
        text_x_pos = max_Ka - 22
        text_y_pos = max_J - 10
        self.categorize_error()
        self.res_dataframe = self.res_dataframe.sort_values(
            by=["Transition Type"], ascending=True
        )
        dataframes = self.split_branches()
        dataframes = tuple(map(self.get_rot, dataframes))
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        for i in range(0, 2):
            condition = dataframes[i]["blend O-C/error"] >= 4
            df_high_error = dataframes[i][condition]
            df_normal = dataframes[i][~condition]
            ax[0].scatter(
                df_normal["Ka'"],
                df_normal["J'"],
                s=4 * df_normal["blend O-C/error"],
                facecolors=df_normal["Transition Type"].map(
                    {
                        "a-type": "#00FFFFFF",
                        "b-type": "#00FFFFFF",
                        "c-type": "#00FFFFFF",
                    }
                ),
                edgecolors=df_normal["Transition Type"].map(
                    {"a-type": "blue", "b-type": "green", "c-type": "orange"}
                ),
                linewidth=0.8,
            )
            ax[0].scatter(
                df_high_error["Ka'"],
                df_high_error["J'"],
                s=4 * df_high_error["blend O-C/error"],
                facecolors="#FF000065",
                edgecolors="red",
                linewidth=0.8,
            )
        condition_q = dataframes[2]["blend O-C/error"] >= 4
        df_high_error_q = dataframes[2][condition_q]
        df_normal_q = dataframes[2][~condition_q]
        ax[1].scatter(
            df_high_error_q["Ka'"],
            df_high_error_q["J'"],
            s=4 * df_high_error_q["blend O-C/error"],
            facecolors="#FF000065",
            edgecolors="red",
            linewidth=0.8,
        )
        ax[1].scatter(
            df_normal_q["Ka'"],
            df_normal_q["J'"],
            s=4 * df_normal_q["blend O-C/error"],
            facecolors=df_normal_q["Transition Type"].map(
                {"a-type": "#00FFFFFF", "b-type": "#00FFFFFF", "c-type": "#00FFFFFF"}
            ),
            edgecolors=df_normal_q["Transition Type"].map(
                {"a-type": "black", "b-type": "black", "c-type": "black"}
            ),
            linewidth=0.8,
        )

        ax[0].axline((0, 0), slope=1, color="lightgrey", linewidth=1.5)
        ax[0].set_xlabel("$K_a''$", fontsize=14)
        ax[0].set_ylabel("$J''$", fontsize=14)
        ax[0].set_ylim(0, max_J)
        ax[0].set_xlim(-0.5, max_Ka)
        ax[0].set_xticks((0, 10, 20, 30, 40, 50, 60))
        ax[0].set_yticks((0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120))
        ax[0].text(text_x_pos, text_y_pos, "R-Branch", fontsize=18)

        ax[1].axline((0, 0), slope=1, color="lightgrey", linewidth=1.5)
        ax[1].set_xlabel("$K_a''$", fontsize=14)
        ax[1].set_ylabel("$J''$", fontsize=14)
        ax[1].set_ylim(0, max_J)
        ax[1].set_xlim(-0.5, max_Ka)
        ax[1].set_xticks((0, 10, 20, 30, 40, 50, 60))
        ax[1].set_yticks((0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120))
        ax[1].text(text_x_pos, text_y_pos, "Q-Branch", fontsize=18)

        file_path = self.file_name + ".jpg"
        fig.savefig(file_path, dpi=800)

    def plot_data_dist_IR_pub(self, max_Ka: int, max_J: int):
        mpl.rc("text", usetex=True)
        plt.style.use(["science", "ieee"])
        text_x_pos = max_Ka - 22
        text_y_pos = max_J - 10
        self.categorize_error()
        self.res_dataframe = self.res_dataframe.sort_values(
            by=["Transition Type"], ascending=True
        )
        dataframes = self.split_branches()
        dataframes = tuple(map(self.get_IR, dataframes))
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        for i in range(0, 2):
            condition = dataframes[i]["blend O-C/error"] >= 4
            df_high_error = dataframes[i][condition]
            df_normal = dataframes[i][~condition]
            ax[0].scatter(
                df_normal["Ka'"],
                df_normal["J'"],
                s=4 * df_normal["blend O-C/error"],
                facecolors=df_normal["Transition Type"].map(
                    {
                        "a-type": "#00FFFFFF",
                        "b-type": "#00FFFFFF",
                        "c-type": "#00FFFFFF",
                    }
                ),
                edgecolors=df_normal["Transition Type"].map(
                    {"a-type": "blue", "b-type": "green", "c-type": "orange"}
                ),
                linewidth=0.8,
            )
            ax[0].scatter(
                df_high_error["Ka'"],
                df_high_error["J'"],
                s=4 * df_high_error["blend O-C/error"],
                facecolors="#FF000065",
                edgecolors="red",
                linewidth=0.8,
            )
        condition_q = dataframes[2]["blend O-C/error"] >= 4
        df_high_error_q = dataframes[2][condition_q]
        df_normal_q = dataframes[2][~condition_q]
        ax[1].scatter(
            df_high_error_q["Ka'"],
            df_high_error_q["J'"],
            s=4 * df_high_error_q["blend O-C/error"],
            facecolors="#FF000065",
            edgecolors="red",
            linewidth=0.8,
        )
        ax[1].scatter(
            df_normal_q["Ka'"],
            df_normal_q["J'"],
            s=4 * df_normal_q["blend O-C/error"],
            facecolors=df_normal_q["Transition Type"].map(
                {"a-type": "#00FFFFFF", "b-type": "#00FFFFFF", "c-type": "#00FFFFFF"}
            ),
            edgecolors=df_normal_q["Transition Type"].map(
                {"a-type": "black", "b-type": "black", "c-type": "black"}
            ),
            linewidth=0.8,
        )

        ax[0].axline((0, 0), slope=1, color="lightgrey", linewidth=1.5)
        ax[0].set_xlabel("$K_a''$", fontsize=14)
        ax[0].set_ylabel("$J''$", fontsize=14)
        ax[0].set_ylim(0, max_J)
        ax[0].set_xlim(-0.5, max_Ka)
        ax[0].set_xticks((0, 10, 20, 30, 40, 50, 60))
        ax[0].set_yticks((0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120))
        ax[0].text(text_x_pos, text_y_pos, "R-Branch", fontsize=18)

        ax[1].axline((0, 0), slope=1, color="lightgrey", linewidth=1.5)
        ax[1].set_xlabel("$K_a''$", fontsize=14)
        ax[1].set_ylabel("$J''$", fontsize=14)
        ax[1].set_ylim(0, max_J)
        ax[1].set_xlim(-0.5, max_Ka)
        ax[1].set_xticks((0, 10, 20, 30, 40, 50, 60))
        ax[1].set_yticks((0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120))
        ax[1].text(text_x_pos, text_y_pos, "Q-Branch", fontsize=18)

        file_path = self.file_name + ".jpg"
        fig.savefig(file_path, dpi=800)


class HandlerEllipse(HandlerPatch):
    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(
            xy=center, width=height + xdescent, height=height + ydescent
        )
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]
