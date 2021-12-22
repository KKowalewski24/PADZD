import os
from datetime import datetime
from typing import Any, List, Union

import pandas as pd


class LatexItem:
    ampersand: str = " & "
    centering: str = "\centering\n"
    float_barrier: str = "\FloatBarrier\n"


class Table(LatexItem):
    begin: str = "\\begin{table}[!htbp]\n"
    back_slashes: str = "\\\\"
    hline: str = "\hline\n"
    end_tabular: str = "\end{tabular}\n"
    end: str = "\end{table}\n"


    def get_begin_tabular(self, table_width: int) -> str:
        columns: str = ""
        for i in range(table_width):
            columns += "|c"
        columns += "|"
        return "\\begin{tabular}{" + columns + "}\n"


    def get_caption(self, text: str) -> str:
        replaced_text = replace_char_for_caption(text)
        return "\caption\n[" + replaced_text + "]{" + replaced_text + "}\n"


    def get_label(self, label: str) -> str:
        return "\label{" + label + "}\n"


class Image(LatexItem):
    begin: str = "\\begin{figure}[!htbp]\n"
    include: str = "\includegraphics\n"
    width: str = "[width=\\textwidth,keepaspectratio]\n"
    end: str = "\end{figure}"


    def __init__(self, directory_name: str) -> None:
        self.directory_name = directory_name


    def get_path(self, filename: str) -> str:
        return "img/" + filename


    def get_latex_path(self, filename: str) -> str:
        return "{" + self.get_path(filename) + ".png}\n"


    def get_caption(self, text: str) -> str:
        replaced_text = replace_char_for_caption(text)
        return "\caption\n[" + replaced_text + "]{" + replaced_text + "}\n"


    def get_label(self, label: str) -> str:
        return "\label{" + label + "}\n"


class LatexGenerator:

    def __init__(self, dir_name: str = "") -> None:
        self.dir_name = dir_name
        self.table = Table()
        self.image = Image(dir_name)


    def generate_vertical_table_df(self, df: pd.DataFrame, filename: str) -> None:
        result: str = self.table.begin + self.table.centering \
                      + self.table.get_begin_tabular(len(df.columns)) + self.table.hline

        header: str = ""
        for i in range(len(df.columns)):
            header += str(df.columns[i])
            if i < len(df.columns) - 1:
                header += self.table.ampersand

        header += " " + self.table.back_slashes + " " + self.table.hline

        body: str = ""
        for i in range(len(df.values)):
            for j in range(len(df.values[i])):
                body += str(df.values[i][j])
                if j < len(df.values[j]) - 1:
                    body += self.table.ampersand
            body += " " + self.table.back_slashes + " " + self.table.hline

        result += header + body + self.table.end_tabular + self.table.get_caption(filename) \
                  + self.table.get_label(filename) + self.table.end + self.table.float_barrier
        self._save_to_file(result, filename)


    def generate_vertical_table(self, header_names: List[str],
                                body_values: List[List[str]],
                                filename: str) -> None:
        if not self._compare_array_with_matrix_rows(header_names, body_values):
            raise Exception("Lists must have equal length")

        result: str = self.table.begin + self.table.centering \
                      + self.table.get_begin_tabular(len(header_names)) + self.table.hline

        header: str = ""
        for i in range(len(header_names)):
            header += self._fix_underscore_issue(str(header_names[i]))
            if i < len(header_names) - 1:
                header += self.table.ampersand

        header += " " + self.table.back_slashes + " " + self.table.hline

        body: str = ""
        for i in range(len(body_values)):
            for j in range(len(body_values[i])):
                body += self._fix_underscore_issue(str(body_values[i][j]))
                if j < len(body_values[i]) - 1:
                    body += self.table.ampersand
            body += " " + self.table.back_slashes + " " + self.table.hline

        result += header + body + self.table.end_tabular + self.table.get_caption(filename) \
                  + self.table.get_label(filename) + self.table.end + self.table.float_barrier
        self._save_to_file(result, filename)


    def generate_horizontal_table_df(self, df: pd.DataFrame, filename: str) -> None:
        result: str = self.table.begin + self.table.centering \
                      + self.table.get_begin_tabular(len(df.columns) + 1) + self.table.hline

        header: str = ""
        for i in range(len(df.columns)):
            header += str(df.columns[i])
            if i <= len(df.columns) - 2:
                header += self.table.ampersand

        result += self.table.ampersand + header + " " \
                  + self.table.back_slashes + " " + self.table.hline

        body: str = ""
        for i in range(len(df.values)):
            body += str(df.index[i]) + self.table.ampersand
            for j in range(len(df.values[i])):
                body += str(df.values[i][j])
                if j < len(df.values[i]) - 1:
                    body += self.table.ampersand

            body += " " + self.table.back_slashes + " " + self.table.hline

        result += body + self.table.end_tabular + self.table.get_caption(filename) \
                  + self.table.get_label(filename) + self.table.end + self.table.float_barrier
        self._save_to_file(result, filename)


    def generate_horizontal_table(self, header_names: Union[List[str], List[int]],
                                  horizontal_column_names: Union[List[str], List[int]],
                                  body_values: Union[List[List[str]], List[List[float]]],
                                  filename: str) -> None:
        if len(horizontal_column_names) != len(body_values):
            raise Exception(
                "horizontal_column_names and body_values must have equal length"
            )

        result: str = self.table.begin + self.table.centering \
                      + self.table.get_begin_tabular(len(body_values[0]) + 1) + self.table.hline

        if self._compare_array_with_matrix_rows(header_names, body_values):
            header: str = ""
            for i in range(len(header_names)):
                header += str(header_names[i])
                if i < len(header_names) - 1:
                    header += self.table.ampersand

            result += self.table.ampersand + header + " " \
                      + self.table.back_slashes + " " + self.table.hline

        body: str = ""
        for i in range(len(body_values)):
            body += str(horizontal_column_names[i]) + self.table.ampersand
            for j in range(len(body_values[i])):
                body += str(body_values[i][j])
                if j < len(body_values[i]) - 1:
                    body += self.table.ampersand
            body += " " + self.table.back_slashes + " " + self.table.hline

        result += body + self.table.end_tabular + self.table.get_caption(filename) \
                  + self.table.get_label(filename) + self.table.end + self.table.float_barrier
        self._save_to_file(result, filename)


    def generate_chart_image(self, filename: str) -> None:
        result: str = self.image.begin + self.image.centering \
                      + self.image.include + self.image.width
        result += self.image.get_latex_path(filename)
        result += self.image.get_caption(self._remove_png_extension(filename))
        result += self.image.get_label(self._remove_png_extension(filename))
        result += self.image.end
        self._save_to_file(result, filename)


    def _compare_array_with_matrix_rows(self, array: List[Any], matrix: List[List[Any]]) -> bool:
        for item in matrix:
            if len(array) != len(item):
                return False

        return True


    def _save_to_file(self, data: str, filename: str) -> None:
        path: str = ""
        if self.dir_name != "":
            path = self.dir_name + "/"
            if not os.path.exists(self.dir_name):
                os.makedirs(self.dir_name)

        path += filename + "-" + datetime.now().strftime("%H%M%S") + ".txt"
        with open(path, "w", encoding="UTF-8") as file:
            file.write(data)


    def _remove_png_extension(self, string: str) -> str:
        return string.replace(".png", "")


    def _fix_underscore_issue(self, string: str) -> str:
        return string.replace("_", "\_")


def replace_char_for_caption(string: str) -> str:
    chars: List[str] = ["-", "_"]
    for char in chars:
        string = string.replace(char, " ")

    return string