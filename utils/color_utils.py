# @author: xiwu
# @license:
# @contact: fzy19931001@gmail.com
# @software: PyCharm
# @file: color_utils.py
# @time: 2019/5/23 16:57
# @desc:

import termcolor

# convert to colored strings


def to_red(content):
    return termcolor.colored(content, "red", attrs=["bold"])


def to_green(content):
    return termcolor.colored(content, "green", attrs=["bold"])


def to_blue(content):
    return termcolor.colored(content, "blue", attrs=["bold"])


def to_cyan(content):
    return termcolor.colored(content, "cyan", attrs=["bold"])


def to_yellow(content):
    return termcolor.colored(content, "yellow", attrs=["bold"])


def to_magenta(content):
    return termcolor.colored(content, "magenta", attrs=["bold"])
