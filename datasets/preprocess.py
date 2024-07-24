import pandas as pd
import numpy as np
from ast import literal_eval
from tqdm import tqdm
import csv

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

path = "/home/mauro/PycharmProjects/Neural-Fly/datasets/CyberZoo/V4.0/logs/test/1/bebop_minsnap_baseline_1_1.1_1.csv"


# strings = open(path).readlines()[0]
# string = strings[42:]
#
# def find_nth_overlapping(haystack, needle, n):
#     start = haystack.find(needle)
#     while start >= 0 and n > 1:
#         start = haystack.find(needle, start+1)
#         n -= 1
#     return start
#
# i = 0
# newdata = ["t,p,p_d,v,v_d,v_ekf,Angles,q,w,R,T_sp,rpm\n"]
# while i < len(string):
#     idx = find_nth_overlapping(string[i:], ']",', 10)
#     newdata.append(string[i:i+idx]+']"\n')
#     i=i+idx+3

# for idx, string in enumerate(strings):
#     strings[idx] = string[:-12] + "\n"

# f = open(path, 'w')
# for row in newdata:
#     f.write(row)
# f.close()

df = pd.read_csv(path)

thres = 0
start = np.argmax(np.linalg.norm(np.stack(df["v_d"].apply(literal_eval)), axis=1) > thres)
end = len(df["v"].apply(literal_eval).values) - 1 - np.argmax(np.flip(np.linalg.norm(np.stack(df["v"].apply(literal_eval)), axis=1)) > thres)

df = df[start:end].reset_index(drop=True)

t = np.array(df["t"])
t-= t[0]

v = df["v"].apply(literal_eval)
v = np.array(v.to_list())

# cmd_ff = df["cmd_FF"].apply(literal_eval)
# cmd_ff = np.array(cmd_ff.to_list())

# plt.plot(t, v[:,0])
# plt.show()

prev = np.array([0, 0])
for idx, value in enumerate(tqdm(v, total=len(v))):
    if (value[0:2] == prev).all():
        df.drop(index=idx, inplace=True)
    prev = value[0:2]

df.to_csv(path, index=False)