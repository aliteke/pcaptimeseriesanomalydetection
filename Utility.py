import json
import os
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits import mplot3d                # plotting joint probability matrix (jpm) in 3D
import numpy as np
import datetime as dt
from scipy.linalg import svd
from sklearn.utils.extmath import softmax
from scipy.stats import entropy
from scipy.linalg import hankel                 # To calculate Hankel Toeplitz Matrix for bins at each time window
from sklearn.metrics import confusion_matrix
import csv
#######################
# Function Definitions
#######################

# Taken from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
# def plot_confusion_matrix(y_true, y_pred, classes,
#                          normalize=False,
#                          title=None,
#                          cmap=plt.cm.Blues):
# (tn, fp, fn, tp)
# TP=tp, FN=fn, FP=fp, TN=tn
def plot_confusion_matrix(TP, FN, FP, TN,
                          classes=None,         # This is an np.array(['Anomaly', 'Bening']) set below
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    # cm = confusion_matrix(y_true, y_pred)
    cm = np.ndarray(shape=(2, 2), buffer=np.array([TP, FN, FP, TN]), dtype=float)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    classes = np.array(['Anomaly', 'Benign'], dtype='<U10')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'     # if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# from numpy import linalg as LA            # LA.eig() will calculate EigenValues and EigenVectors of Henkel Matrix
def hankel_matrix(n):
    n_appended = n
    # if odd number of elements in "n", we append average of "n" to array "n"
    if len(n) % 2 == 1:
        n_appended.append(sum(n) / float(len(n)))
        dim = (len(n_appended) / 2)
    else:
        dim = (len(n_appended) / 2)

    hm = hankel(n_appended)

    h = hm[0:int(dim), 0:int(dim)]  # real hankel matrix
    # eigenvalues, eigenvectors = LA.eig(h)
    u, svd_h, Vh = svd(h)
    # return eigenvalues, eigenvectors, svd_h, h
    return svd_h, h

# Old function, not used
def vmstat_d_disk_reads_sda_total():
    total_number_of_bins = 20
    time_window_in_seconds = 1000
    time_window_shift = 10

    jsondata = "../full_data.json"

    if not os.path.isfile(jsondata):
        print("[!] Couldn't find json data file %s" % jsondata)
        sys.exit()

    with open(jsondata, 'r') as f:
        data_dict = json.load(f)

    print("[+] Total number of items in tree_root: %d" % (len(data_dict["tree_root"])))

    lst_sda_read_total = []
    lst_sda_time = []
    for i in data_dict["tree_root"]:
        for j in i["vmstat_d"]["list_stats"]:
            cur_t = i["vmstat_d"]["date_time"]
            index = cur_t.rfind(":")
            cur_t = str(cur_t[:index] + "." + cur_t[index + 1:]).replace("T", " ")
            cur_t = dt.datetime.strptime(str(cur_t[:-3]), '%Y-%m-%d %H:%M:%S.%f')

            lst_sda_time.append(cur_t)

            bytes = int(j["disk_reads"][0]["sda"][0]["total"])
            lst_sda_read_total.append(bytes)

    total_experiment_in_seconds = (lst_sda_time[len(lst_sda_time) - 1] - lst_sda_time[0]).total_seconds()
    max_read_amount = max(lst_sda_read_total)
    min_read_amount = min(lst_sda_read_total)
    delta_read_bytes = max_read_amount - min_read_amount
    bin_width = delta_read_bytes / total_number_of_bins
    bin_edges = range(min_read_amount, max_read_amount, bin_width)

    # list of 2 values, will keep (start_index, ending_index) for each window
    lst_window_start_end_indices = []

    i = 0
    while i < len(lst_sda_time):
        starting_index = i  # starting index for time window
        curtime = lst_sda_time[i]
        endtime = curtime + dt.timedelta(seconds=time_window_in_seconds)

        while (curtime <= endtime) and (i < len(lst_sda_time)):
            i += 1
            if i >= len(lst_sda_time):
                break
            curtime = lst_sda_time[i]

        ending_index = i - 1  # final index in the current time window
        lst_window_start_end_indices.append((starting_index, ending_index))

        plt.clf()  # clear the figure
        plt.xlabel("Total Disk Read")
        plt.ylabel("# of Elements in a Bin)")
        plt.title("vmstat_d, (Total Disk Reads from sda)," + "\n" +
                  "#bins: %d, sliding_time_window: %d sec, time_delta: %d" %
                  (total_number_of_bins, time_window_in_seconds,
                   (lst_sda_time[ending_index] - lst_sda_time[starting_index]).total_seconds()) +
                  "\n" + "curtime: {}".format(str(lst_sda_time[starting_index])))
        plt.grid(True)
        # n, bins, patches = plt.hist(lst_sda_read_total[starting_index:ending_index], bins=total_number_of_bins, normed=True)
        n, bins, patches = plt.hist(lst_sda_read_total[starting_index:ending_index],
                                    bins=bin_edges,
                                    range=[min_read_amount, max_read_amount],
                                    normed=True)
        cur_mean = np.mean(bins)
        cur_stddev = np.std(bins)
        y = mlab.normpdf(bins, cur_mean, cur_stddev)
        plt.plot(bins, y, '--')

        # print"[+] #bins: %d, time_window: %d sec, from-to: %s-%s, delta: %d, init_index: %d, end_index: %d" % (total_number_of_bins, time_window_in_seconds, str(lst_sda_time[starting_index]), str(lst_sda_time[ending_index]), (lst_sda_time[ending_index]-lst_sda_time[starting_index]).total_seconds(), starting_index, ending_index)
        plt.show()
        plt.savefig("fixed_bins/bins_sda_total_disk_read_vmstatd{}.png".format(i), dpi=500)


# arguments are 6 list of floating point numbers, and last one is list of times in datetime format
def plot_list(lst_avg_cpu_user, lst_avg_cpu_nice, lst_avg_cpu_system, lst_avg_cpu_iowait, lst_avg_cpu_steal,
              lst_avg_cpu_idle, lst_time):
    plt.clf()

    plt.xlabel("Date-Time")
    plt.ylabel("CPU Usage")
    plt.title("CPU Usage averages for User, System, IO-Wait, Steal, Nice, Idle (iostat)")
    plt.grid(True)
    plt.plot(lst_time, lst_avg_cpu_user, 'bo', label='avg_cpu_user')
    plt.plot(lst_time, lst_avg_cpu_nice, 'g+', label="avg_cpu_nice")
    plt.plot(lst_time, lst_avg_cpu_system, 'r*', label="avg_cpu_system")
    plt.plot(lst_time, lst_avg_cpu_iowait, 'c:', label='avg_cpu_iowait')
    plt.plot(lst_time, lst_avg_cpu_steal, 'm--', label="avg_cpu_steal")
    # plt.plot(lst_avg_cpu_idle, lst_time,'k-.', label="avg_cpu_idle")
    plt.legend(loc='upper left')
    plt.show()

def generate_syn_cpu_data(data_length, perc):
    np.random.seed()
    mu, sigma = 0.75, 0.2  # mean and standard deviation
    s = np.random.normal(mu, sigma, data_length)
    su = np.random.uniform(2, 7, int(len(s) * perc))
    anomaly_index = np.random.randint(1, int(len(s)), int(len(su)))

    for i in range(0, len(su) - 1):
        s[anomaly_index[i]] = su[i]

    anomaly_index.sort()
    s[anomaly_index[-1]] = np.random.normal(mu, sigma, 1)

    tmp_lst = (anomaly_index.tolist())
    del (tmp_lst[-1])
    anomaly_index = np.array(tmp_lst)

    return s, anomaly_index


def generate_synthetic_data(data_length):
    np.random.seed()
    # mu, sigma = 0.75, 0.2  # mean and standard deviation
    mu = np.random.uniform(0.2, 2, 2)
    sigma = np.random.uniform(0.1, 1, 2)
    perc = np.random.uniform(0.05, 0.20, 2)     # Percentage of Anomalous Data

    # Normal Data
    s1 = abs(np.random.normal(mu[0], sigma[0], data_length))
    s2 = abs(np.random.normal(mu[1], sigma[1], data_length))

    # Anomalous Data
    su1 = np.random.uniform(2, 7, int(len(s1) * perc[0]))
    su2 = np.random.uniform(2, 7, int(len(s2) * perc[1]))
    su3 = np.random.uniform(1, 8, int(len(s1) * 0.05))       # shared anomalies

    anomaly_index1 = np.random.randint(1, int(len(s1)), int(len(su1)))
    anomaly_index2 = np.random.randint(1, int(len(s2)), int(len(su2)))
    anomaly_index3 = np.random.randint(1, int(len(s1)), int(len(su3)))

    for i in range(0, len(su1) - 1):
        s1[anomaly_index1[i]] = su1[i]

    for i in range(0, len(su2) - 1):
        s2[anomaly_index2[i]] = su2[i]

    for i in range(0, len(su3) - 1):
        s1[anomaly_index3[i]] = su3[i]
        s2[anomaly_index3[i]] = su3[i]

    anomaly_index1.sort()
    anomaly_index2.sort()
    anomaly_index3.sort()

    # s[anomaly_index[-1]] = np.random.normal(mu, sigma, 1)
    #
    # tmp_lst = (anomaly_index.tolist())
    # del (tmp_lst[-1])
    # anomaly_index = np.array(tmp_lst)

    return s1, s2, anomaly_index1, anomaly_index2, anomaly_index3


def extract_time_series_from_csv(csvfilename,
                                 columnindextime=1,
                                 columnindexlabel=8,
                                 columnIndexesToExtractTimeSeries=[2, 3, 4]):
    lst_time = []
    lst_labels = []
    lst_time_series_a = []
    lst_time_series_b = []

    with open(csvfilename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        data_list = list(csv_reader)
        line_count = 0
        for row in data_list:
            if line_count == 0:
                line_count += 1
                continue

            if line_count == 1:  # record the initial timestamp for the first record
                first_timestamp = dt.datetime.strptime(row[1], "%m/%d/%Y, %H:%M:%S.%f")

            lst_time.append(dt.datetime.strptime(row[columnindextime], "%m/%d/%Y, %H:%M:%S.%f"))  # timestamp
            lst_time_series_a.append(float(row[columnIndexesToExtractTimeSeries[0]]))   # Time-Series-1 (AvgEthBytes)
            lst_time_series_b.append(float(row[columnIndexesToExtractTimeSeries[2]]))   # Time-Series-2 (PktCnt)
            lst_labels.append(int(row[columnindexlabel]))                               # Label of data-point: {0, 1}

            line_count += 1
        print(f'Processed {line_count} lines.')

    return lst_time, lst_labels, lst_time_series_a, lst_time_series_b


"""
x: data points in the current window
n: return the values of the histogram bins, number of items in each bin.
bin_edges: start and end points for each bin in the window.
"""
def calc_bin_disribution(x, bin_edges):
    bin_distro = np.zeros((len(bin_edges)-1,), dtype=int)

    # i is the bin index (from 0 through 19, in the case of 20 bins)
    for i in range(0, len(bin_edges)-1):
        cur_bin_start = bin_edges[i]
        cur_bin_end = bin_edges[i+1]
        for cur_data_point in x:
            if cur_data_point >= cur_bin_start and cur_data_point < cur_bin_end:
                bin_distro[i] += 1
            elif i == len(bin_edges)-2:       # if we are at the last bin (i.e. bin_index == 19), include bin_end_edge
                if cur_data_point >= cur_bin_start and cur_data_point <= cur_bin_end:
                    bin_distro[i] += 1

    return bin_distro
