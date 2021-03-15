import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits import mplot3d            # plotting joint probability matrix (jpm) in 3D
import numpy as np
import datetime as dt
from sklearn.utils.extmath import softmax
from scipy.stats import entropy
from scipy.linalg import hankel
from scipy.linalg import svd
import Utility as util                      # All our functions
import re
import pandas as pd                         # we will use DataFrame class

# -- Find the index of each window with at least one anomalous data-point --
# Inputs:   1.(Shifting) Window start-end indexes,
#           2. labels for each Time-Series-Data-Point coming from PCAP.
# Output:   lst_anomaly_window_indices (index of each window that has at least 1 anomalous data-point)
def getIndicesOfAbnormalWindows(lst_labels, lst_window_start_end_indices):
    lst_anomaly_window_indices = []
    anomaly_counter_per_window = 0
    for i in range(0, len(lst_window_start_end_indices)):
        # lets say lst_window_start_end_indices[i] = (35, 50)
        cur_indices = lst_window_start_end_indices[i]
        for j in range(cur_indices[0], cur_indices[1]+1):    # i[1]+1 is to include ending index i.e. 50 in this example
            # j is index of Data-Point in the current window: i.e. [35:50]
            if lst_labels[j] == 1:   # 0->no_anomaly, 1->anomaly
                anomaly_counter_per_window += 1
        print(f"[+] {anomaly_counter_per_window} anomalies in Window#{i}")

        # maybe if more than half of the items in the current window are ab-normal, window should be labeled "Anomaly"
        # define percent of abnormal window elements as threshold for overall window's anomaly label.
        min_percentage_of_abnormal_data = 0.25
        if anomaly_counter_per_window > float((cur_indices[1] - cur_indices[0] + 1) * min_percentage_of_abnormal_data):
            lst_anomaly_window_indices.append(i)

        anomaly_counter_per_window = 0      # clear the counter, for next window.
    return lst_anomaly_window_indices


# Find index of every data-point, that has label: 1 (index of each abnormal data-point).
def getIndicesOfAbnormalDataPoints(lst_labels):
    lst_original_anomaly_indices = []
    for l_i in range(0, len(lst_labels)):
        if lst_labels[l_i] == 1:
            lst_original_anomaly_indices.append(l_i)

    return lst_original_anomaly_indices

# Return a list of indexes that we thought is anomaly.
def get_indices_of_calculated_anomalies(lst_delta, lst_window_start_end_indices,
                                        greenwich, time_window_shift, time_window_in_seconds,
                                        epsilon):
    lst_indices=[]
    # find windows that has anomalies
    for x in range(len(lst_delta)):  # For each window, check if it's delta is more than epsilon
        if lst_delta[x] >= epsilon:
            y = str(greenwich + dt.timedelta(seconds=(x * time_window_shift)))
            z = str(greenwich + dt.timedelta(seconds=((x * time_window_shift) + int(time_window_in_seconds))))

            # If so, that means our method claims there is anomaly in current window
            # Find the time indices for the current window (which is same as anomalous CPU value indices)
            window_limits = lst_window_start_end_indices[x]
            for i in range(window_limits[0], window_limits[1]+1):
                if i not in lst_indices:
                    lst_indices.append(i)
    lst_indices.sort()
    return lst_indices


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def difference(lst1, lst2):
    s1 = set(lst1)
    s2 = set(lst2)
    return s1.difference(s2)

def calculate_tpn_fprn_tnrn_fnn(lst_original_anomaly_indices,       # Original Positives
                                lst_calculated_anomaly_indices,     # True&False Positives
                                first_data_index,                   # First Windows, start index
                                last_data_index):                   # Last Windows, end index
    lst_original_normal_indices = []                                # Original Negatives
    for j in range(first_data_index, last_data_index+1):
        if j not in lst_original_anomaly_indices:
            lst_original_normal_indices.append(j)

    assert len(lst_original_normal_indices) + len(lst_original_anomaly_indices) == \
           (last_data_index - first_data_index + 1)

    true_positives = intersection(lst_original_anomaly_indices, lst_calculated_anomaly_indices)
    false_positives = intersection(lst_original_normal_indices, lst_calculated_anomaly_indices)

    true_negatives = difference(lst_original_normal_indices, lst_calculated_anomaly_indices)
    false_negatives = difference(lst_original_anomaly_indices, lst_calculated_anomaly_indices)

    assert len(true_positives) + len(false_positives) + \
           len(true_negatives)+ len(false_negatives) == \
           (last_data_index - first_data_index + 1)

    return (len(true_positives), len(false_positives), len(true_negatives), len(false_negatives))

def timeseriesDetection(dataFile, inputJSON,
                        p_m1=0.70, p_m2=0.25, p_m3=0.05,
                        p_num_bins=20, p_time_window=100, p_time_shift=20,
                        p_epsilon=0.025, p_gamma=0.3, p_alpha=0.59,
                        numDimensions=1):
    total_number_of_bins = p_num_bins
    time_window_in_seconds = p_time_window
    time_window_shift = p_time_shift
    # we will take average of 3 readings at every iostat command data record.
    lst_time_series_a = []
    lst_time_series_b = []

    # Not needed anymore, keeping for legacy purposes
    lst_avg_cpu_nice = []
    lst_avg_cpu_system = []
    lst_avg_cpu_iowait = []
    lst_avg_cpu_steal = []

    # Synthetic Random Anomaly Data
    lst_syn_cpu_data = []

    # this will collect the time iostat was run at each time
    lst_time = []
    lst_labels = []                 # anomaly labels (0, 1) for each data-point, by respective time-series index.
    lst_window_labels = []          # anomaly labels for each window, calculated from original individual data-labels.

    # this list of Arrays will keep the softmax'ed x
    lst_softmaxed = []

    # list of EigenValues/Vectors for each window, calculated from HenkelMatrix for that time window's bins array
    lst_eigenvalues = []
    lst_eigenvectors = []

    if not os.path.isfile(dataFile):
        print("[!] Couldn't find data file %s" % dataFile)
        sys.exit()

    if inputJSON:
        with open(dataFile, 'r') as f:
            data_dict = json.load(f)
        print("[+] Total number of items in tree_root: %d" % (len(data_dict["tree_root"])))

        for i in data_dict["tree_root"]:
            cur_t = i["iostat"]["date_time"]
            index = cur_t.rfind(":")
            cur_t = str(cur_t[:index] + "." + cur_t[index + 1:]).replace("T", " ")
            cur_t = dt.datetime.strptime(str(cur_t[:-3]), '%Y-%m-%d %H:%M:%S.%f')
            lst_time.append(cur_t)

            # "iowait": "0.08", "system": "0.26", "idle": "97.74", "user": "1.93", "cpu_nice": "0.00", "steal": "0.00"
            avg_cpu_iowait_sum = 0
            avg_cpu_system_sum = 0
            avg_cpu_idle_sum = 0
            avg_cpu_user_sum = 0
            avg_cpu_cpu_nice_sum = 0
            avg_cpu_steal_sum = 0

            for j in i["iostat"]["list_stats"]["list_stats"]:
                avg_cpu_iowait_sum += float(j["avg-cpu"]["iowait"])
                avg_cpu_system_sum += float(j["avg-cpu"]["system"])
                avg_cpu_idle_sum += float(j["avg-cpu"]["idle"])
                avg_cpu_user_sum += float(j["avg-cpu"]["user"])
                avg_cpu_cpu_nice_sum += float(j["avg-cpu"]["cpu_nice"])
                avg_cpu_steal_sum += float(j["avg-cpu"]["steal"])

            lst_time_series_a.append(avg_cpu_user_sum / 3.0)
            lst_avg_cpu_nice.append(avg_cpu_cpu_nice_sum / 3.0)
            lst_avg_cpu_system.append(avg_cpu_system_sum / 3.0)
            lst_avg_cpu_iowait.append(avg_cpu_iowait_sum / 3.0)
            lst_avg_cpu_steal.append(avg_cpu_steal_sum / 3.0)
            lst_time_series_b.append(avg_cpu_idle_sum / 3.0)
    # input files is CSV
    else:

        #lst_time, lst_labels, lst_time_series_a, lst_time_series_b = util.extract_time_series_from_csv(dataFile)
        # TODO: Testing Time-Series-A, Time-Series-B switch (Change This back, 03/11/2021)
        lst_time, lst_labels, lst_time_series_b, lst_time_series_a = util.extract_time_series_from_csv(dataFile)

    # Generate Random, Anomaly Data
    # (lst_random_cpu_data, lst_indices_artificial_anomalies) = util.generate_syn_cpu_data(len(lst_time_series_a), 0.10)
    # lst_time_series_a = lst_random_cpu_data.tolist()
    # Comment if you don't want to use the randomly generated Time-Series data
    # s1, s2, anomaly_index1, anomaly_index2, anomaly_index3 = util.generate_synthetic_data(len(lst_time_series_a))
    # lst_time_series_a = s1
    # lst_time_series_b = s2

    print("[+] Size of first time-series data (list): %d " % (len(lst_time_series_a)))

    # TODO: GENERATE 2d Matrix from 2 time-series
    if numDimensions == 2:
        joint_matrix = util.gen_matrix_from_2_time_series(lst_time_series_a, lst_time_series_b, total_number_of_bins)

    # calculation for one parameter.
    # TODO We should make this part a function, so that we could call for different CPU parameters
    total_experiment_in_seconds = (lst_time[-1] - lst_time[0]).total_seconds()
    print("[+] Total Duration for experiment: %f seconds" % total_experiment_in_seconds)

    # MIN-MAX Values for the first Time-Series
    max_time_series_a = max(lst_time_series_a)
    min_time_series_a = min(lst_time_series_a)

    # MIN-MAX Values for the second Time-Series
    max_time_series_b = max(lst_time_series_b)
    min_time_series_b = min(lst_time_series_b)

    # Distance between Max-Min in both time-series
    delta_time_series_a = max_time_series_a - min_time_series_a  # Distance between maximum value and min value
    delta_time_series_b = max_time_series_b - min_time_series_b    # Distance for second Time-Series

    # bin_width form the time-series
    bin_width = delta_time_series_a / total_number_of_bins  # size of each bin, depending on the number of bins
    bin_width2 = delta_time_series_b / total_number_of_bins  # size of each bin in the second time-series

    # BIN_EDGES for each time-series
    bin_edges = np.arange(min_time_series_a, max_time_series_a, bin_width).tolist()  # calculate each bin's boundaries
    bin_edges2 = np.arange(min_time_series_b, max_time_series_b, bin_width2).tolist()  # calculate each bin's boundaries

    bin_edges.append(max_time_series_a)
    bin_edges2.append(max_time_series_b)

    # TODO: We need to slide the time window so that it overlaps with the previous window
    greenwich = lst_time[0]  # First time point from the experiment's log file.
    i = 0
    number_of_time_shifts = 0  # at each iteration we will shift the current window "time_window_shift"
    starting_index = 0  # starting index for current time window

    # list of 2 value tuples, will keep (start_index, ending_index) for each window
    lst_window_start_end_indices = []

    while i < len(lst_time):
        total_shift = number_of_time_shifts * time_window_shift
        number_of_time_shifts += 1

        curtime = greenwich + dt.timedelta(seconds=total_shift)

        # find the current window's starting index,
        # so that lst_time[starting_index] is less than or equal to curtime
        # lst_time[ starting_index ] <= curtime

        while lst_time[starting_index] <= curtime:
            starting_index += 1

        starting_index -= 1
        i = starting_index  # reset "i" to start from the start_index for the current window
        curtime = lst_time[starting_index]  # reset curtime to starting time for the current window

        endtime = curtime + dt.timedelta(seconds=int(time_window_in_seconds))  # upper bound for time record in window

        while (curtime <= endtime) and (i < len(lst_time)):  # loop until we found the index for final time record
            i += 1
            if i >= len(lst_time):
                break
            curtime = lst_time[i]

        ending_index = i - 1  # index for biggest time value in the current time window

        # add (starting_index, ending_index) to list of window indexes
        lst_window_start_end_indices.append((starting_index, ending_index))

        # x = lst_time_series_a[starting_index:ending_index + 1]    # data for the current time window in 1st TS
        x = lst_time_series_a[starting_index:ending_index]          # Not sure why we added +1 before?
        if numDimensions > 1:
            # y = lst_time_series_b[starting_index:ending_index + 1]        # Not sure why we added +1 before?
            y = lst_time_series_b[starting_index:ending_index]              # data for the current time window in 2nd TS

        n = util.calc_bin_disribution(x, bin_edges)

        if numDimensions > 1:      # Only if we are using 2 Time-Series
            n2, bins2, patches2 = plt.hist(y,
                                           bins=bin_edges2,
                                           range=[min_time_series_b, max_time_series_b],  # MIN/MAX for 2nd time-series
                                           #normed=False,
                                           rwidth=0.85,
                                           histtype='step'
                                           )
            plt.close()

            n2_compare = util.calc_bin_disribution(y, bin_edges2)
            if n2.all() == n2_compare.all():
                print(f'[+] Our bin distribution is working. (len(n2)={len(n2)}), (len(n2_compare)={len(n2_compare)})')
            else:
                print('[+] Our bin distribution is NOT working')

        if numDimensions > 1:
            jpm, n_npm = util.gen_matrix_from_N_time_series(n, n2, n2, n2)
            # jpm, n_npm = util.gen_matrix_from_3_time_series(n, n, n2)
            # jpm, n_npm = util.gen_matrix_from_2_time_series(n, n2)
            jpm_raveled = jpm.ravel()
        else:       # for one-time-series only
            jpm_raveled = n

        jpm_raveled = jpm_raveled.astype(float)

        # SOFTMAX'ing the distribution of data-points into BINS at the current window.
        x1 = np.asarray(jpm_raveled)
        x2 = np.reshape(x1, (1, len(x1)))
        x3 = -x2                                # TODO: Ask Korkut abi, why multiply by -1 ?
        x4 = softmax(x3)
        x5 = np.reshape(x4, len(jpm_raveled))
        x6 = x5.tolist()

        lst_softmaxed.append(x6)  # Probability distribution of cpu usage

    # Now we went through whole array of values, calculated soft-maxes, it's time to calculate anomaly_scores
    print("[+] Size of lst_softmaxed: %d" % (len(lst_softmaxed)))

    # These are the weights for KL calculations
    m1 = p_m1       # 0.70
    m2 = p_m2       # 0.25
    m3 = p_m3       # 0.05

    epsilon = p_epsilon     # 0.025         # Threshold value for anomaly: f(w) - ψ > ε         (Equation-8)
    gamma = p_gamma         # 0.3           # In paper gamma is used in Eq.7 to calculate MU_w
    alpha = p_alpha         # 0.59          # In paper alpha is used in Eq.8 to calculate ψ = (µ{w−1} + α*σ{w−1})

    # List of Dr. Bruno's anomaly scores. List consisting of f(w) for each window, starting from window#3.
    lst_anomaly_scores_T = []

    # µw => Moving Average of f(w).
    lst_mvavg = []             # µw, (i.e. MU_w): moving average(µ) for current window (w). Equation-7 in SymKL Paper.

    # σw => Standard Deviation, that are recursively updated below
    lst_std = []               # σw, (i.e. SIGMA_w): Std Deviation of current window (w). Eq-7 in paper.

    # anomaly threshold -       # ψ =  µ_{w−1} + (α * σ{w−1} )      # Equation-8 in sym-kl paper.
    lst_anomaly_runningavg = []

    # difference between f(w) and moving averages  # ∆
    lst_delta = []       # Equation-8:  DELTA = f(w) - ( MU_{w-1} + ALPHA*SIGMA_{w−1} )

    # this will count till 3 before calculating new moving averages
    reset_wait_counter = 0

    # anomaly detected
    b_anomaly_detected = False

    # right after an anomaly, we need to start counting,
    # keep another boolean to detect the start of counting time
    b_start_timer = False

    ######################################################################
    # calculate KL distance starting from index 3 (indexing starts from 0)
    # Will compare current item, (i), with  (i-1), (i-2), (i-3)
    # m1 * KL( lst_softmaxed[i], lst_sofmaxed[i-1] ) +
    # m2 * KL( lst_softmaxed[i], lst_softmaxed[i-2] ) +
    # m3 * KL ( lst_softmaxed[i], lst_softmaxed[i-3])
    ######################################################################
    for i in range(0, len(lst_softmaxed)):
        cur_window_f_w = 0                  # f(w)
        cur_window_moving_avg = 0           # µ(w)
        cur_window_std_dev = 0              # σ(w)
        cur_window_psi = 0                  # ψ(w)
        tl1=0
        tl2=0

        if i == 0:
            j4 = lst_softmaxed[i]       # NOT KL, it's ENTROPY of 1-distribution only. (i.e. 2.88104)

        elif i == 1:
            j1 = [z * m1 for z in lst_softmaxed[i - 1]]
            j4 = [sum(index1) for index1 in zip(j1)]        # j1 == j4 (is TRUE)

        elif i == 2:
            j1 = [z * m1 for z in lst_softmaxed[i - 1]]
            j2 = [z * m2 for z in lst_softmaxed[i - 2]]
            j4 = [sum(index1) for index1 in zip(j1, j2)]

        elif i >= 3:
            # lst_softmaxed -> paper's equation-(6)
            j1 = [z * m1 for z in lst_softmaxed[i - 1]]
            j2 = [z * m2 for z in lst_softmaxed[i - 2]]
            j3 = [z * m3 for z in lst_softmaxed[i - 3]]
            j4 = [sum(index1) for index1 in zip(j1, j2, j3)]

        tl1 = entropy(lst_softmaxed[i], j4)     # j4 = m1*P1 + m2*P2 + m3*P3 (Equation-6)
        tl2 = entropy(j4, lst_softmaxed[i])     # KL is implemented in entropy(q,p) function.
        cur_window_f_w = tl1 + tl2                         # f(w), equation-6's result.

        # lst_anomaly_scores_T[i] -> f(w)
        if b_start_timer and not b_anomaly_detected and 3 >= reset_wait_counter > 0:
            cur_window_f_w = 0                                  # f(w)
            cur_window_moving_avg = 0                           # µ(w)
            cur_window_std_dev = 0                              # σ(w)
            lst_anomaly_scores_T.append(cur_window_f_w)
            lst_mvavg.append(cur_window_moving_avg)
            lst_std.append(cur_window_std_dev)

        else:
            lst_anomaly_scores_T.append(cur_window_f_w)         # f(w)

            # Calculate µ (mean) for current window.  µ(w_i): for i=0, µ(w_i)=0
            #cur_window_moving_avg = (gamma * lst_mvavg[i - 4]) + ((1 - gamma) * lst_anomaly_scores_T[i - 4])
            if i == 0:
                cur_window_moving_avg = 0
            else:
                cur_window_moving_avg = (gamma * lst_mvavg[i - 1]) + ((1 - gamma) * cur_window_f_w)
            lst_mvavg.append(cur_window_moving_avg)

            # Calculate σ (Standard Deviation) for current window.  σ(w_i): for i=0, σ(w_i)=0
            if i == 0:
                cur_window_std_dev = 0
            else:
                cur_window_std_dev = np.sqrt(
                    (gamma * (lst_std[i - 1] ** 2)) +
                    ((1 - gamma) * ((cur_window_f_w - cur_window_moving_avg)**2)))
            lst_std.append(cur_window_std_dev)

        # lst_anomaly_runningavg -> ψ -> paper's Equation-8: MU_{w-1} + alpha*sigma{w-1}
        if i == 0:
            cur_window_psi = 0
        else:
            cur_window_psi = lst_mvavg[i - 1] + (alpha * lst_std[i - 1])        # ψ(w), based on (w-1)
        lst_anomaly_runningavg.append(cur_window_psi)

        lst_delta.append(cur_window_f_w - cur_window_psi)             # ∆(w_i) = f(w_i) - ψ(w_i)

##################################################################################################
####### This section only changes BOOLEAN values that effects next iteration######################
        if lst_delta[-1] > epsilon and not b_anomaly_detected:
            b_anomaly_detected = True
            # reset_wait_counter += 1

        # We are in ANOMALY REGION, check for leaving ANOMALY
        elif lst_delta[-1] > epsilon and b_anomaly_detected:
            # do nothing
            continue
        # Going back below epsilon threshold,
        # change the boolean(detected) to false,
        # start the counter (reset_wait_counter)
        elif lst_delta[-1] <= epsilon and b_anomaly_detected:
            b_anomaly_detected = False
            b_start_timer = True

        if b_start_timer and reset_wait_counter < 3:
            reset_wait_counter += 1
        elif b_start_timer and reset_wait_counter == 3:
            b_start_timer = False
            reset_wait_counter = 0
##################################################################################################

    # Generates index of every data-point, that has label: 1 (index of each abnormal data-point)
    lst_original_anomaly_indices = getIndicesOfAbnormalDataPoints(lst_labels)

    # This will return a list of indices that our method thinks is anomalous
    lst_calculated_anomaly_indices = get_indices_of_calculated_anomalies(lst_delta, lst_window_start_end_indices,
                                                                         greenwich, time_window_shift,
                                                                         time_window_in_seconds, epsilon)

    # Given original anomaly indices, compare to our method's calculated anomaly indices and find tp/fp/tn/fn results
    results = calculate_tpn_fprn_tnrn_fnn(lst_original_anomaly_indices, lst_calculated_anomaly_indices,
                                lst_window_start_end_indices[0][0], lst_window_start_end_indices[-1][1])

    # From Dr. Issa's anomaly detection paper:
    # False Positive Rate (FPR) , Detection Rate (DR)
    # DR is same as Recall
    # FPR = 100 * ( sum(FP) / sum(FP+TN) )
    # DR  = 100 * ( sum(TP) / sum(TP+FN) )

    # results = (true_positives, false_positives, true_negatives, false_negatives)
    TP = results[0]
    FP = results[1]
    TN = results[2]
    FN = results[3]
    #print(f"[+] New Results: nTP: {TP}, nFP: {FP}, nTN: {TN}, nFN: {FN}")

    precision = recall = FPR = 0
    # Precision, Positive Predictive Value (PPV)
    # precision = TP / (TP + FP)
    if (float(TP) + float(FP)) != 0:
        precision = float(TP) / (float(TP) + float(FP))
    else:
        precision = 0

    # TRUE POSITIVE RATE, Recall, Sensitivity, Hit Rate, Detection Rate.
    # Recall = TP / (TP + FN)
    if (float(TP) + float(FN)) != 0:
        recall = TP / (float(TP) + float(FN))
    else:
        recall = 0

    # FALSE POSITIVE RATE == (Fall-out, False Alarm Ratio)
    if (float(FP) + float(TN)) != 0:
        FPR = float(FP) / (float(FP) + float(TN))
    else:
        FPR = 0

    nab = False

    filename = dataFile.split("/")[-1]
    # RANDOM ARTIFICIAL DATA
    if not nab:
        #print("[+] FileName: %s, #DataPoints: %d, #OriginalAnomalies:%d, TP: %.4f, FP: %.4f, TN: %.4f, FN: %.4f, " \
        #      "recall: %.4f, precision: %.4f, FPR: %.4f\n" % \
        #      (filename, len(lst_time), len(lst_original_anomaly_indices), TP, FP, TN, FN, recall, precision, FPR))
        return TP, FP, TN, FN, recall, precision, FPR

    plt.close()
    #plt.clf()
    fig = plt.figure(figsize=(12.8, 9.6))
    plt.subplot(3, 1, 1)
    plt.xlabel("Sliding Time Window")
    plt.ylabel("Anomaly Score")
    plt.title("Anomaly Score Graph\n#Windows: %d, window: %d sec, "
              "win_slide: %d sec, m1: %.2f, m2: %.2f, m3: %.2f, "
              "alpha: %.2f, gamma: %.2f, epsilon: %.2f" %
              ((len(lst_anomaly_scores_T) + 3), time_window_in_seconds, time_window_shift, m1, m2, m3, alpha, gamma, epsilon))
    plt.grid(True)
    plt.plot(lst_anomaly_scores_T, 'b', label='f(w)')  # f(w)
    plt.plot(lst_anomaly_runningavg, 'r', label=r"$(\mu_{w-1} + \alpha \sigma_{w-1})$")  # nu_{w-1} + alpha*sigma{w-1}
    plt.legend(loc='upper left')
    plt.subplot(3, 1, 2)
    # plt.xlabel("Sliding Time Window")
    plt.ylabel(r"$f(w) - \mu_{w-1} + \alpha \sigma_{w-1}$")
    plt.plot(lst_delta, 'g', label="Delta")  # delta, difference between f(w) and moving averages
    plt.plot(epsilon * np.ones(len(lst_delta)), 'y', label="Epsilon")
    plt.legend(loc='upper left')

    plt.subplot(3, 1, 3)
    plt.xlabel("Time")
    plt.ylabel(r"Synthetic Data - CPU Usage")
    plt.title("File: %s, DataPoints: %d, DetectRate(Recall): %.4f, correct_detection_count: %d, "
              "total_number_of_anomalies: %d" %
              (filename, len(lst_time), detection_rate, correct_detection_counter, lst_original_anomaly_indices.size))
    # plt.plot(lst_time_series_a, 'g', label="CPU")
    plt.plot(lst_time, lst_time_series_a, 'bo', label='CPU_Artificial_data')
    plt.plot(lst_time, 2 * np.ones(len(lst_time_series_a)), 'r-', label="Lower Bound on Anomalous")
    plt.legend(loc='upper left')

    '''
        Uncomment below to SAVE THE FIGURE in file
    '''
    pathtostats = "/".join(dataFile.split("/")[:-2])
    pathtostats = "/".join(dataFile.split("/")[:-7]) + "/CPU_usage_plots/SyntheticCPU/anomalies/JointTimeSeries2DMatrix"

    # addition for SYNTHETIC CPU values
    imagefilename = (pathtostats + "/anomaly_score_%s.png") % filename
    plt.savefig(imagefilename, dpi=1000, bbox_inches='tight')
    plt.close(fig)

    '''
        Plotting confusion matrix for each file.
    '''
    # Try to plot confusion matrix for each of the NAB file's result
    np.set_printoptions(precision=2)

    util.plot_confusion_matrix(TP=TP, FN=FN, FP=FP, TN=TN,
                               title='TP: %d, FN: %d, FP: %d, TN: %d, %s' %
                                     (int(TP), int(FN), int(FP), int(TN), filename), normalize=False)

    plt.savefig((pathtostats+"/ConfMatrix_%s.png") % filename, format='png')

    # Plot normalized confusion matrix
    # plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
    #                      title='Normalized confusion matrix')
    #plt.show()

    # print("[+] Size of Anomaly_Scores: %d" % (len(anomaly_scores)))
    print("[+] Finished with file: %s, \n======\n" % filename)

def main():
    # if this value is True, input is CSV, otherwise input is JSON
    inputJSON=False
    numDimensions=1

    if inputJSON:
        # ali-Laptop directories
        hA_jsonsfolder = "/home/tekeoglu/MEGA/MEGAsync/uvic/ISOT-CID/logs/phase2/hypervisorA/stat/jsons/"
        hB_jsonsfolder = "/home/tekeoglu/MEGA/MEGAsync/uvic/ISOT-CID/logs/phase2/hypervisorB/stat/jsons/"

        # for JSON file input
        hA_jsons = os.listdir(hA_jsonsfolder)
        hB_jsons = os.listdir(hB_jsonsfolder)

        for iteration in range(0, 1):
            print("[---------------------------] Iteration#%d" % (iteration))
            for x in hA_jsons:
                timeseriesDetection(os.path.join(hA_jsonsfolder, x), inputJSON)

            print("[+] Done with Hypervisor-A jsons...")

            for y in hB_jsons:
                timeseriesDetection(os.path.join(hB_jsonsfolder, y), inputJSON)

            print("[+] Done with Hypervisor-B jsons...")

    else:

        # Input CSV file
        inputCSV = f"CSVs/1215_1300_Window500msec.csv"

        # Output log file
        logfile = f"CSVs/output_1215_1300_Window500msec_TimeSeries_PktCnt.log"

        #for x in csvs:
        #if '1215_1300_Window500msec' in x:
        """
        (dataFile, inputJSON,
         p_m1=0.70, p_m2=0.25, p_m3=0.05,
         p_num_bins=20, p_time_window=100, p_time_shift=20,
         p_epsilon=0.025, p_gamma=0.3, p_alpha=0.59,
         numDimensions=1)
        """
        with open(logfile, 'w') as logwriter:
            p_m1_values = np.arange(0.5, 0.8, 0.05)                 # [0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 ]
            for cur_p_m1 in p_m1_values:
                remain = 1-cur_p_m1
                cur_p_m2 = 0.6 * remain
                cur_p_m3 = 0.4 * remain
                num_bins_values = np.arange(10, 21, 10)                  # [ 10, 20]
                for cur_num_bins in num_bins_values:
                    time_window_values = np.arange(20, 121, 20)         # [20, 40, 60, 80, 100, 120]
                    for cur_time_window in time_window_values:
                        cur_time_shift = int(cur_time_window * 0.2)          # Shift is 20%, TODO: we might vary this too.
                        epsilon_values = np.arange(0.0025, 0.0900, 0.0150)     # [0.0025, 0.0175, 0.0325, 0.0475, 0.0625, 0.0775]
                        for cur_epsilon in epsilon_values:
                            gamma_values = np.arange(0.10, 0.6, 0.10)       # [0.1, 0.2, 0.3, 0.4, 0.5]
                            for cur_gamma in gamma_values:
                                alpha_values = np.arange(0.40, 0.85, 0.08)  # [0.4 , 0.48, 0.56, 0.64, 0.72, 0.8 ]
                                for cur_alpha in alpha_values:
                                    myTP, myFP, myTN, myFN, myrecall, myprecision, myFPR = \
                                        timeseriesDetection(inputCSV, inputJSON,
                                                            p_m1=cur_p_m1, p_m2=cur_p_m2, p_m3=cur_p_m3,
                                                            p_num_bins=cur_num_bins, p_time_window=cur_time_window,
                                                            p_time_shift=cur_time_shift, p_epsilon=cur_epsilon,
                                                            p_gamma=cur_gamma, p_alpha=cur_alpha,
                                                            numDimensions=1)

                                    strresults=f"TP: {myTP}\tFP: {myFP}\tTN: {myTN}\tFN: {myFN}\t" \
                                               f"Recall: {myrecall}\tPrecision: {myprecision}\tFPR:{myFPR}\t" \
                                               f"m1: {cur_p_m1}\tm2: {cur_p_m2}\tm3: {cur_p_m3}\t" \
                                               f"numBins: {cur_num_bins}\ttimeWndw: {cur_time_window}\t" \
                                               f"timeShift: {cur_time_shift}\tEpsilon: {cur_epsilon}\t" \
                                               f"Gamma: {cur_gamma}\tAlpha: {cur_alpha}"
                                    print(strresults)
                                    logwriter.write(f"{strresults}\n")
                                    logwriter.flush()


        print(f"[+] Done with {logfile}...")


def plot_precision_recall_FPR(logfile):
    lst_tp = []
    lst_fp = []
    lst_tn = []
    lst_fn = []
    lst_recall = []
    lst_precision = []
    lst_accuracy = []
    lst_f1score = []
    lst_fpr = []
    lst_m1 = []
    lst_m2 = []
    lst_m3 = []
    lst_numBins = []
    lst_timeWndw = []
    lst_timeShift = []
    lst_epsilon = []
    lst_gamma = []
    lst_alpha = []
    line_num = 0
    with open(logfile, 'r') as lf:
        for line in lf:
            # i.e.:
            # TP: 41	FP: 0	TN: 2160	FN: 498	Recall: 0.07606679035250463	Precision: 1.0	FPR:0.0	m1: 0.5	m2: 0.3	m3: 0.2	numBins: 10	timeWndw: 20	timeShift: 4	Epsilon: 0.0025	Gamma: 0.1	Alpha: 0.4
            x = re.search(r"TP: (\d+)\s+FP: (\d+)\s+TN: (\d+)\s+FN: (\d+)\s+Recall: (\d.\d+)\s+Precision: (\d.\d+|\d+)\s+"
                          r"FPR:(\d.\d+)\sm1: (\d.\d+)\sm2: (\d.\d+)\sm3: (\d.\d+)\snumBins: (\d+)\stimeWndw: (\d+)\s"
                          r"timeShift: (\d+)\sEpsilon: (\d.\d+)\sGamma: (\d.\d+)\sAlpha: (\d.\d+)", line)
            if x:
                lst_tp.append(int(x.group(1)))
                lst_fp.append(int(x.group(2)))
                lst_tn.append(int(x.group(3)))
                lst_fn.append(int(x.group(4)))

                lst_recall.append(float(x.group(5)))
                lst_precision.append(float(x.group(6)))
                lst_fpr.append(float(x.group(7)))

                # accuracy = (tp + tn) / (tp+tn+fp+fn)
                total_population = (lst_tp[-1] + (lst_tn[-1])) + (lst_fp[-1] + (lst_fn[-1]))
                lst_accuracy.append(float((lst_tp[-1] + lst_tn[-1]) / total_population))

                # f1score = 2 * ( (precision * recall) / (precision + recall) )
                if (lst_precision[-1] + lst_recall[-1]) == 0:
                    f1score = 0
                else:
                    f1score = float(2 * (lst_precision[-1] * lst_recall[-1]) / (lst_precision[-1] + lst_recall[-1]))
                lst_f1score.append(f1score)

                lst_m1.append(float(x.group(8)))
                lst_m2.append(float(x.group(9)))
                lst_m3.append(float(x.group(10)))
                lst_numBins.append(int(x.group(11)))
                lst_timeWndw.append(int(x.group(12)))
                lst_timeShift.append(int(x.group(13)))
                lst_epsilon.append(float(x.group(14)))
                lst_gamma.append(float(x.group(15)))
                lst_alpha.append(float(x.group(16)))
            else:
                print(f"[!] Problem with {line_num}!!! ")

            line_num += 1


    # Pandas Data Frame for Sorting the values in parallel
    data = pd.DataFrame(data={'TP': lst_tp, 'FP': lst_fp, 'TN': lst_tn, 'FN': lst_fn,
                              'TPR': lst_recall, 'FPR': lst_fpr,
                              'Accuracy': lst_accuracy, 'F1score': lst_f1score,
                              'm1': lst_m1, 'm2': lst_m2, 'm3': lst_m3,
                              'numBins': lst_numBins, 'timeWndw': lst_timeWndw,
                              'timeShift': lst_timeShift, 'epsilon': lst_epsilon,
                              'gamma': lst_gamma, 'alpha': lst_alpha})

    sorted_df = data.sort_values(by=['Accuracy', 'F1score', 'TPR'], ascending=False)

    """
    # Write the sorted Data Frame into file as latex Tabular.
    str = sorted_df.to_latex(index=False)
    with open('CSVs/testing_df_to_latex.txt', 'w') as f:
        f.write(str)

    # plot the values
    fig = plt.figure(figsize=(15, 20))

    ##### ROC CURVE PLOT #####
    plt.clf()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("TPR vs FPR (ROC Curve)")
    plt.grid(True)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(lst_recall, lst_fpr, marker='x', label='roc_curve')
    plt.legend(loc='upper left')
    plt.show()
    """

    ###### 5 subplots
    plt.clf()
    # plot the values
    #fig = plt.figure(figsize=(15, 20))
    #plt.tight_layout()
    plt.subplots_adjust(wspace=3.5)
    """
    plt.subplot(5, 1, 1)
    plt.xlabel("log_file_index")
    plt.ylabel("# of TP,FP,TN,FN")
    plt.title("Grid Search, Variable Parameters - TP,FP,TN,FN")
    plt.grid(True)
    plt.plot(lst_tp, 'b', label='tp')
    plt.plot(lst_fp, 'r', label='fp')
    plt.plot(lst_tn, 'g', label='tn')
    plt.plot(lst_fn, 'y', label='fn')
    plt.legend(loc='upper left')
    """
    plt.subplot(3, 1, 1)
    #plt.xlabel("log_file_index")
    plt.ylabel("Recall, Precision, FPR")
    plt.title("Grid Search, Variable Parameters - Recall, Precision, FPR")
    plt.grid(True)
    plt.plot(lst_recall, 'b--', label='recall')
    plt.plot(lst_precision, 'r--', label='precision')
    plt.plot(lst_fpr, 'g--', label='fpr')
    plt.legend(loc='upper left')

    plt.subplot(3, 1, 2)
    #plt.xlabel("log_file_index")
    plt.ylabel("m1, m2, m3")
    plt.title("Grid Search, Variable Parameters - m1, m2, m3")
    plt.grid(True)
    plt.plot(lst_m1, 'b', label='m1')
    plt.plot(lst_m2, 'r', label='m2')
    plt.plot(lst_m3, 'g', label='m3')
    plt.legend(loc='upper left')

    plt.subplot(3, 1, 3)
    plt.xlabel("Grid Search Run index")
    plt.ylabel("numBins, timeWindow, timeShift")
    plt.title("Grid Search, Variable Parameters - numBins, timeWindow, timeShift")
    plt.grid(True)
    plt.plot(lst_numBins, 'b', label='nBins')
    plt.plot(lst_timeWndw, 'r', label='window')
    plt.plot(lst_timeShift, 'g', label='shift')
    plt.legend(loc='upper left')

    plt.savefig("CSVs/output_1215_1300_Window500msec_TimeSeries_PktCnt.png", bbox_inches='tight')

if __name__ == "__main__":
    main()
    plot_precision_recall_FPR("CSVs/output_1215_1300_Window500msec_TimeSeries_PktCnt.log")
