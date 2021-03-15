#!/usr/bin/env python
"""
Use DPKT to read in a pcap file and print out the contents of the packets
This example is focused on the fields in the Ethernet Frame and IP packet
"""
import dpkt
import datetime
import socket
from dpkt.compat import compat_ord
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import time
import os

# from dpkt.utils import mac_to_str, inet_to_str
# These 2 funcitons are from dpkt.utils
def mac_to_str(address):
    """Convert a MAC address to a readable/printable string
       Args:
           address (str): a MAC address in hex form (e.g. '\x01\x02\x03\x04\x05\x06')
       Returns:
           str: Printable/readable MAC address
    """
    return ':'.join('%02x' % compat_ord(b) for b in address)


def inet_to_str(inet):
    """Convert inet object to a string
        Args:
            inet (inet struct): inet network address
        Returns:
            str: Printable/readable IP address
    """
    # First try ipv4 and then ipv6
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)


def calculateLabel(curWinTimestamp):
    # [+] First attack: Exfiltrate Historian Data (5 mins attack + 10 mins sleep) x 4 cycles
    # [+] Second attack: Disrupt sensor readings and process (3 mins + 10 mins sleep) x 5 cycles
    # Timestamps in PCAP are 8 hours behind (GMT+8)
    # https://docs.google.com/document/d/17kTdPUT3YfgKsUA65VkFoFJ3k4B8xIEn/edit
    label = 0       # Attack_True = 1, Attack_False = 0
    attack_times = {"10:05-10:20": "false",     # Pre-attack capture for 15 mins
                    "10:20-10:30": "false",     # TODO: Infiltrate SCADA WS via USB thumb drive w/malware NO NETWORK TRAFFIC. Double Check Network Traffic in this interval for malicious packets?
                    "10:30-10:35": "true",      # [+] Exfiltrate Historian Data (5mins)
                    "10:35-10:45": "false",     # Sleep (10 mins)
                    "10:45-10:50": "true",      # [+] Exfiltrate Historian Data (5mins)
                    "10:50-11:00": "false",     # Sleep (10 mins)
                    "11:00-11:05": "true",      # [+] Exfiltrate Historian Data (5mins)
                    "11:05-11:15": "false",     # Sleep (10 mins)
                    "11:15-11:20": "true",      # [+] Exfiltrate Historian Data (5mins)
                    "11:20-12:30": "false",     # "Sleep" starts 11:20 / "Rest 60 mins" starts @11:30
                    "12:30-12:33": "true",      # [+] Disrupt Sensor and Actuator (3mins)
                    "12:33-12:43": "false",     # Sleep (10 mins)
                    "12:43-12:46": "true",      # [+] Disrupt Sensor and Actuator (3mins)
                    "12:46-12:56": "false",     # Sleep (10 mins)
                    "12:56-12:59": "true",      # [+] Disrupt Sensor and Actuator (3mins)
                    "12:59-13:09": "false",     # Sleep (10 mins)
                    "13:09-13:12": "true",      # [+] Disrupt Sensor and Actuator (3mins)
                    "13:12-13:22": "false",     # Sleep (10 mins)
                    "13:22-13:25": "true",      # [+] Disrupt Sensor and Actuator (3mins)
                    "13:25-13:30": "false",     # Sleep (5 mins)
                    "13:30-13:45": "false",     # Capture post-attack pcap (15 mins)
    }

    for at in attack_times:
        adjustedWinTimestamp = curWinTimestamp + datetime.timedelta(hours=8)
        start = at.split("-")[0]
        end = at.split("-")[1]
        startTS = datetime.datetime(year=2019, month=12, day=6,
                                    hour=int(start.split(":")[0]), minute=int(start.split(":")[1]))
        endTS = datetime.datetime(year=2019, month=12, day=6,
                                  hour=int(end.split(":")[0]), minute=int(end.split(":")[1]))
        if adjustedWinTimestamp >= startTS and adjustedWinTimestamp < endTS:
            if attack_times[at] == "false":
                label = 0
            elif attack_times[at] == "true":
                label = 1
            break
        else:
            label = 0
    return label


def calc_num_of_uniq_SRC_DST_IPs(lst_packets_curwindow):
    """Calcute number of unique source & destination IP addresses in the current window

    Paramaters
    ----------
    lst_packets_curwindow:  list
        Ethernet packets in current time-window

    Returns
    -------
    len(lst_src_ips): int
        Number of unique source IPs
    len(lst_dst_ips): int
        Number of unique destination IPs
    """
    lst_src_ips=[]
    lst_dst_ips=[]
    n_nonIP4packets=0

    for e in lst_packets_curwindow:
        # Make sure the Ethernet frame contains an IP packet
        if not isinstance(e.data, dpkt.ip.IP):
            #print('Non IP Packet type not supported %s\n' % e.data.__class__.__name__)
            # TODO: We might have to filter for IP packets and ignore the following ones in the calculations;
            # TODO: Non-IPv4 protocols: {ARP, IP6, LLC}
            n_nonIP4packets += 1
            continue
        else:
            ip = e.data
            src_ip_addr = inet_to_str(ip.src)
            dst_ip_addr = inet_to_str(ip.dst)
            if src_ip_addr not in lst_src_ips:
                lst_src_ips.append(src_ip_addr)
            if dst_ip_addr not in lst_dst_ips:
                lst_dst_ips.append(dst_ip_addr)

    return len(lst_src_ips), len(lst_dst_ips), n_nonIP4packets


def compute_pcap_to_timeseries(pcap, cdw, time_interval = 1):
    """Print out information about each packet in a pcap
       Args:
           pcap: dpkt pcap reader object (dpkt.pcap.Reader)
      output:
          result = [[firstWindow], ..... [veryLastWindow]] while
        [each window] --> [averageOfFeature1, AverageofFeature2, AverageofFeature3]
        each window is of timeWindow units of seconds
    """
    packetNumber = 0
    timeZero = 0
    curTime = 0
    headTime = 0
    timeWindow = time_interval          # width of each window in seconds
    curWindowStartTimestamp = ""
    curWindowEndTimestamp = ""
    windowNumber = 0

    ETHlensSum = 0
    windowPktCnt = 0

    lst_packets_curwindow = []
    cur_win_results = []

    # For each packet in the pcap process the contents
    for timestamp, buf in pcap:
        curTime = datetime.datetime.utcfromtimestamp(timestamp)
        packetNumber += 1
        if packetNumber == 1:
            timeZero = curTime
            headTime = curTime

        #windowPktCnt += 1
        timePassed = curTime - headTime
        timePassedSeconds = timePassed.total_seconds()
        eth = dpkt.ethernet.Ethernet(buf)

        # Delta between Current Packets time and Win
        if timePassedSeconds > timeWindow:
            try:
                avgLen = ETHlensSum / windowPktCnt
            except ZeroDivisionError:
                print('Warning: no packet read in this window')
                avgLen = 0.0
            except:
                print("Some other error occurs and all average result set to 0")
                avgLen = 0.0
            finally:
                # now we can calculate other features for the current window on list of packets in
                # variable "lst_packets_curwindow"
                """
                tmp_sum=0
                for p in lst_packets_curwindow:
                    tmp_sum += len(p)
                print(f"[+] Sum_bytes: {tmp_sum}, NumPkts: {len(lst_packets_curwindow)}, AvgBytes: {tmp_sum/len(lst_packets_curwindow)}")
                """
                n_src_ips, n_dst_ips, n_nonIP4packets = calc_num_of_uniq_SRC_DST_IPs(lst_packets_curwindow)

                # TODO: more features for each window, implement as a function and call here.
                # add timestamp as well, maybe with window number
                windowMidTimestamp = headTime + datetime.timedelta(seconds=(timeWindow / 2))
                label = calculateLabel(windowMidTimestamp)
                print(f"[+] Window# {windowNumber}, "
                      f"Delta: {timePassedSeconds}, "       # curTime - headTime
                                                            # time distance from current packet to,
                                                            # beginning of current window)
                      f"medianTimestamp: {windowMidTimestamp}, "
                      f"avgPktLen: {avgLen}, "
                      f"ETHlensSumCurWndw: {ETHlensSum}, "
                      f"packetsInCurrentWindow: {windowPktCnt}, "
                      f"numSrcIPs: {n_src_ips}, "
                      f"numDstIPs: {n_dst_ips}, "
                      f"nonIP4packets: {n_nonIP4packets}, "
                      f"Label: {label}")

                cur_win_results.append([windowNumber,
                                windowMidTimestamp,
                                avgLen,
                                ETHlensSum,
                                windowPktCnt,
                                n_src_ips,
                                n_dst_ips,
                                n_nonIP4packets,
                                label])

                # writing cur_win_results to CSV file
                for r in cur_win_results:
                    t = (r[1] + datetime.timedelta(hours=8)).strftime("%m/%d/%Y, %H:%M:%S.%f")
                    cdw.writerow({'WinIndex': str(r[0]),
                                  'MidTime': t,
                                  'AvgEthBytes': str(r[2]),
                                  'SumEthBytes': str(r[3]),
                                  'PktCnt': str(r[4]),
                                  'numSrcIPs': str(r[5]),
                                  'numDstIPs': str(r[6]),
                                  'nonIP4packets': str(r[7]),
                                  'AttackLabel': str(r[8])}  # TODO: label is wrong, check the problem
                                 )
                # Delete the contents of current window list
                lst_packets_curwindow.clear()
                cur_win_results.clear()
                print("=======")

            headTime += datetime.timedelta(seconds=timeWindow)      # shift the "head time" of the Window to next.
            windowNumber += 1
            ETHlensSum = (len(eth))
            windowPktCnt = 1        # reset the number of packets in current window to 1.
            lst_packets_curwindow.append(eth)

        else:
            # Here we extract the future from each packet. In this case packet-length.
            lst_packets_curwindow.append(eth)
            ETHlensSum += (len(eth))
            windowPktCnt += 1
    return # results

#########################################################################################

def main():
    time_interval = 0.5       # width of each window in seconds

    """Open up a test pcap file and print out the packets"""
    pcapFolderPrefix = "/SWAT_PCAPS"                                # MAC Folder

    # each pcap is 15 mins.
    pcapFiles = [pname for pname in os.listdir(pcapFolderPrefix) if ".pcap" in pname]
    pcapFiles.sort()

    pcap_start_time = (int(pcapFiles[-1].split('_')[-1][8:-7]))
    hours = int((int(pcapFiles[-1].split('_')[-1][8:-7])) / 100)
    mins = (pcap_start_time % 100)
    if mins + 15 >= 60:
        hours += 1
        hours = str(hours)
        mins = (mins + 15) % 60
        mins = str(mins)

        if len(mins) == 1:
            mins = f"0{mins}"
        if len(hours) == 1:
            hours = f"0{hours}"
    else:
        mins = mins + 15

    pcap_end_time = f"{hours}{mins}"

    outputCSVfile = f"CSVs/{pcapFiles[0].split('_')[-1][8:-7]}_{pcap_end_time}_Window{int(time_interval*1000)}msec.csv"

    with open(outputCSVfile, 'w+', newline='') as csvfile:
        headerrow = ['WinIndex', 'MidTime', 'AvgEthBytes', 'SumEthBytes',
                     'PktCnt', 'numSrcIPs', 'numDstIPs', 'nonIP4packets',
                     'AttackLabel']
        cdw = csv.DictWriter(csvfile, fieldnames=headerrow)
        cdw.writeheader()

        for p in pcapFiles:
            with open(f"{pcapFolderPrefix}/{p}", 'rb') as f:
                pcap = dpkt.pcap.Reader(f)
                compute_pcap_to_timeseries(pcap, cdw, time_interval)
    return len(pcapFiles)

if __name__ == '__main__':
    t1 = time.time()
    n_pcaps = main()
    t2 = time.time()
    print(f"[+] Finished {n_pcaps} PCAPs in {float(t2-t1)/60} minutes...")
