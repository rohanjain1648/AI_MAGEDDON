# AI_MAGEDDON
THIS IS A HACKATHON PROJECT BY TEAM DR


INTRODUCTION:

INTRUSION DETECTION:

Intrusion detection is the process of identifying malicious activity or unauthorized access in a computer network.

Intrusions can include attacks such as DoS (Denial-of-Service), U2R (User to Root), R2L (Remote to user), PROBE.

There are two main types of intrusion detection: signature-based and anomaly-based.

Intrusion detection is a critical component of network security because it can help identify and respond to security incidents in real-time.

DoS (Denial-of-Service) attacks aim to disrupt the availability of network resources by flooding the network with traffic.

Probe attacks are used to gather information about a network's vulnerabilities and configuration.




Dataset Description:

DATASET: NSL-KDD

The dataset used for intrusion detection research is an improved version of the KDDCup 99 data set, which was widely used as one of the few publicly available data sets for evaluating intrusion detection systems (IDSs) until the release of the NSL-KDD data set. The NSL-KDD data set includes network traffic data from a local area network (LAN) simulation environment that is designed to resemble a typical US Air Force LAN.

The NSL-KDD data set has been preprocessed to eliminate redundancy and inconsistency in the original KDDCup 99 data set.


Redefined Problem Statement:

• Given a network traffic dataset, develop a machine learning model that can accurately classify each network connection as either normal or malicious. The model should be able to identify different types of attacks, such as Denial of Service (DoS), Probe, User to Root (U2R), and Remote to Local (R2L) attacks.



INTRODUCTION Contd...

R2L (Remote-to-Local) attacks target vulnerabilities in the network's authentication and access controls to gain unauthorized access.

U2R (User-to-Root) attacks exploit vulnerabilities in the system to gain root-level access.

We used NSL-KDD dataset, which is a modified version of the KDD Cup 1999 dataset, which is a widely used benchmark dataset for intrusion detection research.

The NSL-KDD dataset includes a preprocessed version of the KDD Cup 1999 dataset that has been normalized and de-duplicated to reduce noise and improve accuracy.

The dataset is widely used in intrusion detection research as a benchmark for evaluating the performance of different machine learning algorithms such as Random Forest, Naive Bayes', KNN(K-Nearest Neighbour), SVM (Support Vector Machines), Decision Trees, Logistic Regression, ANN (Artificial Neural Networks).



DATA DESCRIPTION:

Used two different datasets, one is train dataset and the other is test dataset.

Variables in train dataset are:

"duration": the length of time in seconds that the connection lasted.

"protocol_type": the protocol used for the connection (tcp, udp, etc.).

"service": the type of service being used (ftp_data, http, etc.).

"flag": a flag indicating the status of the connection (SF, SO, REJ, etc.).

"sre bytes": the number of data bytes from the source to the destination in the connection.

"dst_bytes": the number of data bytes from the destination to the source in the connection.

"land": a flag indicating if the connection is from/to the same host/port.

"wrong_fragment": the number of "wrong" fragments in the connection.

"urgent": the number of urgent packets in the connection


• Variables in test dataset are:

Protocol type (e.g., tep, icmp)

Service (e.g., ftp_data, http, eco_i)

Flag (e.g., SF, REJ, RSTO)

Various numerical features (e.g., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

Attack type (e.g., neptune, normal, saint, mscan)
Shape of Training Dataset: (125972, 43)

Shape of Testing Dataset: (22543, 43)



Columns of Existing Training and Test Data Set:

1. duration

2. protocol_type

3. service

4. flag

5. src_bytes

6. dst_bytes

7. land

8. wrong fragment

9. urgent

10. hot

11. num failed logins

12. logged_in

13. num_compromised

14. root shell

15. su_attempted

16. num_root

17. num file creations

18. num_shells

19. num access files

20. num outbound_cmds

21. is host login

22. is guest_login

23. count

24. srv count

25. serror_rate

26. srv_serror_rate

27. rerror rate

28. srv_rerror_rate

29. same srv rate

30. diff srv rate

32. dst host count

31. srv diff host_rate

33. dst host srv count

34. dst host same srv rate

35. dst host diff_srv_rate

36. dst host same src port rate

37. dst_host_srv_diff_host_rate

38. dst host serror rate

39. dst host srv serror rate

40. dst_host_rerror_rate

41. dst host srv rerror rate

42. attack

43. level



DATA PREPROCESSING:

Data preprocessing is a crucial step in machine learning that involves transforming raw data into a suitable format for building predictive models. The main goal of data preprocessing is to ensure that the data is clean, consistent, and in the right format to be used by machine learning algorithms.

The following has been performed in order:

Checking for Null values

Checking for Duplicate Rows

Column Names for Training and Test Data Set

Identify Categorical Features

Add six missing Categories from Train Set to Test Set




We've used the attacks, which are classified into the following four categories:

Denial-of-service (DoS): An attacker tries to make a machine or network resource unavailable to its intended users by overwhelming it with a flood of traffic or by sending malformed packets that cause the resource to crash or become unavailable.

Probe: An attacker sends packets to gather information about a target network or system. This can involve port scanning, fingerprinting, or other techniques that can reveal vulnerabilities or provide information that can be used in a subsequent attack.

Remote-to-local (R2L): An attacker attempts to gain unauthorized access to a target system from a remote location. This can involve exploiting vulnerabilities in network protocols or applications, or using brute-force techniques to crack passwords.

User-to-root (U2R): An attacker who has already gained access to a user account on a target system tries to elevate their privileges to gain root access and take control of the system.



We've used the attacks, which are classified into the following four categories:

Denial-of-service (DoS): An attacker tries to make a machine or network resource unavailable to its intended users by overwhelming it with a flood of traffic or by sending malformed packets that cause the resource to crash or become unavailable.

Probe: An attacker sends packets to gather information about a target network or system. This can involve port scanning, fingerprinting, or other techniques that can reveal vulnerabilities or provide information that can be used in a subsequent attack.

Remote-to-local (R2L): An attacker attempts to gain unauthorized access to a target system from a remote location. This can involve exploiting vulnerabilities in network protocols or applications, or using brute-force techniques to crack passwords.

User-to-root (U2R): An attacker who has already gained access to a user account on a target system tries to elevate their privileges to gain root access and take control of the system.
