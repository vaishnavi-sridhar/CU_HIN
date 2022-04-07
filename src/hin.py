import argparse
from args import get_default_ArgumentParser, process_common_arguments
from dataprun import GenerateWL, GenerateDomain2IP
import logging
from DomainNameSimilarity import getDomainSimilarityCSR
from ip_to_ip import ip_to_ip
from time import time
from label import Label, LabelFiles
from domain2IP_matrix import getDomainResolveIpCSR
from ClientDomain import getClientQueriesDomainCSR
from PathSim import PathSim
from scipy.sparse import csr_matrix
from affinity_matrix import affinity_matrix, converge
import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
import scipy


def print_nnz_info(M: csr_matrix, name: str):
    """ Prints nnz info
  """
    n = M.shape[0]
    m = M.shape[1]
    nnz = M.nnz
    total = n * m
    percent = float(100 * nnz) / total
    logging.info("nonzero entries (" + str(nnz) + "/" + str(total) +
                 ") in " + name + " " +
                 "{:.4f}".format(percent) + "%")


def main():
    message = ("Runs a hetergeneous information network on the supplied data.")
    parser = get_default_ArgumentParser(message)
    parser.add_argument("--dns_files", type=str, nargs='+', required=True,
                        help="The dns log file(s) to use.")
    parser.add_argument("--netflow_files", type=str, nargs='+', required=True,
                        help="The netflow log file(s) to use.")
    parser.add_argument("--domain_similarity_threshold", type=float, default=0.5,
                        help="The threshold to use to determine if a domain similarity is " +
                             "represented or zeroed out.")
    parser.add_argument("--affinity_threshold", type=float, default=0.5,
                        help="If affinity is below threshold we set to zero.")

    # Exclude certain matrices
    parser.add_argument('--exclude_domain_similarity', action='store_true',
                        help="If set, will not compute domain similarity.")
    parser.add_argument('--exclude_ip2ip', action='store_true',
                        help="If set, will not compute domain similarity.")
    parser.add_argument('--exclude_domain2ip', action='store_true',
                        help="If set, will not compute domainResolveIp.")
    parser.add_argument('--exclude_clientQdomain', action='store_true',
                        help="If set, will not compute clientQueryDomain.")

    # START: Added this arg to exclude domain similarity and cname - to compute performance of the hindom with and
    # without these 2 metapaths
    parser.add_argument('--exclude_domain_similarity_cname', action='store_true',
                        help="If set, will not compute domain similarity(S) and cname meta-paths (C).")
    parser.add_argument('--exclude_cname', action='store_true',
                        help="If set, will not compute cname meta-paths (C).")
    # END

    parser.add_argument('--mu', type=float, default=0.5,
                        help="Mu parameter used to balance between original labels and new info.")
    parser.add_argument('--tol', type=float, default=0.001,
                        help="Tolerance parameter that determines when converges is close enough.")

    parser.add_argument('--good', type=str, default=None,
                        help="Location of file with good domains.")
    parser.add_argument('--bad', type=str, default=None,
                        help="Location of file with bad domains.")

    FLAGS = parser.parse_args()
    process_common_arguments(FLAGS)

    logging.info("DNS files: " + str(FLAGS.dns_files))
    logging.info("Netflow files: " + str(FLAGS.netflow_files))

    # START : Modified GenerateWL to return CNameRecords - List<List<String>> while reading data from dns file
    RL, domain2index, ip2index, CNameRecords = GenerateWL(FLAGS.dns_files)
    # END

    # START : Converting CName records from dns file into list of tuples(domain combinations)
    final_domain_pairs = generate_cname_combo(CNameRecords, domain2index)
    # END

    # print(RL) #Commenting out for faster runtime
    domain2ip = GenerateDomain2IP(RL, domain2index)  # maps domain to resolved ip list

    numDomains = len(domain2ip)
    numDomainsindex = len(domain2index)
    domainMatrixSize = max(domain2index.values()) + 1
    ipMatrixSize = max(ip2index.values()) + 1
    numIps = len(ip2index)
    print("Number of domains in domain2ip ", str(numDomains))
    print("Number of domains in domain2index ", str(numDomainsindex))
    print("Number of ips in ip2index ", str(numIps))
    print("Domain matrix size: ", str(domainMatrixSize))

    ################## Labels #######################################
    if FLAGS.good is not None and FLAGS.bad is not None:
        label = LabelFiles(FLAGS.good, FLAGS.bad)
    else:
        label = Label()
    labels = label.get_domain_labels(domain2index)
    logging.info("Shape of labels: " + str(labels.shape))

    ################### Domain similarity (S) ##########################
    domainSimilarityCSR = None
    print("Newly added flag:", FLAGS.exclude_domain_similarity_cname)
    if not FLAGS.exclude_domain_similarity or not FLAGS.exclude_domain_similarity_cname:
        time1 = time()
        domainSimilarityCSR = getDomainSimilarityCSR(domain2index,
                                                     FLAGS.domain_similarity_threshold)
        logging.info("Time for domain similarity " +
                     "{:.2f}".format(time() - time1))
        print_nnz_info(domainSimilarityCSR, "domain similarity")
    else:
        print("Excluding domain similarity")

    ################### ip to ip ###################################
    if not FLAGS.exclude_ip2ip:
        time1 = time()
        ip2ip = ip_to_ip(ip2index, FLAGS.netflow_files)
        print("Time for ip2ip " +
              "{:.2f}".format(time() - time1))
        print_nnz_info(ip2ip, "ip2ip")
    else:
        print("Excluding ip2ip")
        ip2ip = None
    ################# Client query domain (Q) ############################
    if not FLAGS.exclude_clientQdomain:
        time1 = time()
        clientQueryDomain = getClientQueriesDomainCSR(RL, domain2index, ip2index)
        print("Time for clientQueryDomain " +
              "{:.2f}".format(time() - time1))
        print_nnz_info(clientQueryDomain, "clientQueryDomain")

    ################### Domain resolve ip (R) #############################
    if not FLAGS.exclude_domain2ip:
        time1 = time()
        domainResolveIp = getDomainResolveIpCSR(domain2ip, domain2index, ip2index)
        print("Time for domainResolveIp " +
              "{:.2f}".format(time() - time1))
        print_nnz_info(domainResolveIp, "domainResolveIp")
    else:
        print("Excluding domainResolveIp")
        domainResolveIp = None

    # ################### CNAME matrix (C) ########################################
    # START: Cname matrix creation
    """We create the Cname matrix using a pandas dataframe where the columns and
    rows are the domain names and whenever the domain name in column has a cname
    relationship with the domain name in the row then we make that position in the
    matrix a value of 1.Then we print out the number of Cname connections (count
    non zero values in matrix)"""
    cname_sparsed = None
    if not FLAGS.exclude_cname or not FLAGS.exclude_domain_similarity_cname:
        cname_sparsed = generate_cname_csr(domain2index, final_domain_pairs)
    else:
        print("Excluding Cname")

    # END: Cname matrix creation
    ################### Creating metapaths ############################
    if clientQueryDomain is not None:
        time1 = time()
        domainQueriedByClient = clientQueryDomain.transpose()
        domainQueriedBySameClient = domainQueriedByClient * clientQueryDomain  # Q*Q^T
        print("Time to domainQueriedBySameClient " +
              "{:.2f}".format(time() - time1))
    else:
        domainQueriedBySameClient = None

    if domainResolveIp is not None:
        time1 = time()
        ipResolvedToDomain = domainResolveIp.transpose()
        domainsShareIp = domainResolveIp * ipResolvedToDomain
        print("Time to create domainShareIp " +
              "{:.2f}".format(time() - time1))
    else:
        domainsShareIp = None

    domainsFromSameClientSegment = None  # Not done

    fromSameAttacker = None  # Not mentioned in paper
    # if domainResolveIp is not None:
    #     R = domainResolveIp
    #     Rt = R.transpose()
    #     fromSameAttacker = R * Rt * R * Rt
    # else:
    #     fromSameAttacker = None

    ################### Combine Matapaths ############################
    timeTotal = time()
    M = csr_matrix((domainMatrixSize, domainMatrixSize))

    if domainSimilarityCSR is not None:
      time1 = time()
      M = M + PathSim(domainSimilarityCSR)
      logging.info("Time pathsim domainSimilarityCSR " +
                   "{:.2f}".format(time() - time1))

    # START: Adding Cname metapath (C) to matrix M which is affinity matrix
    """Once we have the Cname matrix, we do not have to do create the metapath since
    the Cname is already a metapath. Then, we combine the Cname metapath with the
    other metapaths using the PathSim function. Afterwards, the matrix M will have
    the Cname metapath and is included in the affinity matrix. """
    if cname_sparsed is not None:
        time1 = time()
        M = M + PathSim(cname_sparsed)
        logging.info("Time pathsim cnameCSR " +
                     "{:.2f}".format(time() - time1))
    # END: Adding Cname meta-path to matrix M which is affinity matrix

    # Q*Q^T
    if domainQueriedBySameClient is not None:
        time1 = time()
        M = M + PathSim(domainQueriedBySameClient)
    print("Time pathsim domainQueriedBySameClient " +
          "{:.2f}".format(time() - time1))

    # R*R^T
    if domainsShareIp is not None:
        time1 = time()
        M = M + PathSim(domainsShareIp)
    print("Time pathsim domainShareIp " +
          "{:.2f}".format(time() - time1))

    if domainsFromSameClientSegment is not None:
        time1 = time()
        M = M + PathSim(domainsFromSameClientSegment)
    print("Time pathsim domainsFromSameClientSegment " +
          "{:.2f}".format(time() - time1))

    if fromSameAttacker is not None:
        time1 = time()
        M = M + PathSim(fromSameAttacker)
    print("Time pathsim fromSameAttacker " +
          "{:.2f}".format(time() - time1))
    print("Time to calculate PathSim " +
          "{:.2f}".format(time() - time1))

    ################## Creating Affinity Matrix #########################
    time1 = time()
    M = affinity_matrix(M, FLAGS.affinity_threshold)
    logging.info("Time to calculate affinity " +
                 "{:.2f}".format(time() - time1))
    nnz = M.nnz
    total = domainMatrixSize * domainMatrixSize
    logging.info("nonzero entries (" + str(nnz) + "/" + str(total) +
                 ") in M after affinity " + str(float(100 * nnz) / total) + "%")

    index2domain = {v: k for k, v in domain2index.items()}

    ################## Iterating to convergence ########################
    time1 = time()
    F = converge(M, labels, FLAGS.mu, FLAGS.tol)
    print("Y F domain")
    ''' START - Writing classification output - converged F values, labels and corresponding domain name to a log file
    '''
    f = open("convergence_log.txt", "a")
    for i in range(len(F)):
        log = labels[i, :], F[i, :], index2domain[i]
        f.write(''.join(map(str, log)) + "\n")
    f.close()
    ''' END '''


'''
START - Function to create csr matrix for cname meta path based on domain pair combinations
Parameter1 : domain2index dictionary , key: domain name string, value: index (integer)
Parameter2 : final_domain_pairs - list of tuples
'''


def generate_cname_csr(domain2index, final_domain_pairs):
    cname_matrix = pd.DataFrame(0, index=list(domain2index.values()), columns=list(domain2index.keys()))
    print("CName shape:", cname_matrix.shape)
    for (i, j) in final_domain_pairs:
        cname_matrix.iat[i, j] = 1
        cname_matrix.iat[j, i] = 1
    cname_sparsed = scipy.sparse.csr_matrix(cname_matrix.values)
    return cname_sparsed


''' END '''

'''
START - Function
Input: CNameRecords- List of List of domain names belonging to a single cname record
        domain2index- Dictionary of key-domain name , value - index/integers starting from 0
Output: final_lst: List of tuples containing all possible 2 domain combinations with replacement
Example : combinations_with_replacement(‘ABCD’, 2) ==> [AA, AB, AC, AD, BB, BC, BD, CC, CD, DD]
'''


def generate_cname_combo(cname_records, domain2index):
    list_domain_tuples = []
    tuples_list = []
    for innerList in cname_records:
        list_domain_tuples.append(list(combinations_with_replacement(innerList, 2)))
    for item in list_domain_tuples:
        for i in range(3):
            tuples_list.append(item[i])
    print("List of tuples:", tuples_list)
    final_domain_pairs = []
    for tuple_item in tuples_list:
        if tuple_item[0] in domain2index.keys() and tuple_item[1] in domain2index.keys():
            final_domain_pairs.append((domain2index[tuple_item[0]], domain2index[tuple_item[1]]))
    print(final_domain_pairs)
    return final_domain_pairs


if __name__ == '__main__':
    main()
