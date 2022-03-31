import argparse
from args import get_default_ArgumentParser, process_common_arguments
from dataprun import GenerateWL, GenerateDomain2IP
import logging
#from DomainNameSimilarity import getDomainSimilarityCSR
from ip_to_ip import ip_to_ip
from time import time
from label import Label, LabelFiles
from domain2IP_matrix import getDomainResolveIpCSR
from ClientDomain import getClientQueriesDomainCSR
from PathSim import PathSim
from scipy.sparse import csr_matrix
from affinity_matrix import affinity_matrix, converge

def print_nnz_info(M: csr_matrix, name: str):
  """ Prints nnz: number of non zeros info
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
  message  =("Runs a hetergeneous information network on the supplied data.")
  parser = get_default_ArgumentParser(message)
  """why is this happening at all? Why would we need to parse this message?"""
  parser.add_argument("--dns_files", type=str, nargs='+', required=True,
    help="The dns log file(s) to use.")
  parser.add_argument("--netflow_files", type=str, nargs='+', required=True,
    help="The netflow log file(s) to use.")
    """These two are required. and need to be added to the argument when running code """
  parser.add_argument("--domain_similarity_threshold", type=float, default=0.5,
    help="The threshold to use to determine if a domain similarity is " +
      "represented or zeroed out.")
  parser.add_argument("--affinity_threshold", type=float, default=0.5,
    help="If affinity is below threshold we set to zero.")
    """The affinity matrix is commonly referred to as the similarity matrix. Organi
    zes techniques used to organize the mutual similarities between a set of data points.
    Have a threshold on how similar the traffic is benign or malicious."""

  # Exclude certain matrices
  parser.add_argument('--exclude_domain_similarity', action='store_true',
    help="If set, will not compute domain similarity.")
  parser.add_argument('--exclude_ip2ip', action='store_true',
    help="If set, will not compute domain similarity.")
  parser.add_argument('--exclude_domain2ip', action='store_true',
    help="If set, will not compute domainResolveIp.")
  parser.add_argument('--exclude_clientQdomain', action='store_true',
    help="If set, will not compute clientQueryDomain.")

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

  RL, domain2index, ip2index =  GenerateWL(FLAGS.dns_files)
  """This function act as a wrapper to be generate valid list (Whitelist) of Domains
  and IPs with Assigned index. Returns:
  Dictionary read logs.
  Dictionary of all valid domains
  Dictionary of all valid client and answered IPs"""
  #print(RL)#hard to
  domain2ip = GenerateDomain2IP(RL, domain2index)
  """This function return a dictionary maps the relation on (Answer)IPs. Return:
  A dictionary maps Domain to IPs: """

  numDomains = len(domain2ip)"""dictionary of domains to ips"""
  domainMatrixSize = max(domain2index.values()) + 1 """dictionary of all valid domains"""
  ipMatrixSize = max(ip2index.values()) + 1 """dictionary of all valid client and answered IPs"""
  numIps = len(ip2index)"""dictionary of all valid client and answered IPs"""

  print("Number of domains in domain2ip " + str(numDomains))
  print("Number of domains in domain2index " + str(numDomains))
  print("Number of ips in ip2index " + str(numIps))
  print("Domain matrix size: " + str(domainMatrixSize))


  ################## Labels #######################################
  if FLAGS.good is not None and FLAGS.bad is not None:
    label = LabelFiles(FLAGS.good, FLAGS.bad)
  else:
    label = Label()
  labels = label.get_domain_labels(domain2index)
  logging.info("Shape of labels: " + str(labels.shape))

  """Returns label matrix for domains.

  If domain is on blacklist --> [1, 0]
  If domain is on whitelist --> [0, 1]
  If domain is on both blacklist and whitelist --> [1, 0]
  If domain is on neither blacklist and whitelist --> [0, 0]"""

  ################### Domain similarity ##########################
  #if not FLAGS.exclude_domain_similarity:
  #  time1 = time()
  #  domainSimilarityCSR = getDomainSimilarityCSR(domain2index,
  #                                          domain2ip,
  #                                          FLAGS.domain_similarity_threshold)
  #  logging.info("Time for domain similarity " +
  #               "{:.2f}".format(time() - time1))
  #  print_nnz_info(domainSimilarityCSR, "domain similarity")
  #else:
  #  logging.info("Excluding domain similarity")
  #  domainSimilarityCSR = None

"""the function that gets us the domainSimilarityCSR matrix comes from DomainNameSimilarity.py
Returns compressed sparse row matrix of which domains are similar"""
  #################### ip to ip ###################################
  if not FLAGS.exclude_ip2ip:
    time1 = time()
    ip2ip = ip_to_ip(ip2index, FLAGS.netflow_files)
    logging.info("Time for ip2ip " +
                 "{:.2f}".format(time() - time1))
    print_nnz_info(ip2ip, "ip2ip")
  else:
    logging.info("Excluding ip2ip")
    ip2ip = None


  ################### Domain resolve ip #############################
  if not FLAGS.exclude_domain2ip:
    time1 = time()
    domainResolveIp = getDomainResolveIpCSR(domain2ip, domain2index, ip2index)
    logging.info("Time for domainResolveIp " +
                 "{:.2f}".format(time() - time1))
    print_nnz_info(domainResolveIp, "domainResolveIp")
  else:
    logging.info("Excluding domainResolveIp")
    domainResolveIp = None

  ################## Client query domain ############################
  if not FLAGS.exclude_clientQdomain:
    time1 = time()
    clientQueryDomain = getClientQueriesDomainCSR(RL, domain2index, ip2index)
    logging.info("Time for clientQueryDomain " +
                 "{:.2f}".format(time() - time1))
    print_nnz_info(clientQueryDomain, "clientQueryDomain")

    """the function that gets us the clientQueryDomain matrix comes from ClientDomain.py
    Returns compressed sparse row matrix of which clients queried which domains"""
  ################### CNAME ########################################
  cnameCSR = None # Not complete


  ################### Creating metapaths ############################
  if clientQueryDomain is not None:
    time1 = time()
    domainQueriedByClient = clientQueryDomain.transpose()
    domainQueriedBySameClient = domainQueriedByClient * clientQueryDomain
    logging.info("Time to domainQueriedBySameClient " +
                 "{:.2f}".format(time() - time1))
  else:
    domainQueriedBySameClient = None

  if domainResolveIp is not None:
    time1 = time()
    ipResolvedToDomain = domainResolveIp.transpose()
    domainsShareIp = domainResolveIp * ipResolvedToDomain
    logging.info("Time to create domainShareIp " +
                 "{:.2f}".format(time() - time1))
  else:
    domainsShareIp = None

  domainsFromSameClientSegment = None # Not done

  if domainResolveIp is not None:
    R = domainResolveIp
    Rt = R.transpose()
    fromSameAttacker = R * Rt * R * Rt
  else:
    fromSameAttacker = None



  ################### Combine Matapaths ############################
  timeTotal = time()
  M = csr_matrix((domainMatrixSize, domainMatrixSize))
  #if domainSimilarityCSR is not None:
  #  time1 = time()
  #  M = M + PathSim(domainSimilarityCSR)
  #  logging.info("Time pathsim domainSimilarityCSR " +
  #               "{:.2f}".format(time() - time1))
  if cnameCSR is not None:
    time1 = time()
    M = M + PathSim(cnamneCSR)
    logging.info("Time pathsim cnameCSR " +
                 "{:.2f}".format(time() - time1))
  if domainQueriedBySameClient is not None:
    time1 = time()
    M = M + PathSim(domainQueriedBySameClient)
    logging.info("Time pathsim domainQueriedBySameClient " +
                 "{:.2f}".format(time() - time1))
  if domainsShareIp is not None:
    time1 = time()
    M = M + PathSim(domainsShareIp)
    logging.info("Time pathsim domainShareIp " +
                 "{:.2f}".format(time() - time1))
  if domainsFromSameClientSegment is not None:
    time1 = time()
    M = M + PathSim(domainsFromSameClientSegment)
    logging.info("Time pathsim domainsFromSameClientSegment " +
                 "{:.2f}".format(time() - time1))
  if fromSameAttacker is not None:
    time1 = time()
    M = M + PathSim(fromSameAttacker)
    logging.info("Time pathsim fromSameAttacker " +
                 "{:.2f}".format(time() - time1))
  logging.info("Time to calculate PathSim " +
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
  """Computes the affinity matrix for matrix M.  Each row is compared to each
    other row and a gaussian is used to calculate how close the rows are.
    If the value is below a threshold, we set it to zero (which won't
    be represented in the sparse matrix)"""
  ################## Iterating to convergence ########################
  time1 = time()
  F = converge(M, labels, FLAGS.mu, FLAGS.tol)
  print("Y F domain")
  for i in range(len(F)):
    print(labels[i,:], F[i,:], index2domain[i])


if __name__ == '__main__':
  main()
