from goatools.base import download_go_basic_obo
from goatools.obo_parser import GODag

from settings import GO_PATH


def id2int(id):
    num = id.split(":")[-1]
    return int(num)


def int2id(int_):
    return f"GO:{int_:07d}"


def get_GO_dag():
    try:
        GO_dag = GODag(obo_file=GO_PATH.as_posix())
    except Exception:
        obo_fname = download_go_basic_obo(obo=GO_PATH.as_posix())
        GO_dag = GODag(obo_file=GO_PATH.as_posix())
    return GO_dag


def filter_GO_dag_by_namespace(GO_dag, namespace):
    edges = []
    for node_str in GO_dag:
        node = GO_dag.get(node_str)
        for ch in node.children:
            if ch.namespace == namespace:
                edge = (id2int(node.id), id2int(ch.id))
                edges.append(edge)
    return sorted(set(edges))
