from numpy.lib.scimath import log
from build_graph_utils import NewsDataSet, build_graph, save_graph

print("Building the graph")
data_sets = ['2020-1', '2020-2']
loader = NewsDataSet('data/news_row', 'data/news_embed',  data_sets)
ssm, feats = build_graph(loader)
save_graph(ssm, feats, "data/2mnews")

