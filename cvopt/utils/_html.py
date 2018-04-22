from html.parser import HTMLParser
from ._base import scale
from ..utils import _htmlsrc as hs

class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.cr_tag = None
        self.start_lines = []
        self.end_lines = []
        self.data = []
        
    def feed(self, data, tgt, tgt_type):
        self.tgt = tgt
        self.tgt_type = tgt_type
        super().feed(data)
        
    def handle_starttag(self, tag, attrs):
        if self.tgt_type == "factor":
            if tag == self.tgt:
                self.start_lines.append(self.getpos()[0])

    def handle_endtag(self, tag):
        if self.tgt_type == "factor":
            if tag == self.tgt:
                self.end_lines.append(self.getpos()[0])
                
def select_html(file, tgt):
    parser = MyHTMLParser()
    parser.feed(data=file, tgt=tgt, tgt_type="factor")
    return parser.start_lines, parser.end_lines, 

def arrang_graph_file(graph, model_id, add_head, pjs, search_algo, n_iter):
    n_addline = 0
    with open(graph) as f:
        lines = f.readlines()

    title_start, _ = select_html(file="".join(lines), tgt="title")
    title_start = title_start[0]

    _, head_end = select_html(file="".join(lines), tgt="head")
    head_end = head_end[0]

    body_start, _ = select_html(file="".join(lines),tgt="body")
    body_start = body_start[0]

    title_start, _ = select_html(file="".join(lines), tgt="title")
    title_start = title_start[0]
    lines[title_start-1] = lines[title_start-1].replace("Bokeh Plot", "Search Results ("+ model_id +")")

    lines.insert(head_end-1, add_head)
    n_addline += 1

    pjs = pjs.replace("REP_AGENT_TYPE", hs.pjs_setting[search_algo]["AGENT_TYPE"])
    pjs = pjs.replace("REP_CIRCLE_TYPE", hs.pjs_setting[search_algo]["CIRCLE_TYPE"])
    pjs = pjs.replace("REP_N_CIRCLE", 
                      str(int(scale(val=n_iter, from_range=(hs.n_iter_setting["min"],  hs.n_iter_setting["max"]), 
                                    to_range=(hs.pjs_setting[search_algo]["min_n_circle"], hs.pjs_setting[search_algo]["max_n_circle"], 
                                    )))),
                      )

    lines.insert(body_start+n_addline, pjs)
    n_addline += 1

    with open(graph, "w") as f:
        f.writelines(lines)