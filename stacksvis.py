import requests
import pandas as pd
import os
from path import Path
import json
import numpy as np
import re


def get_graph_json():
    tags = pd.read_csv('https://raw.githubusercontent.com/stacks/stacks-project/master/tags/tags', header=None, comment='#')[0]
    types = ['force', 'cluster', 'collapsible']
    url_tmpl = 'http://stacks.math.columbia.edu/data/tag/{tag}/graph/{graph_type}'
    for graph_type in types:
        print(graph_type)
        if not os.path.exists(graph_type):
            os.mkdir(graph_type)
        for tag in tags:
            url = url_tmpl.format(tag=tag, graph_type=graph_type)
            r = requests.get(url)
            if r.status_code == 200:
                with open('%s/%s.json' % (graph_type, tag), 'w', encoding=r.encoding) as f:
                    f.write(r.text)
            else:
                print(url)


def combine_force_json(folder='force', save_to='force.json'):
    # node: {tag: meta-data + children + parents}
    # link: {source: tag, target: tag}
    graph = {'nodes': {}, 'links': set()}
    nodes = graph['nodes']
    links = graph['links']
    keys_to_remove = ['numberOfChildren', 'size', 'depth']
    for file in Path(folder).glob('*.json'):
        with open(file, 'r') as f:
            sub_graph = json.load(f)
        idx2tag = {}
        sub_nodes = sub_graph['nodes']
        sub_links = sub_graph['links']
        for idx, node in enumerate(sub_nodes):
            tag = node['tag']
            idx2tag[idx] = tag
            if tag not in nodes:
                nodes[tag] = node
                nodes[tag]['children'] = set()
                nodes[tag]['parents'] = set()
                for k in keys_to_remove:
                    if k in nodes[tag]:
                        del nodes[tag][k]

        for link in sub_links:
            source = idx2tag[link['source']]
            target = idx2tag[link['target']]
            links.add((source, target))
            nodes[source]['children'].add(target)
            nodes[target]['parents'].add(source)
    for tag, node in graph['nodes'].items():
        node['children'] = list(node['children'])
        node['parents'] = list(node['parents'])
    graph['links'] = [{'source': source, 'target': target} for source, target in links]
    with open(save_to, 'w') as f:
        json.dump(graph, f, sort_keys=True, indent=4)


def chord_diagram_matrices(force_json_path, save_to='chord_diagram.json'):
    # rows/columns: topics/chapters
    # chapters: 3~88
    # mat(i,j) = #links from topic/chapter i to topic/chapter j
    # return {'chapters': chapters, 'chapter_matrix': chapter_matrix}

    with open(force_json_path, 'r') as f:
        graph = json.load(f)
    nodes = graph['nodes']
    links = graph['links']

    chapters = ["Set Theory", "Categories", "Topology", "Sheaves on Spaces", "Sites and Sheaves", "Stacks", "Fields", "Commutative Algebra", "Brauer groups", "Homological Algebra", "Derived Categories", "Simplicial Methods", "More on Algebra", "Smoothing Ring Maps", "Sheaves of Modules", "Modules on Sites", "Injectives", "Cohomology of Sheaves", "Cohomology on Sites", "Differential Graded Algebra", "Divided Power Algebra", "Hypercoverings", "Schemes", "Constructions of Schemes", "Properties of Schemes", "Morphisms of Schemes", "Cohomology of Schemes", "Divisors", "Limits of Schemes", "Varieties", "Topologies on Schemes", "Descent", "Derived Categories of Schemes", "More on Morphisms", "More on Flatness", "Groupoid Schemes", "More on Groupoid Schemes", "\\'Etale Morphisms of Schemes", "Chow Homology and Chern Classes", "Intersection Theory", "Picard Schemes of Curves", "Adequate Modules", "Dualizing Complexes", "Algebraic Curves", "Resolution of Surfaces", "Semistable Reduction", "Fundamental Groups of Schemes", "\\'Etale Cohomology", "Crystalline Cohomology", "Pro-\\'etale Cohomology", "Algebraic Spaces", "Properties of Algebraic Spaces", "Morphisms of Algebraic Spaces", "Decent Algebraic Spaces", "Cohomology of Algebraic Spaces", "Limits of Algebraic Spaces", "Divisors on Algebraic Spaces", "Algebraic Spaces over Fields", "Topologies on Algebraic Spaces", "Descent and Algebraic Spaces", "Derived Categories of Spaces", "More on Morphisms of Spaces", "Pushouts of Algebraic Spaces", "Groupoids in Algebraic Spaces", "More on Groupoids in Spaces", "Bootstrap", "Quotients of Groupoids", "Simplicial Spaces", "Formal Algebraic Spaces", "Restricted Power Series", "Resolution of Surfaces Revisited", "Formal Deformation Theory", "Deformation Theory", "The Cotangent Complex", "Algebraic Stacks", "Examples of Stacks", "Sheaves on Algebraic Stacks", "Criteria for Representability", "Artin's axioms", "Quot and Hilbert Spaces", "Properties of Algebraic Stacks", "Morphisms of Algebraic Stacks", "Cohomology of Algebraic Stacks", "Derived Categories of Stacks", "Introducing Algebraic Stacks", "More on Morphisms of Stacks"]
    n_chapters = len(chapters)
    chapter_mat = np.zeros((n_chapters, n_chapters), dtype=int)

    # see chapters.tex
    topics = ["Preliminaries", "Schemes", "Topics in Scheme Theory", "Algebraic Spaces", "Topics in Geometry", "Deformation Theory", "Algebraic Stacks"]
    n_topics = len(topics)
    chapter_idx_ranges = [(0, 22), (22, 38), (38, 50), (50, 66), (66, 70), (70, 74), (76, 86)]

    t2c = np.zeros((n_topics, n_chapters), dtype=int)
    for i, rng in enumerate(chapter_idx_ranges):
        start, stop = rng
        t2c[i, start:stop] = 1
        # print(topics[i])
        # for j in range(start, stop):
        #     print('\t(%d) %s' % (j+3, chapters[j]))

    # return row/column index for mat
    def tag2idx(tag):
        book_id = nodes[tag]['book_id']
        chapter = int(re.match(r'\d+', book_id).group(0))
        return chapter - 3

    for link in links:
        source = link['source']
        target = link['target']
        source_idx = tag2idx(source)
        target_idx = tag2idx(target)

        if source_idx < n_chapters and target_idx < n_chapters:
            # skip chapter 89 (Example) and chapter 94 (Obsolete)
            chapter_mat[source_idx, target_idx] += 1
    topic_mat = t2c.dot(chapter_mat).dot(t2c.T)
    with open(save_to, 'w') as f:
        json.dump({'topics': topics, 'topic_matrix': topic_mat.tolist(), 'chapters': chapters, 'chapter_matrix': chapter_mat.tolist()}, f, sort_keys=True, indent=4)


def tag_scatter(force_json_path, save_to='scatter.csv'):
    with open(force_json_path, 'r') as f:
        nodes = json.load(f)['nodes']

    tags = nodes.keys()
    scatter_df = pd.DataFrame(np.nan, index=tags, columns=['type', 'referring', 'referred'])
    for tag, node in nodes.items():
        scatter_df.loc[tag] = [node['type'], len(node['children']), len(node['parents'])]
    scatter_df = scatter_df[(scatter_df.referring > 0) & (scatter_df.referred > 0)].sort_index()
    scatter_df.to_csv(save_to, index_label='tag', float_format='%d')


def important_theorems(scatter_csv_path, n_top, save_to='top_theorems.csv', graph=None):
    scatter = pd.read_csv(scatter_csv_path, index_col='tag')
    top_referred = scatter.sort_values(by='referred', ascending=False).head(n_top).referred
    top_referring = scatter.sort_values(by='referring', ascending=False).head(n_top).referring
    slope = np.mean(top_referred.values/top_referring.values)
    scatter['score'] = scatter['referring'] + scatter['referred']/slope
    top_theorems = scatter.sort_values(by='score', ascending=False).head(n_top+1)
    x1 = (top_theorems.iloc[-1].score + top_theorems.iloc[-2].score) / 2
    print('(for top_theorems.html) x1 = %f\tslope = %f' % (x1, slope))
    # print(top_theorems)
    top_theorems = top_theorems.iloc[:-1]
    if graph is not None:
        nodes = graph['nodes']
        chapters = [nodes[tag]['chapter'] for tag in top_theorems.index]
        sections = [nodes[tag]['section'] for tag in top_theorems.index]
        book_ids = [nodes[tag]['book_id'] for tag in top_theorems.index]
        top_theorems['chapter'] = chapters
        top_theorems['section'] = sections
        top_theorems['book_id'] = book_ids
    top_theorems.to_csv(save_to)


if __name__ == '__main__':
    # get_graph_json()
    # combine_force_json(folder='force', save_to='force.json')
    # chord_diagram_matrices('force.json', save_to='chord_diagram.json')
    # tag_scatter('force.json', save_to='scatter.csv')
    with open('force.json', 'r') as f:
        graph = json.load(f)
        important_theorems('scatter.csv', n_top=10, save_to='top_theorems.csv', graph=graph)
