import requests
import pandas as pd
import os
from path import Path
import json
import numpy as np
import re
from collections import defaultdict
from enum import Enum
from diacritic import conv_tex_diacritic


def get_graph_json():
    """
    Fetch all tags and save as json files
    """
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


def combine_force_json(src_folder='force', save_to='force.json', conv_diacritic=True):
    """
    Combine subgraphs into one graph and save it as json file.
    Both the subgraphs and the whole graph are formatted as understandable by d3.layout.force,
    which is like
    {
        nodes: {tag: {property_key: property_value, ...}},
        links: [{"source": source_tag, "target": target_tag}]
    }
    :param src_folder: the folder that contains all json files of subgraphs
    :param save_to: saving destination of the whole graph
    """
    # node: {tag: meta-data + children + parents}
    # link: {source: tag, target: tag}
    graph = {'nodes': {}, 'links': set()}
    nodes = graph['nodes']
    links = graph['links']
    keys_to_remove = ['numberOfChildren', 'size', 'depth']
    for file in Path(src_folder).glob('*.json'):
        with open(file, 'r') as f:
            sub_graph = json.load(f)
        idx2tag = {}
        sub_nodes = sub_graph['nodes']
        if conv_diacritic:
            for node in sub_nodes:
                node['chapter'] = conv_tex_diacritic(node['chapter'])
                node['section'] = conv_tex_diacritic(node['section'])
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


def parse(chapters_tex='chapters.tex', skip_topics=None, skip_chapters=None):
    """
    :return: 
        {'chapters': [chapter_names], 
        'topics':[topic_names], 
        'topic_ranges':[(chapter_start, chapter_end)]}
        (inclusive end)
    """
    class State(Enum):
        TOPIC = 0
        CHAPTER = 1

    chapters = []
    topic_names = []
    topic_ranges_txt = []
    inv_topic_range_tbl = {}
    with open(chapters_tex, 'r') as f:
        topic_patt = re.compile(r'^[A-Za-z\s]+$')
        chapter_patt = re.compile(r'^\\item *\\hyperref\[(.+?)\]{(.+?)}$')
        prev = State.TOPIC
        ch_start = ''
        ch_end = ''
        for line in f:
            line = line.strip()
            topic_match = topic_patt.match(line)
            chapter_match = chapter_patt.match(line)
            if topic_match:
                t_name = topic_match.group(0)
                topic_names.append(t_name)
                if prev == State.CHAPTER:
                    ch_end = chapters[-1]
                    inv_topic_range_tbl[ch_end] = len(topic_ranges_txt)
                    topic_ranges_txt.append([ch_start,ch_end])
                prev = State.TOPIC
            elif chapter_match:
                ch_name = chapter_match.group(2)

                chapters.append(ch_name)
                if prev == State.TOPIC:
                    ch_start = ch_name
                    inv_topic_range_tbl[ch_start] = len(topic_ranges_txt)
                prev = State.CHAPTER
        ch_end = chapters[-1]
        inv_topic_range_tbl[ch_end] = len(topic_ranges_txt)
        topic_ranges_txt.append([ch_start,ch_end])

    chapter_ids = list(range(1, len(chapters)+1))

    if skip_topics is not None:
        for topic in skip_topics:
            t_idx = topic_names.index(topic)
            ch_start = topic_ranges_txt[t_idx][0]
            ch_start_idx = chapters.index(ch_start)
            ch_end = topic_ranges_txt[t_idx][1]
            ch_end_idx = chapters.index(ch_end)
            del chapters[ch_start_idx:(ch_end_idx+1)]
            del chapter_ids[ch_start_idx:(ch_end_idx+1)]
            del topic_names[t_idx]
            del topic_ranges_txt[t_idx]
    if skip_chapters is not None:
        for chapter in skip_chapters:
            # could be removed when removing topics already
            if chapter in chapters:
                ch_idx = chapters.index(chapter)
                if chapter in inv_topic_range_tbl:
                    # need to change the topic range
                    t_idx = inv_topic_range_tbl[chapter]
                    ch_start, ch_end = topic_ranges_txt[t_idx]
                    if ch_start == ch_end:
                        del topic_names[t_idx]
                        del topic_ranges_txt[t_idx]
                    else:
                        if ch_start == chapter:
                            ch_new_start = chapters[ch_idx + 1]
                            topic_ranges_txt[t_idx][0] = ch_new_start
                            inv_topic_range_tbl[ch_new_start] = t_idx
                            del inv_topic_range_tbl[ch_start]
                        else:
                            assert ch_end == chapter
                            ch_new_end = chapters[ch_idx - 1]
                            topic_ranges_txt[t_idx][1] = ch_new_end
                            inv_topic_range_tbl[ch_new_end] = t_idx
                            del inv_topic_range_tbl[ch_end]

                del chapters[ch_idx]
                del chapter_ids[ch_idx]

    assert len(topic_names) == len(topic_ranges_txt)
    assert len(chapters) == len(chapter_ids)
    topic_ranges = [[chapters.index(t_range[0]), chapters.index(t_range[1])] for t_range in topic_ranges_txt]
    return {'chapters': chapters, 'chapter_ids': chapter_ids, 'topics': topic_names, 'topic_ranges': topic_ranges}


def chord_diagram_matrices(force_json='force.json', chapters_tex='stacks-project/chapters.tex', save_to='chord_diagram.json'):
    """
    Create the adjacency matrices between chapters and topics.
    In detail: 
    :param force_json: path of the tag-reference graph
    :param chapters_tex: path of chapters.tex
    :param save_to: where to save the adjacency matrices
    """
    # rows/columns: topics/chapters
    # chapters: 3~88
    # mat(i,j) = #links from topic/chapter i to topic/chapter j
    # return {'chapters': chapters, 'chapter_matrix': chapter_matrix}

    with open(force_json, 'r') as f:
        graph = json.load(f)
    nodes = graph['nodes']
    links = graph['links']

    skipped_chapters = ['Introduction', 'Conventions']
    skip_topics = ['Miscellany']
    chapter_info = parse(chapters_tex=chapters_tex, skip_topics=skip_topics, skip_chapters=skipped_chapters)
    chapters = chapter_info['chapters']
    n_chapters = len(chapters)
    chapter_mat = np.zeros((n_chapters, n_chapters), dtype=int)

    topics = chapter_info['topics']
    n_topics = len(topics)
    chapter_idx_ranges = chapter_info['topic_ranges']
    chapter_id2idx = {ch: i for i, ch in enumerate(chapter_info['chapter_ids'])}

    t2c = np.zeros((n_topics, n_chapters), dtype=int)
    for i, rng in enumerate(chapter_idx_ranges):
        start, stop = rng
        stop += 1
        t2c[i, start:stop] = 1

    # return row/column index for mat
    def tag2cid(tag):
        book_id = nodes[tag]['book_id']
        chapter_id = int(re.match(r'\d+', book_id).group(0))
        return chapter_id

    for link in links:
        source = link['source']
        target = link['target']
        source_id = tag2cid(source)
        target_id = tag2cid(target)

        if source_id in chapter_id2idx and target_id in chapter_id2idx:
            chapter_mat[chapter_id2idx[source_id], chapter_id2idx[target_id]] += 1

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
    print('(for scatter.html) x1 = %f\tslope = %f' % (x1, slope))
    # print(top_theorems)
    top_theorems = top_theorems.iloc[:-1]
    if graph is not None:
        nodes = graph['nodes']
        chapters = [nodes[tag]['chapter'] for tag in top_theorems.index]
        sections = [nodes[tag]['section'] for tag in top_theorems.index]
        book_ids = [nodes[tag]['book_id'] for tag in top_theorems.index]
        top_theorems['book_id'] = book_ids
    top_theorems.to_csv(save_to)


def reverse_force_json(force_json_path, save_to='force_reversed.json'):
    with open(force_json_path, 'r') as f:
        graph = json.load(f)
    nodes = graph['nodes']
    links = graph['links']
    for tag, node in nodes.items():
        node['children'], node['parents'] = node['parents'], node['children']
    for link in links:
        link['source'], link['target'] = link['target'], link['source']
    with open(save_to, 'w') as f:
        json.dump(graph, f)


def chapter_section_table(collapsible_folder, save_to='tag_chapter_section.csv'):
    table = pd.DataFrame(columns=['chapter', 'section'])
    for file in Path(collapsible_folder).glob('*.json'):
        with open(file, 'r') as f:
            root = json.load(f)
        chapters = root['children']
        for chapter in chapters:
            assert chapter['nodeType'] == 'chapter'
            sections = chapter['children']
            for section in sections:
                assert section['nodeType'] == 'section'
                for tag in section['children']:
                    # if tag['type'] != 'chapter' and tag['type'] != 'section':
                    if tag['tag'] not in table.index:
                        table.loc[tag['tag']] = [chapter['tag'], section['tag']]
    table.index.rename('tag', inplace=True)
    table.sort_index(inplace=True)
    table.to_csv(save_to)


def collapsible_from_tag(tag, graph_file, chapter_section_file, direction='children', include_section=True, save_to=None):
    with open(graph_file, 'r') as f:
        graph = json.load(f)
    chapter_section_info = pd.read_csv(chapter_section_file, index_col='tag')
    nodes = graph['nodes']

    # tag names only first
    collapsible = nodes[tag].copy()
    collapsible['nodeType'] = 'root'
    if save_to is None:
        save_to = '%s_%s_collapsible.json' % (tag, direction)
    children = collapsible[direction]
    del collapsible['children']
    del collapsible['parents']

    root_chapter_set = set()
    chapter_section_dict = defaultdict(set)
    section_tag_dict = defaultdict(set)
    chapter_tag_dict = defaultdict(set)
    for child in children:
        chapter, section = chapter_section_info.loc[child]
        root_chapter_set.add(chapter)
        chapter_section_dict[chapter].add(section)
        section_tag_dict[section].add(child)
        chapter_tag_dict[chapter].add(child)

    collapsible['children'] = [nodes[chapter].copy() for chapter in root_chapter_set]
    for chapter_info in collapsible['children']:
        del chapter_info['parents']
        del chapter_info['children']
        chapter = chapter_info['tag']
        if include_section:
            chapter_info['children'] = [nodes[section].copy() for section in chapter_section_dict[chapter]]
            for section_info in chapter_info['children']:
                del section_info['parents']
                del section_info['children']
                section = section_info['tag']
                section_info['children'] = [nodes[tag].copy() for tag in section_tag_dict[section]]
                for tag_info in section_info['children']:
                    del tag_info['parents']
                    del tag_info['children']
        else:
            chapter_info['children'] = [nodes[tag].copy() for tag in chapter_tag_dict[chapter]]
            for tag_info in chapter_info['children']:
                del tag_info['parents']
                del tag_info['children']

    with open(save_to, 'w') as f:
        json.dump(collapsible, f, sort_keys=True)


if __name__ == '__main__':
    # get_graph_json()
    combine_force_json(src_folder='force', save_to='force.json')
    chord_diagram_matrices('force.json', save_to='chord_diagram.json')
    tag_scatter('force.json', save_to='scatter.csv')
    with open('force.json', 'r') as f:
        graph = json.load(f)
        important_theorems('scatter.csv', n_top=12, save_to='top_theorems.csv', graph=graph)
    chapter_section_table('collapsible', save_to='tag_chapter_section.csv')
    collapsible_from_tag('01UA', graph_file='force.json', chapter_section_file='tag_chapter_section.csv', include_section=False, direction='parents')
