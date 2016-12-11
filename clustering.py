# clustering.py

"""
Use hierarchical clustering to cluster stocks/ETFs and evaluate results.

Selected stocks across sectors and expected to see sector clustering.

Created on Wed Nov 09 2016
@author: Linchang
"""

from math import sqrt
import pandas as pd
from PIL import Image, ImageDraw
#from statistics import mean

def euclidean_dist(series1, series2):
    if len(series1) != len(series2):
        raise Exception("Length of two series must equal.")
    series_diff = series1 - series2
    dist = series_diff.pow(2).sum()

    return sqrt(dist)


def single_link(node1, node2, distance=euclidean_dist):
    if node1.left == None:
        if node2.left == None:
            dist = distance(node1.data, node2.data)
        else:
            dist = min(single_link(node1, node2.left, distance), single_link(node1, node2.right, distance))
    else:
        if node2.left == None:
            dist = min(single_link(node1.left, node2, distance), single_link(node1.right, node2, distance))
        else:
            dist = min(single_link(node1.left, node2.left, distance), single_link(node1.left, node2.right, distance),
                       single_link(node1.right, node2.left, distance), single_link(node1.right, node2.right, distance))
    return dist


def draw_node(draw, node, x, y, scaling, labels):
    if node.get_depth() > 1:
        h1 = node.left.get_depth() * 20
        h2 = node.right.get_depth() * 20
        top = y - (h1 + h2) / 2
        bottom = y + (h1 + h2) / 2
        # Line length
        ll = node.get_width() * scaling
        # Vertical line from this cluster to children
        draw.line((x, top + h1 / 2, x, bottom - h2 / 2), fill=(255, 0, 0))

        # Horizontal line to left item
        draw.line((x, top + h1 / 2, x + ll, top + h1 / 2), fill=(255, 0, 0))

        # Horizontal line to right item
        draw.line((x, bottom - h2 / 2, x + ll, bottom - h2 / 2), fill=(255, 0, 0))

        # Call the function to draw the left and right nodes
        draw_node(draw, node.left, x+ll, top+h1/2, scaling, labels)
        draw_node(draw, node.right, x+ll, bottom-h2/2, scaling, labels)
    else:
        # If this is an endpoint, draw the item label
        draw.text((x + 5, y - 5), labels[node.index], (0, 0, 0))

class Node:
    def __init__(self, data, index, label=None, left=None, right=None):
        self.data = data
        self.index = index
        self.label = label
        self.left = left
        self.right = right

    def get_distance(self, distance=euclidean_dist, linkage=single_link):
        '''Calculate the distance of the left and right leaves of the node.
        Return 0 if end-node.'''

        if self.left == None and self.right == None:
            return 0
        else:
            return linkage(self.left, self.right, distance)

    def get_width(self, distance=euclidean_dist):
        if self.left == None and self.right == None:
            return 0
        else:
            return distance(self.left.data, self.right.data)

    def get_depth(self):
        '''Calculate the depth of one node.
        Return 1 if end-node.'''

        if self.left == None and self.right == None:
            return 1
        return max(self.left.get_depth(), self.right.get_depth()) + 1


class HierarchCluster:
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels
        self.num_nodes = data.shape[1]
        self.num_clusters = self.num_nodes

        if self.labels == None:
            self.labels = (None, ) * self.num_nodes

        self.nodes = []
        self.clusters = []
        for i in range(self.num_nodes):
            self.nodes.append(Node(data.iloc[:, i], index=i, label=labels[i]))
            self.clusters.append(Node(data.iloc[:, i], index=i, label=labels[i]))

    def build_up(self, distance=euclidean_dist, linkage=single_link):
        if len(self.clusters) == 1:
            raise Exception("Already in one cluster!")

        closest_pair = (0, 1)
        closest_distance = linkage(self.clusters[0], self.clusters[1], distance)

        for i in range(len(self.clusters)):
            for j in range(i+1, len(self.clusters)):
                dist = linkage(self.clusters[i], self.clusters[j], distance)
                if dist < closest_distance:
                    closest_distance = dist
                    closest_pair = (i, j)

        new_node_data = pd.concat([self.clusters[closest_pair[0]].data, self.clusters[closest_pair[1]].data], axis=1).mean(axis=1)
        new_node = Node(new_node_data,
                        index=self.num_nodes,
                        left=self.clusters[closest_pair[0]],
                        right=self.clusters[closest_pair[1]])

        self.num_nodes += 1
        self.num_clusters -= 1
        self.nodes.append(new_node)
        self.clusters.remove(self.clusters[max(closest_pair)]) # first remove the node with larger index to avoid conflict
        self.clusters.remove(self.clusters[min(closest_pair)])
        self.clusters.append(new_node)

    def build_to(self, target_num_clusters=1):
        if target_num_clusters >= self.num_clusters:
            raise Exception("Number of clusters is already equal or smaller than targeted number!")

        while self.num_clusters > target_num_clusters:
            self.build_up()

    def get_depth(self):
        _depth = []
        for cluster in self.clusters:
            _depth.append(cluster.get_depth())
        return max(_depth)

    def draw_dendrogram(self, plot_name='clusters.jpg'):
        # height and width
        height = self.get_depth() * 30
        width = 500
        s = self.clusters[-1].get_width()

        # width is fixed, so scale distances accordingly
        if s != 0:
            scaling = float(height - 150) / (s*8)
        else:
            scaling = 0

        # Create a new image with a white background
        img = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        draw.line((0, height/2, 10, height/2), fill=(255, 0, 0))

        # Draw the first node
        draw_node(draw, self.clusters[-1], 10, height/2, scaling, self.labels)
        img.save(plot_name, 'JPEG')

