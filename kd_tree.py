from random import randrange
from collections import namedtuple


class KD_Tree:

    class Node:
        def __init__(self, point, depth, parent, left, right):
            self.point = point
            self.depth = depth
            self.parent = parent
            self.left = left
            self.right = right

    def __init__(self, points):
        assert points, 'No points given'
        assert len(set(len(point) for point in points)) == 1, 'Points have different dimensions'
        self.k = len(points[0])
        self.root = self._build_tree(points)

    def _build_tree(self, points, depth=0):
        dimension = depth % self.k
        if not points:
            return None
        points = sorted(points, key=(lambda p: p[dimension]))
        median = len(points) // 2
        node = KD_Tree.Node(
            point=points[median],
            depth=depth,
            parent=None,
            left=self._build_tree(points[:median], depth + 1),
            right=self._build_tree(points[median + 1:], depth + 1)
        )
        if node.left is not None:
            node.left.parent = node
        if node.right is not None:
            node.right.parent = node
        return node

    def _find_last_split_point(self, point, node=None):
        if node is None:
            node = self.root
        dimension = node.depth % self.k
        if point[dimension] <= node.point[dimension]:
            if node.left is None:
                return node
            else:
                return self._find_last_split_point(point, node.left)
        else:
            if node.right is None:
                return node
            else:
                return self._find_last_split_point(point, node.right)

    def traverse_tree(self, point, node=None, result=None):
        if node is None:
            node = self.root
        if result is None:
            result = []
        result.append(node.point)
        dimension = node.depth % self.k
        if point[dimension] <= node.point[dimension]:
            if node.left is None:
                return result
            else:
                return self.traverse_tree(point, node.left, result)
        else:
            if node.right is None:
                return result
            else:
                return self.traverse_tree(point, node.right, result)

    def find_nearest_neighbors(self, point, node=None):
        if node is None:
            node = self.root
        curr_node = self._find_last_split_point(point, node=node)
        min_points = set()
        min_dist = float('Inf')
        while True:
            dimension = curr_node.depth % self.k
            # don't bother with sqrt for ordinal comparisons
            curr_dist = sum((point[i] - curr_node.point[i]) ** 2 for i in range(self.k))
            # update closest node
            if curr_dist < min_dist:
                min_points = set([curr_node.point])
                min_dist = curr_dist
            elif curr_dist == min_dist:
                min_points.add(curr_node.point)
            # check the other side of the boundary
            if min_dist > (point[dimension] - curr_node.point[dimension]) ** 2:
                if point[dimension] <= curr_node.point[dimension]:
                    other_subtree = curr_node.right
                else:
                    other_subtree = curr_node.left
                if other_subtree is not None:
                    other_points, other_dist = self.find_nearest_neighbors(point, node=other_subtree)
                    if other_dist < min_dist:
                        min_points = other_points
                        min_dist = other_dist
                    elif other_dist == min_dist:
                        min_points |= other_points
            if curr_node.point != node.point and curr_node.parent is not None:
                curr_node = curr_node.parent
            else:
                break
        if not min_points:
            return set(), float('Inf')
        else:
            return min_points, min_dist

    def to_tuple(self, node=None):
        if node is None:
            node = self.root
        return tuple([
            node.point,
            (None if node.left is None else self.to_tuple(node.left)),
            (None if node.right is None else self.to_tuple(node.right)),
        ])

    def pretty_print(self, node=None):
        if node is None:
            node = self.root
        print('{indent}{point} (d={depth})'.format(
            indent=node.depth * '  ',
            point=node.point,
            depth=node.depth,
        ))
        if node.left is not None:
            self.pretty_print(node.left)
        if node.right is not None:
            self.pretty_print(node.right)


def find_nearest_neighbors(point, centroids):
    distance = min(
        sum((x1 - x2) ** 2 for x1, x2 in zip(point, centroid))
        for centroid in centroids
    )
    return set([
        centroid for centroid in centroids
        if distance == sum((x1 - x2) ** 2 for x1, x2 in zip(point, centroid))
    ])


def get_random_point(n):
    return tuple([randrange(256) for _ in range(n)])


def test_structure():
    centroids = [(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)]
    kd_tree = KD_Tree(centroids)
    assert kd_tree.to_tuple() == (
        (7, 2),
         ((5, 4), ((2, 3), None, None), ((4, 7), None, None)),
         ((9, 6), ((8, 1), None, None), None),
    )

def test_traversal():
    centroids = [(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)]
    kd_tree = KD_Tree(centroids)
    expected = (4, 7)
    actual = kd_tree.traverse_tree((2, 8))[-1]
    assert expected == actual, 'Expected {} but got {}'.format(expected, actual)
    expected = (4, 7)
    actual = kd_tree.traverse_tree((2.5, 4.5))[-1]
    assert expected == actual, 'Expected {} but got {}'.format(expected, actual)

def test_nn():
    centroids = [(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)]
    kd_tree = KD_Tree(centroids)
    point = (3, 8)
    expected = find_nearest_neighbors(point, centroids)
    actual, _ = kd_tree.find_nearest_neighbors(point)
    assert expected == actual, 'Expected {} but got {}'.format(expected, actual)
    point = (2.5, 4.5)
    expected = find_nearest_neighbors(point, centroids)
    actual, _ = kd_tree.find_nearest_neighbors(point)
    assert expected == actual, 'Expected {} but got {}'.format(expected, actual)

def test_2d():
    # nearest neighbor test
    num_centroids = 50
    centroids = [get_random_point(2) for _ in range(num_centroids)]
    kd_tree = KD_Tree(centroids)
    num_tests = 1000
    for _ in range(num_tests):
        point = get_random_point(2)
        old = find_nearest_neighbors(point, centroids)
        new, _ = kd_tree.find_nearest_neighbors(point)
        assert old == new, 'Expected {} but got {}'.format(old, new)


def test_3d():
    # nearest neighbor test
    num_centroids = 50
    centroids = [get_random_point(3) for _ in range(num_centroids)]
    kd_tree = KD_Tree(centroids)
    num_tests = 1000
    num_multiples = 0
    for _ in range(num_tests):
        point = get_random_point(3)
        old = find_nearest_neighbors(point, centroids)
        new, _ = kd_tree.find_nearest_neighbors(point)
        assert old == new, 'Expected {} but got {}'.format(old, new)
        if len(old) > 1:
            num_multiples += 1


def main():
    test_structure()
    test_traversal()
    test_nn()
    test_2d()
    test_3d()


if __name__ == '__main__':
    main()
