"""二叉树"""


class TreeNode:
    def __init__(self, val=None):
        self.val = val
        self.left = None
        self.right = None


class BinaryTree:
    def __init__(self):
        self.root = None

    def add(self, elem):
        node = TreeNode(elem)
        if self.root is None:
            self.root = node
        else:
            cur_root = self.root
            queue = [cur_root]
            while queue:
                cur_root = queue.pop(0)
                if cur_root.left is None:
                    cur_root.left = node
                    return
                else:
                    queue.append(cur_root.left)
                if cur_root.right is None:
                    cur_root.right = node
                    return
                else:
                    queue.append(cur_root.right)

    def pop(self):
        """删除最下层最后一个节点"""
        if not self.root:
            return
        pass

    def pre_travel(self, root):
        """前序遍历"""
        if not root:
            return
        print(root.val, end=' ')
        self.pre_travel(root.left)
        self.pre_travel(root.right)

    def mid_travel(self, root):
        """中序遍历"""
        if not root:
            return
        self.pre_travel(root.left)
        print(root.val, end=' ')
        self.pre_travel(root.right)

    def app_travel(self, root):
        """后序遍历"""
        if not root:
            return
        self.pre_travel(root.left)
        self.pre_travel(root.right)
        print(root.val, end=' ')

    def span_travel(self, root):
        """广度优先遍历"""
        if not root:
            return
        queue = [root]
        while queue:
            node = queue.pop(0)
            print(node.val, end=' ')
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)


if __name__ == '__main__':
    tr = BinaryTree()
    for i in range(10):
        tr.add(i)
    tr.span_travel(tr.root)
    print()
    tr.pre_travel(tr.root)
    print()
    tr.mid_travel(tr.root)
    print()
    tr.app_travel(tr.root)
