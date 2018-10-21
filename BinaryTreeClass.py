"""4题，重建二叉树：输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。
       假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍
       历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，
       则重建二叉树并返回。"""
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution4:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        if not pre or not tin:
            return
        root = TreeNode(pre[0])
        i = tin.index(pre[0])
        root.left = self.reConstructBinaryTree(pre[1:i+1], tin[:i])
        root.right = self.reConstructBinaryTree(pre[i+1:], tin[i+1:])
        return root


"""17题，树的子结构：输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）"""
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution17:
    def HasSubtree(self, pRoot1, pRoot2):
        # write code here
        res = False
        if not pRoot1 or not pRoot2:
            return res
        if pRoot1.val == pRoot2.val:
            res = self.IsSubTree(pRoot1, pRoot2)
        if not res:
            res = self.IsSubTree(pRoot1.left, pRoot2)
        if not res:
            res = self.IsSubTree(pRoot1.right, pRoot2)
        return res
    def IsSubTree(self, pRoot1, pRoot2):
        if not pRoot2:
            return True
        if not pRoot1:
            return False
        if pRoot1.val != pRoot2.val:
            return False
        else:
            return self.IsSubTree(pRoot1.left, pRoot2.left) and self.IsSubTree(pRoot1.right, pRoot2.right)


"""18题，二叉树镜像：操作给定的二叉树，将其变换为源二叉树的镜像。"""
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution18:
    # 返回镜像树的根节点
    def Mirror(self, root):
        # write code here
        if not root:
            return
        root.left, root.right = root.right, root.left
        self.Mirror(root.left)
        self.Mirror(root.right)
        return root


"""22题，从上往下打印二叉树：从上往下打印出二叉树的每个节点，同层节点从左至右打印。"""
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution22:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        # write code here
        if not root:
            return []
        queue = [root]
        res = []
        while queue:
            node = queue.pop(0)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            res.append(node.val)
        return res


"""23题，二叉搜索树的后续遍历序列：输入一个整数数组，判断该数组是不是某二叉搜
        树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意
        两个数字都互不相同。"""
# -*- coding:utf-8 -*-
class Solution23:
    def VerifySquenceOfBST(self, sequence):
        # write code here
        if not sequence:
            return
        n = len(sequence)
        """寻找右子树的起始位置"""
        index = 0
        for index in range(n-1):
            if sequence[index] > sequence[-1]:
                break
        """检测右子树是否都大于根"""
        for i in range(index+1,n-1):
            if sequence[i] < sequence[-1]:
                return False
        """检测左右子树是否满足"""
        left = True
        if index > 0:
            left = self.VerifySquenceOfBST(sequence[:index])
        right = True
        if index < n-1:
            right = self.VerifySquenceOfBST(sequence[index:-1])
        return left and right


"""24题，二叉树中和为某一值的路径：输入一颗二叉树的跟节点和一个整数，打印出二
        叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下
        一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组
        长度大的数组靠前)"""
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution24:
    # 返回二维列表，内部每个列表表示找到的路径
    def FindPath(self, root, expectNumber):
        # write code here
        if not root:
            return []
        if not root.left and not root.right and root.val == expectNumber:
            return [[root.val]]
        res = []
        left = self.FindPath(root.left, expectNumber - root.val)
        right = self.FindPath(root.right, expectNumber - root.val)
        for i in left+right:
            res.append([root.val] + i)
        return res


"""26题，二叉搜索树与双向链表：输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的
         双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。"""
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution26:
    def Convert(self, pRootOfTree):
        # write code here
        if not pRootOfTree:
            return
        self.res = []
        self.Mid_travel(pRootOfTree)
        if len(self.res) == 1:
            return self.res[0]
        head = self.res[0]

        for i in range(len(self.res) - 1):
            self.res[i].right = self.res[i + 1]
            self.res[i + 1].left = self.res[i]
        return head

    def Mid_travel(self, pRootOfTree):
        if not pRootOfTree:
            return
        self.Mid_travel(pRootOfTree.left)
        self.res.append(pRootOfTree)
        self.Mid_travel(pRootOfTree.right)


"""27题，字符串的排列：输入一个字符串,按字典序打印出该字符串中字符的所有排列。
        例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串
        abc,acb,bac,bca,cab和cba。"""
# -*- coding:utf-8 -*-
class Solution27:
    def Permutation(self, ss):
        # write code here
        if not ss:
            return ''
        if len(ss) == 1:
            return ss
        res = []
        for i in range(len(ss)):
            s = self.Permutation(ss[:i] + ss[i+1:])
            for j in s:
                res.append(ss[i]+j)
        return sorted(set(res))


"""38题，二叉树的深度：输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点
        （含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。"""
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution38:
    def TreeDepth(self, pRoot):
        # write code here
        if pRoot is None:
            return 0
        return max(self.TreeDepth(pRoot.left), self.TreeDepth(pRoot.right)) + 1


"""39题，平衡二叉树：输入一棵二叉树，判断该二叉树是否是平衡二叉树。"""
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution39:
    def TreeDepth(self, pRoot):
        # write code here
        if pRoot is None:
            return 0
        return max(self.TreeDepth(pRoot.left), self.TreeDepth(pRoot.right)) + 1

    def IsBalanced_Solution(self, pRoot):
        # write code here
        if pRoot is None:
            return True
        if abs(self.TreeDepth(pRoot.left) - self.TreeDepth(pRoot.right)) > 1:
            return False
        return self.IsBalanced_Solution(pRoot.left) and self.IsBalanced_Solution(pRoot.right)


"""57题，二叉树的下一个节点：给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。
        注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针"""
# -*- coding:utf-8 -*-
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None
class Solution57:
    def GetNext(self, pNode):
        # write code here
        if not pNode:
            return
        if pNode.right:
            pNode = pNode.right
            while pNode.left:
                pNode = pNode.left
            return pNode
        while pNode.next:
            if pNode.next.left is pNode:
                return pNode.next
            pNode = pNode.next
        return None


"""58题，对称的二叉树：请实现一个函数，用来判断一颗二叉树是不是对称的。
        注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。"""
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution58:
    def isSymmetrical(self, pRoot):
        # write code here
        if not pRoot:
            return True
        return self.selfIsSymmetrical(pRoot.left, pRoot.right)
    def selfIsSymmetrical(self, left, right):
        if not left and not right:
            return True
        if left and right:
            return left.val == right.val and self.selfIsSymmetrical(left.left, right.right) and self.selfIsSymmetrical(left.right, right.left)


"""59题，按之字形打印二叉树：请实现一个函数按照之字形打印二叉树，即第一行按照从
        左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序
        打印，其他行以此类推。"""
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution59:
    def Print(self, pRoot):
        # write code here
        if not pRoot:
            return []
        nodes = [pRoot]
        res = []
        sign = 1
        while nodes:
            cur_queue, next_queue = [], []
            for i in nodes:
                cur_queue.append(i.val)
                if i.left:
                    next_queue.append(i.left)
                if i.right:
                    next_queue.append(i.right)
            res.append(cur_queue[::sign])
            sign *= -1
            nodes = next_queue
        return res


"""60题，把二叉树打印成多行：从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。"""
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution60:
    # 返回二维列表[[1,2],[4,5]]
    def Print(self, pRoot):
        # write code here
        if not pRoot:
            return []
        nodes = [pRoot]
        res = []
        while nodes:
            cur_queue, next_queue = [], []
            for i in nodes:
                cur_queue.append(i.val)
                if i.left:
                    next_queue.append(i.left)
                if i.right:
                    next_queue.append(i.right)
            res.append(cur_queue)
            nodes = next_queue
        return res


"""61题，序列化二叉树：请实现两个函数，分别用来序列化和反序列化二叉树"""
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution61:
    def __init__(self):
        self.index = -1
    def Serialize(self, root):
        # write code here
        '''前序序列化：转换成字符串'''
        if not root:
            return '#,'
        return str(root.val) + ',' + self.Serialize(root.left) + self.Serialize(root.right)
    def Deserialize(self, s):
        # write code here
        '''前序反序列化，转化成二叉树'''
        self.index += 1
        l = s.split(',')
        n = len(s)
        if not s or self.index >= n or l[self.index] == '#':
            return
        root = TreeNode(int(l[self.index]))
        root.left = self.Deserialize(s)
        root.right = self.Deserialize(s)
        return root


"""62题，二叉搜索树的第k个节点：给定一棵二叉搜索树，请找出其中的第k小的结点。例如，
        （5，3，7，2，4，6，8）中，按结点数值大小顺序第三小结点的值为4。"""
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution62:
    # 返回对应节点TreeNode
    def __init__(self):
        self.mid_traces = []

    def KthNode(self, pRoot, k):
        if not pRoot or k < 1:
            return
        self.mid_travel(pRoot)
        if len(self.mid_traces) < k:
            return
        return self.mid_traces[k - 1]

    def mid_travel(self, pRoot):
        if not pRoot:
            return
        self.mid_travel(pRoot.left)
        self.mid_traces.append(pRoot)
        self.mid_travel(pRoot.right)


"""补充题，二叉树的四种遍历："""
class Solution:
    def __init__(self):
        self._alist = []

    def pre_travel(self, root):
        """前序遍历"""
        if not root:
            return
        print(root.val,)
        self.pre_travel(root.left)
        self.pre_travel(root.right)

    def mid_travel(self, root):
        """中序遍历"""
        if not root:
            return
        self.pre_travel(root.left)
        print(root.val, )
        self.pre_travel(root.right)

    def app_travel(self, root):
        """后序遍历"""
        if not root:
            return
        self.pre_travel(root.left)
        self.pre_travel(root.right)
        print(root.val, )

    def span_travel(self, root):
        """广度优先遍历"""
        if not root:
            return
        queue = [root]
        while queue:
            node = queue.pop(0)
            print(node.val,)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)