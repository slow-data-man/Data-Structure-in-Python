"""5题，用两个栈实现队列：用两个栈来实现一个队列，完成队列的Push和Pop操作。
    队列中的元素为int类型。"""
class Solution5:
    def __init__(self):
        self.list1 = []
        self.list2 = []
    def push(self, node):
        # write code here
        self.list1.append(node)
    def pop(self):
        # return xx
        if not self.list1:
            return
        while self.list1:
            self.list2.append(self.list1.pop())
        res = self.list2.pop()
        while self.list2:
            self.list1.append(self.list2.pop())
        return res


"""15题，反转链表：输入一个链表，反转链表后，输出新链表的表头。"""
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution15:
    # 返回ListNode
    def ReverseList(self, pHead):
        # write code here
        if not pHead or not pHead.next:
            return pHead
        p1, p2 = pHead.next, pHead.next.next
        pHead.next = None
        while p2:
            p1.next = pHead
            pHead = p1
            p1, p2 = p2, p2.next
        p1.next = pHead
        pHead = p1
        return pHead


"""16题，合并两个排序的链表：输入两个单调递增的链表，输出两个链表合成后的链表，
        当然我们需要合成后的链表满足单调不减规则。"""
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution16:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        # write code here
        if not pHead1:
            return pHead2
        if not pHead2:
            return pHead1
        if pHead1.val <= pHead2.val:
            pHead = pHead1
            pHead1 = pHead1.next
        else:
            pHead = pHead2
            pHead2 = pHead2.next
        rear = pHead
        while pHead1 and pHead2:
            if pHead1.val <= pHead2.val:
                rear.next = pHead1
                pHead1 = pHead1.next
                rear = rear.next
            else:
                rear.next = pHead2
                pHead2 = pHead2.next
                rear = rear.next
        if not pHead1:
            rear.next = pHead2
        else:
            rear.next = pHead1
        return pHead


"""19题，顺时针打印矩阵：输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，
        例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
        则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10."""
class Solution19:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):
        # write code here
        res = []
        while matrix:
            res += matrix.pop(0)
            matrix = self.rotate(matrix)
        return res

    def rotate(self, matrix):
        if not matrix:
            return
        rows, cols = len(matrix), len(matrix[0])
        for i in range(rows):
            j = 0
            while j < cols - j - 1:
                matrix[i][j], matrix[i][cols - j - 1] = matrix[i][cols - j - 1], matrix[i][j]
                j += 1
        res = [[None] * rows for i in range(cols)]
        for i in range(rows):
            for j in range(cols):
                res[j][i] = matrix[i][j]
        return res


"""19题，顺时针打印矩阵"""
class Solution19:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):
        # write code here
        res = []
        while matrix:
            res += matrix.pop(0)
            matrix = self.rotate(matrix)
        return res

    def rotate(self, matrix):
        if not matrix:
            return
        rows, cols = len(matrix), len(matrix[0])
        for i in range(rows):
            j = 0
            while j < cols - j - 1:
                matrix[i][j], matrix[i][cols - j - 1] = matrix[i][cols - j - 1], matrix[i][j]
                j += 1
        res = [[None] * rows for i in range(cols)]
        for i in range(rows):
            for j in range(cols):
                res[j][i] = matrix[i][j]
        return res


"""20题，包含min函数的栈：定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小
       元素的min函数（时间复杂度应为O（1））。"""
class Solution20:
    def __init__(self):
        self.stack = []
    def push(self, node):
        # write code here
        return self.stack.append(node)
    def pop(self):
        # write code here
        return self.stack.pop()
    def top(self):
        # write code here
        return self.stack[-1]
    def min(self):
        # write code here
        return min(self.stack)


"""21题，栈的压入弹出序列：输入两个整数序列，第一个序列表示栈的压入顺序，
       请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。
       例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的
       一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。
       （注意：这两个序列的长度是相等的）"""
class Solution:
    def IsPopOrder(self, pushV, popV):
        # write code here
        res = []
        for i in pushV:
            res.append(i)
            if popV[0] == res[-1]:
                res.pop()
                popV.pop(0)
        while res:
            if res[-1] == popV[0]:
                res.pop()
                popV.pop(0)
            else:
                return False
        return True


"""25题，复杂链表的复制"""
# class RandomListNode:
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None
class Solution:
    # 返回 RandomListNode
    def Clone(self, pHead):
        # write code here
        if pHead is None:
            return
        self.CloneNode(pHead)
        self.CloneRandom(pHead)
        return self.Cut(pHead)

    def CloneNode(self, pHead):
        if not pHead:
            return
        p = pHead
        while p:
            cloneNode = RandomListNode(p.label)
            cloneNode.next = p.next
            p.next = cloneNode
            p = cloneNode.next

    def CloneRandom(self, pHead):
        if not pHead:
            return
        p = pHead
        while p:
            pclone = p.next
            if p.random is None:
                pclone.random = None
            else:
                pclone.random = p.random.next
            p = pclone.next

    def Cut(self, pHead):
        if not pHead:
            return
        p = pHead
        pcloneHead = p.next
        pclone = p.next
        while pclone.next:
            p.next = pclone.next
            pclone.next = p.next.next

            p = p.next
            pclone = p.next
        p.next = None
        return pcloneHead


"""31题，整数中1出现的次数：求出1~13的整数中1出现的次数,并算出100~1300的整数中1
       出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共
       出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,
       可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。"""
class Solution31:
    def NumberOf1Between1AndN_Solution(self, n):
        # write code here
        if n < 1:
            return 0
        i = 1
        res = 0
        while i <= n:
            a = n // i
            b = n % i
            if a % 10 == 1:
                res += a//10*i + b+1
            else:
                res += (a+8)//10*i
            i *= 10
        return res


"""34题，第一个只出现一次的字符位置：在一个字符串(0<=字符串长度<=10000，
       全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则
       返回 -1（需要区分大小写.）"""
class Solution34:
    def FirstNotRepeatingChar(self, s):
        if not s:
            return -1
        ss = list(s)
        d = {}
        for i in ss:
            if i not in d:
                d[i] = 0
            d[i] += 1

        for j in ss:
            if d[j] == 1:
                return ss.index(j)
        return -1


"""36题，两个链表的第一个公共节点"""


# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        # write code here
        if not pHead1 or not pHead2:
            return
        n1 = self.list_length(pHead1)
        n2 = self.list_length(pHead2)
        n1_n2 = n1 - n2
        if n1_n2 > 0:
            while n1_n2 > 0:
                pHead1 = pHead1.next
                n1_n2 -= 1
        else:
            while n1_n2 < 0:
                pHead2 = pHead2.next
                n1_n2 += 1
        while pHead1 is not pHead2:
            pHead1 = pHead1.next
            pHead2 = pHead2.next
        return pHead1

    def list_length(self, pHead):
        count = 0
        while pHead:
            count += 1
            pHead = pHead.next
        return count


"""49题，把字符串转化成整数：输入一个字符串,包括数字字母符号,可以为空"""
class Solution49:
    def StrToInt(self, s):
        # write code here
        d_str2int = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7,'8':8, '9':9}
        index = 0
        res = []
        if len(s) == 0:
            return 0
        while index < len(s):
            if (s[index] == '+' or s[index] == '-') and index == 0:
                index += 1
            elif s[index] in d_str2int:
                res.append(d_str2int[s[index]])
                index += 1
            else:
                return False
        res_int = 0
        for i in res:
            res_int = res_int*10 + i
        if s[0] == '-':
            return 0 - res_int
        else:
            return res_int


"""51题，构建乘积数组：给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],
    其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。"""
class Solution:
    def multiply(self, A):
        # write code here
        n = len(A)
        B = [0] * n
        for i in range(n):
            s = 1
            j = 0
            while j < n:
                if j != i:
                    s *= A[j]
                j += 1
            B[i] = s
        return B


"""52题，正则匹配：请实现一个函数用来匹配包括'.'和'*'的正则表达式。
        模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。
        在本题中，匹配是指字符串的所有字符匹配整个模式。
        例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配"""
class Solution:
    # s, pattern都是字符串
    def match(self, s, pattern):
        if not s and not pattern:
            return
        s_index, p_index = 0, 0
        return self.matchcore(s, s_index, pattern, p_index)

    def matchcore(self, s, s_index, pattern, p_index):
        slen, plen = len(s), len(pattern)
        if s_index == slen and p_index == plen:
            return True
        if s_index != slen and p_index == plen:
            return False
        if p_index+1 < plen and pattern[p_index+1] == '*':
            if s_index < slen and (s[s_index] == pattern[p_index] or pattern[p_index] == '.'):
                return self.matchcore(s, s_index+1, pattern, p_index) or self.matchcore(s, s_index, pattern, p_index+2)
            else:
                return self.matchcore(s, s_index, pattern, p_index+2)

        if s_index < slen and (s[s_index] == pattern[p_index] or pattern[p_index] == '.'):
            return self.matchcore(s, s_index+1, pattern, p_index+1)
        else:
            return False


"""53题，表示数值的字符串：请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。
        例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 
        但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。"""
# -*- coding:utf-8 -*-
class Solution:
    # s字符串
    def isNumeric(self, s):
        # write code here
        if not s:
            return
        s = s.lower()
        s_list = list(s)
        if not self.is_right_str(s_list):
            return False
        if s_list.count('e'):
            s_pre = s.split('e')[0]
            s_last = s.split('e')[1]
            return self.pre_e_isright(s_pre) and self.last_e_isright(s_last)
        return self.pre_e_isright(s_list)

    def is_right_str(self, s):
        if not s:
            return False
        count_sign = count_e = count_dot = 0
        for i in s:
            if i not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '+', '-', 'e', '.']:
                return False
            if i == '+' or i == '-':
                count_sign += 1
            elif i == 'e':
                count_e += 1
            elif i == '.':
                count_dot += 1
        if count_sign > 2 or count_dot > 1 or count_e > 1:
            return False
        return True

    def pre_e_isright(self, s):
        if not s:
            return False
        if s[0] not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '+', '-']:
            return False
        for i in s[1:-1]:
            if i not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '.']:
                return False
        if s[-1] not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
            return False
        return True

    def last_e_isright(self, s):
        if not s:
            return False
        if s[0] not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '+', '-']:
            return False
        for i in s[1:]:
            if i not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
                return False
        if s[-1] not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
            """保证最后一位一定是数字"""
            return False
        return True


"""54题，字符流中第一个不重复的字符：请实现一个函数用来找出字符流中第一个只出现一次的字符。
        例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。
        当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。"""


# -*- coding:utf-8 -*-
class Solution:
    # 返回对应char
    def __init__(self):
        self.adict = {}
        self.alist = []

    def FirstAppearingOnce(self):
        # write code here
        while len(self.alist) > 0 and self.adict[self.alist[0]] > 1:
            self.alist.pop(0)
        if len(self.alist) == 0:
            return '#'
        else:
            return self.alist[0]

    def Insert(self, char):
        # write code here
        if char not in self.adict:
            self.adict[char] = 0
            self.alist.append(char)
        self.adict[char] += 1


"""55题，链表中环的入口节点：给一个链表，若其中包含环，请找出该链表的环的入口结点，
        否则，输出null。"""


# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def EntryNodeOfLoop(self, pHead):
        # write code here
        if not self.IsLoop(pHead):
            return
        p_slow = pHead.next
        p_fast = pHead.next.next
        while p_slow is not p_fast:
            p_slow, p_fast = p_slow.next, p_fast.next.next
        p_fast = pHead
        while p_slow is not p_fast:
            p_slow, p_fast = p_slow.next, p_fast.next
        return p_slow

    def IsLoop(self, pHead):
        '''判断是否有环'''
        if not pHead or not pHead.next:
            return False
        list_node = [pHead]
        p = pHead.next
        while p and p not in list_node:
            list_node.append(p)
            p = p.next
        if not p:
            return False
        return True


"""55题，删除链表中重复的节点：在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，
       重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 
       处理后为 1->2->5"""
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def deleteDuplication(self, pHead):
        # write code here
        if not pHead:
            return
        p = pHead
        d = {}
        while p:
            if p.val not in d:
                d[p.val] = 1
            else:
                d[p.val] += 1
            p = p.next
        while pHead and d[pHead.val] > 1:
                pHead = pHead.next
        if not pHead:
            return
        p1, p2 = pHead, pHead.next
        while p2:
            if d[p2.val] > 1:
                p1.next = p2.next
                p2 = p1.next
            else:
                p1 = p2
                p2 = p1.next
        return pHead


"""63题，数据流中的中位数：如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，
那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数
就是所有数值排序之后中间两个数的平均值。我们使用Insert()方法读取数据流，
使用GetMedian()方法获取当前读取数据的中位数。"""
class Solution63:
    def __init__(self):
        self.num = []
    def Insert(self, num):
        # write code here
        self.num.append(num)
        self.num.sort()
    def GetMedian(self, fuck):
        # write code here
        n = len(self.num)
        if n == 0:
            return []
        elif n % 2 == 1:
            return self.num[n//2]
        else:
            return (self.num[n//2] + self.num[n//2-1])/2.0


