"""1题，二维数组的查找：在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都
      按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，
      判断数组中是否含有该整数。"""
class Solution1:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        rows = len(array)
        cols = len(array[0])
        i = 0
        j = cols - 1
        while i < rows and j >= 0:
            if array[i][j] == target:
                return True
            elif array[i][j] > target:
                j -= 1
            else:
                i += 1
        return False


"""6题，旋转数组的最小数字：把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组
       的旋转。输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。 例如
       数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。 NOTE：给出的所有
       元素都大于0，若数组大小为0，请返回0。"""
class Solution6:
    def minNumberInRotateArray(self, rotateArray):
        low,high = 0,len(rotateArray)-1
        if len(rotateArray) == 0:
            return 0
        if rotateArray[low] < rotateArray[high]:
            # 第一位小于最后一位，表明数列非递减，即第一位就是最小数
            return rotateArray[low]

        while low < high:  # 搜索条件,不满足该条件截止循环
            mid = (low + high) // 2 # 设置中间下标，用于比较
            if rotateArray[mid] < rotateArray[high]:
                """ 当中间值小于最后的值，表明从中间到最后非递减，
                最小的数不在此区间，故将high移动到mid"""
                high = mid
            elif rotateArray[mid] > rotateArray[high]:
                """当中间值大于最后的值，表明最小值在此区间，
                故将low移动到mid+1的位置"""
                low = mid + 1
            else:
                """当中间值等于最后的值，无法判断最小值在那个区间，
                故将high前移一位，mid也就跟着前移一位，然后继续搜索"""
                high -= 1
        """当还剩下2个数时，low == high，return rotateArray[low]也是可以的"""
        return rotateArray[low]


"""13题，调整数组顺序使奇数位于偶数前面：输入一个整数数组，实现一个函数来调整该数组中
       数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于位于数组的后半部分，
       并保证奇数和奇数，偶数和偶数之间的相对位置不变。"""
class Solution13:
    def reOrderArray(self, array):
        # write code here
        if not array:
            return []
        t1, t0 = [], []
        for i in array:
            if i % 2 == 0:
                t0.append(i)
            else:
                t1.append(i)
        return t1 + t0


"""28题，数组中出现次数超过一半的数字"""
class Solution28:
    def MoreThanHalfNum_Solution(self, numbers):
        # write code here
        if not numbers:
            return
        p = numbers[0]
        flag = 1
        for i in range(1,len(numbers)):
            if flag == 0:
                p = numbers[i]
                flag = 1
            elif numbers[i] == p:
                flag += 1
            else:
                flag -= 1
        if numbers.count(p) * 2 > len(numbers):
            return p
        return False


"""30题，连续子数组的最大和：HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。
       今天测试组开完会后,他又发话了:在古老的一维模式识别中,常常需要计算连续子
       向量的最大和,当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,
       是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},
       连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，
       返回它的最大连续子序列的和，你会不会被他忽悠住？(子向量的长度至少是1)"""
class Solution30:
    def FindGreatestSumOfSubArray(self, array):
        # write code here
        if not array:
            return
        i = 0
        n = len(array)
        while i < n and array[i] < 0:
            i += 1
        if i == n:
            return max(array)
        res_max = array[i]
        res = array[i]
        for j in range(i+1,n):
            res += array[j]
            if res < 0:
                res = 0
            res_max = max(res_max,res)
        return res_max


"""32题，把数组排成最小的数：输入一个正整数数组，把数组里所有数字拼接起来排成一个数，
        打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三
        个数字能排成的最小数字为321323。"""
class Solution32:
    def PrintMinNumber(self, numbers):
        # write code here
        if not numbers: return ""
        numbers = list(map(str, numbers))
        numbers.sort(cmp=lambda x, y: cmp(x + y, y + x))
        return "".join(numbers).lstrip('0') or'0'


"""33题，丑数：把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，
        但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的
        顺序的第N个丑数。"""
class Solution33:
    def GetUglyNumber_Solution(self, index):
        # write code here
        if index < 1:
            return 0
        ugnum = [1] * index
        cur_index = 1
        index2 = 0
        index3 = 0
        index5 = 0
        while cur_index < index:
            min_val = min(ugnum[index2]*2, ugnum[index3]*3, ugnum[index5]*5)
            ugnum[cur_index] = min_val
            # 2，3，5只能乘每个位置一次，乘完就后移一位
            while ugnum[index2]*2 <= min_val:
                index2 += 1 #后移一位
            while ugnum[index3]*3 <= min_val:
                index3 += 1
            while ugnum[index5]*5 <= min_val:
                index5 += 1
            cur_index += 1
        return ugnum[-1]


"""35题，数组中的逆序对：在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个
        数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。
        并将P对1000000007取模的结果输出。 即输出P%1000000007"""
class Solution35:
    def InversePairs_1(self, data):
        """常规方法"""
        count = 0
        i = 0
        for i in range(1, len(data)):
            if data[i - 1] > data[i]:
                count += 1
        return count % 1000000007

    def InversePairs_2(self, data):
        """ 归并排序法的应用：还未分析"""
        length = len(data)
        if data == None or length <= 0:
            return 0
        copy = data.copy()
        count = self.InversePairsCore(data, copy, 0, length - 1)
        return count

    def InversePairsCore(self, data, copy, start, end):
        """ 归并排序法的应用"""
        if start == end:
            copy[start] = data[start]
            return 0
        length = (end - start) // 2  # 分割
        left = self.InversePairsCore(copy, data, start, start + length)
        right = self.InversePairsCore(copy, data, start + length + 1, end)
        # left = self.InversePairsCore( data, copy,start, start+length) ##这里如果先定义data结果不对,多加1了
        # right = self.InversePairsCore( data, copy, start+length+1, end)

        # i初始化为前半段最后一个数字的下标
        i = start + length
        # j初始化为后半段最后一个数字的下标
        j = end
        # 书上的p3指针下标，初始为最后。即排序新数组从后往前排序
        indexCopy = end
        count = 0
        while i >= start and j >= start + length + 1:
            if data[i] > data[j]:
                copy[indexCopy] = data[i]
                indexCopy -= 1
                i -= 1
                count += j - start - length
            else:
                copy[indexCopy] = data[j]
                indexCopy -= 1
                j -= 1

        while i >= start:
            copy[indexCopy] = data[i]
            indexCopy -= 1
            i -= 1
        while j >= start + length + 1:
            copy[indexCopy] = data[j]
            indexCopy -= 1
            j -= 1
        return left + right + count

    def InversePairs_3(self, data):
        """此方法太腻害了：啊"""
        copy_data = data.copy()
        copy_data.sort()
        count = 0
        for i in range(len(copy_data)):
            count += data.index(copy_data[i])
            data.remove(copy_data[i])
        return count


"""37题，数字在排序数组中出现的次数：统计一个数字在排序数组中出现的次数。"""
class Solution37:
    def GetNumberOfK(self, data, k):
        # write code here
        if not data or len(data) == 0:
            return 0
        low = 0
        high = len(data)-1
        count = 0
        while low <= high:
            if data[low] == k:
                count += 1
            if low != high and data[high] == k:
                count += 1
            low += 1
            high -= 1
        return count


"""41题，和为S的连续正数序列：小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,
        他马上就写出了正确答案是100。但是他并不满足于此,他在想究竟有多少种连续的正数
        序列的和为100(至少包括两个数)。没多久,他就得到另一组连续正数和为100的序列:
        18,19,20,21,22。现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck!"""
class Solution41:
    def FindContinuousSequence(self, tsum):
        # write code here
        import copy as cp
        if tsum < 3:
            return []
        res = []
        res1 = []
        for i in range(1,(tsum+3)//2):
            res.append(i)
            cur_sum = sum(res)
            while cur_sum > tsum:
                res.pop(0)
                cur_sum = sum(res)
            if cur_sum == tsum:
                res1.append(cp.copy(res))  #  防止跟着变化
        return res1


"""41题，和为S的两个数字：输入一个递增排序的数组和一个数字S，在数组中查找两个数，
        使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。"""
class Solution:
    def FindNumbersWithSum(self, array, tsum):
        # write code here
        import copy as cp
        n = len(array)
        if not array or n < 3:
            return []
        res = []
        low, high = 0, n - 1
        while low < high:
            if array[low] + array[high] == tsum:
                res.append([array[low],array[high]])
                low += 1
                high -= 1
            elif array[low] + array[high] > tsum:
                high -= 1
            else:
                low += 1
        if res == []:
            return []
        rescopy = cp.copy(res)
        rescopy_fun = list(map(lambda alist: alist[0] * alist[1], rescopy))
        return res[rescopy_fun.index(min(rescopy_fun))]


"""48题，求1+2+3+...+n：要求不能使用乘除法、for、while、if、else、switch、case
等关键字及条件判断语句（A?B:C）"""
class Solution48:
    def Sum(self, n):
        return n and n + self.Sum(n-1)

"""扑克牌顺子：LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,
             2个小王(一副牌原本是54张^_^)...他随机从中抽出了5张牌,想测测自
             己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！
             “红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子.....
             LL不高兴了,他想了想,决定大\小 王可以看成任何数字,并且A看作1,
             J为11,Q为12,K为13。上面的5张牌就可以变成“1,2,3,4,5”
             (大小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。 
             现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何，
              如果牌能组成顺子就输出true，否则就输出false。为了方便起见,
              你可以认为大小王是0。"""
class Solution:
    def IsContinuous(self, numbers):
        # write code here
        if len(numbers) == 0:
            return
        numbers.sort()
        pre = 0
        while numbers[pre] == 0:
            pre += 1
        num0 = pre
        last = pre+1
        if last < len(numbers) and numbers[pre] == numbers[last]:
            return False
        gapcount = 0
        while last < len(numbers):
            gapcount += numbers[last] - numbers[pre] - 1
            pre, last = last, last + 1
        if gapcount <= num0:
            return True
        else:
            return False


"""64题，滑动窗口的最大值：给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。
        例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，
        他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有
        以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}，
        {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， 
        {2,3,4,2,6,[2,5,1]}。"""
class Solution64:
    def maxInWindows(self, num, size):
        if size < 1 or not num:
            return []
        res = []
        for i in range(len(num)-size+1):
            res.append(max(num[i:i+size]))
        return res


"""65题，矩阵中的路径：请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所
       有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，
       向右，向上，向下移动一个格子。如果一条路径经过了矩阵中的某一个格子，则之后不能
       再次进入这个格子。 例如 a b c e s f c s a d e e 这样的3 X 4 矩阵中包含一条
       字符串"bcced"的路径，但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了
       矩阵中的第一行第二个格子之后，路径不能再次进入该格子。"""
class Solution:
    def hasPath(self, matrix, rows, cols, path):
        # write code here
        matrix = self.reshape(matrix, rows, cols)
        self.row, self.col = rows, cols
        self.flag = False
        '''寻找路径入口'''
        for i in range(self.row):
            for j in range(self.col):
                if matrix[i][j] == path[0]:
                    self.search(matrix, [(i, j)], i, j, path[1:])
        return self.flag

    def search(self, matrix, dict, i, j, words):
        '''搜索路径'''
        if words == "":
            self.flag = True
            return
        if j != 0 and (i, j - 1) not in dict and matrix[i][j - 1] == words[0]:
            self.search(matrix, dict + [(i, j - 1)], i, j - 1, words[1:])
        if j != self.col - 1 and (i, j + 1) not in dict and matrix[i][j + 1] == words[0]:
            self.search(matrix, dict + [(i, j + 1)], i, j + 1, words[1:])
        if i != 0 and (i - 1, j) not in dict and matrix[i - 1][j] == words[0]:
            self.search(matrix, dict + [(i - 1, j)], i - 1, j, words[1:])
        if i != self.row - 1 and (i + 1, j) not in dict and matrix[i + 1][j] == words[0]:
            self.search(matrix, dict + [(i + 1, j)], i + 1, j, words[1:])

    def reshape(self, matrix, rows, cols):
        '''合成矩阵'''
        if not matrix or len(matrix) != rows * cols:
            return
        a = []
        c = 0
        for i in range(rows):
            c += cols
            a.append(matrix[c - cols:c])
        return a

"""66题，机器人的动态范围：地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，
       每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和
       大于k的格子。 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。
       但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？"""
class Solution:
    def movingCount(self, threshold, rows, cols):
        self.row = rows
        self.col = cols
        self.dict = set()
        self.search(threshold, 0, 0)
        return len(self.dict)

    def judy(self, threshold, i, j):
        return sum(map(int, list(str(i)))) + sum(map(int, list(str(j)))) > threshold

    def search(self, threshold, i, j):
        if self.judy(threshold, i, j) or (i, j) in self.dict:
            return
        self.dict.add((i, j))
        if i != self.row - 1:
            self.search(threshold, i + 1, j)
        if j != self.col - 1:
            self.search(threshold, i, j + 1)