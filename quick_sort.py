def qsort_1(alist):
    """三行实现快排，平均O(2nlogn)，生产新的list"""
    if len(alist) <= 1:
        return alist
    return qsort_1([left for left in alist[1:] if left <= alist[0]]) + alist[0:1] + qsort_1([right for right in alist[1:] if right > alist[0]])


def qsort(alist, first, last):
    """传统快排，评价O(nlogn)，原地操作"""
    if first >= last:
        return
    mid_val = alist[first]
    low, high = first, last
    while low < high:
        while low < high and alist[high] >= mid_val:
            high -= 1
        alist[low] = alist[high]

        while low < high and alist[low] < mid_val:
            low += 1
        alist[high] = alist[low]
    alist[low] = mid_val

    qsort(alist, first, low)
    qsort(alist, low+1, last)

if __name__ == '__main__':
    alist = [2, 0, 1, 8, 1, 0, 2, 1]
    print('待排序列展示：', alist)
    print('三行快排结果：', qsort_1(alist))
    qsort(alist, 0, len(alist)-1)
    print('传统快排结果：', alist)