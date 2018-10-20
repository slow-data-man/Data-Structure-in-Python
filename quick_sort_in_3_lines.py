def qsort(alist):
    if len(alist) <= 1:
        return alist
    return qsort([left for left in alist[1:] if left <= alist[0]]) + alist[0:1] + qsort([right for right in alist[1:] if right > alist[0]])


if __name__ == '__main__':
    alist = [2, 0, 1, 8, 1, 0, 2, 1]
    print(alist)
    print(qsort(alist))