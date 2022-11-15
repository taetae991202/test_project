from collections import deque
import sys

d = deque()

s = sys.stdin.readline().strip()

cnt = 0
pre = ''

for current in s:
    if (current == '('):
        print(1)
        d.append(current)

    elif (pre == '(' and current == ')'):
        d.pop()
        print(2)
        cnt += len(d)

    else:
        d.pop()
        print(3)
        cnt += 1

    pre = current

print(cnt)