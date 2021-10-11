tp, fp, fn, tn = [int(x) for x in input().split()]

p = float(tp/(tp+fp))
r = float(tp/(tp+fn))

print(float((tp+tn)/(tp+fp+fn+tn)))
print(p)
print(round(r, 4))