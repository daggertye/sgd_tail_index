import matplotlib.pyplot as plt
import csv
import numpy as np

#maps = {0.01: 0, 0.05: 1, 0.1: 2}
maps = {128: 0, 192: 1, 256: 2}
test_accs = [[], [], []]
ph_dims = [[], [], []]
# test_accs = []
# ph_dims = []

fig, ax = plt.subplots()

# with open('dims_alexnet.txt', newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',')
#     for row in spamreader:
#         if row[0].startswith('Run: cifar10_'): continue
#         train_acc = float(row[1][-6:])
#         test_acc = float(row[2][-6:])
#         ph_dim = float(row[3][-5:])

#         meta = row[0].split('_')
#         lr = float(meta[1])
#         bs = int(meta[2])

#         if train_acc < 60: continue
#         print(train_acc, test_acc, train_acc - test_acc)
#         test_accs[maps[lr]].append(train_acc - test_acc)
#         ph_dims[maps[lr]].append(ph_dim)

with open('dims_vgg.txt', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        if row[0].startswith('Run: cifar10_'): continue
        train_acc = float(row[1][-6:])
        test_acc = float(row[2][-6:])
        ph_dim = float(row[3][-5:])

        meta = row[0].split('_')
        lr = float(meta[2])
        bs = int(meta[3])

        if train_acc < 60: continue
        print(train_acc, test_acc, train_acc - test_acc)
        test_accs[maps[bs]].append(train_acc - test_acc)
        ph_dims[maps[bs]].append(ph_dim)

plt.scatter(ph_dims[0], test_accs[0], label='128')
plt.scatter(ph_dims[1], test_accs[1], label='192')
plt.scatter(ph_dims[2], test_accs[2], label='256')
# plt.scatter(ph_dims, test_accs)
plt.xlabel("Ph Dim")
plt.ylabel("Test Gap")
ax.legend()
plt.savefig('test.png')
