import json

labels = set()
for file in ['train']:  # , 'train', 'val']:
    with open("cnews.%s.txt" % file, "r") as f:
        lines = f.readlines()
        for l in lines[1:]:
            lbl, text = l.strip().split("\t")
            print(len(text))
            labels.add(lbl)
print(labels)
label_names = list(labels)
label_names.sort()
print(label_names)
