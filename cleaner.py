# %%
import regex as re
from util import clean

# %%
train_dir = 'Dataset/train.txt'
val_dir = 'Dataset/val.txt'
test_dir = 'Dataset/test.txt'

train_dump = 'Dataset/train_clean.txt'
val_dump = 'Dataset/val_clean.txt'
test_dump = 'Dataset/test_clean.txt'

# %%
with open(train_dir, 'r', encoding='utf8') as f:
    train = f.read()
with open(val_dir, 'r', encoding='utf8') as f:
    val = f.read()
with open(test_dir, 'r', encoding='utf8') as f:
    test = f.read()

# %%
# # train_new = re.sub(r'\([^ء-ي]*\)', ' ', train)
# arr = re.findall(r'\([^ء-ي()]*\)', train)
# with open('Dataset/removed.txt', 'w', encoding='utf8') as f:
#     for item in arr:
#         f.write("%s\n" % item)

# %%
train_clean = clean(train)
val_clean = clean(val)
test_clean = clean(test)

# %%
non_arabic = set(re.findall(r'[^ء-يًٌٍَُِّْ\s]', train_clean))
non_arabic_test = set(re.findall(r'[^ء-يًٌٍَُِّْ\s]', test_clean))

print(non_arabic)
print(non_arabic_test)

# %%
# th = r'.{20} / .{20}'
# th = r'.{4}(?<!(\(|\)|\.|\،)) / (?!(\(|\)|\.|\،)).{4}'
# ans = [m.group() for m in re.finditer(th, train_clean)]
# for i in ans:
#     print(i)

# %%
# between_slashes = re.findall(r'/[^/]*/', train_clean)

# with open('Dataset/between.txt', 'w', encoding='utf8') as f:
#     for item in between_slashes:
#         f.write("%s\n" % item)


# %%

# train_clean = re.sub(r'\.', '.\n', train_clean)
# val_clean = re.sub(r'\.', '.\n', val_clean)

with open(train_dump, 'w', encoding='utf8') as f:
    f.write(train_clean)
with open(val_dump, 'w', encoding='utf8') as f:
    f.write(val_clean)
with open(test_dump, 'w', encoding='utf8') as f:
    f.write(test_clean)

# %%
# opening = re.findall(r'\(', train_clean)
# closing = re.findall(r'\)', train_clean)
# matched = re.findall(r'\([^()]*\)', train_clean)
# print(len(matched))
# print(len(opening))
# print(len(closing))



