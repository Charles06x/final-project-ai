# Modify dataset


def modifyDataSet(ds, n):
    elevatedX = []
    for i in ds:
        elevatedX.append(list(i))
    for i in range(2, n + 1):
        for j, k in zip(ds, range(len(ds))):
            for l in range(len(j)):
                elevatedX[k].append(j[l] ** i)

    return elevatedX
