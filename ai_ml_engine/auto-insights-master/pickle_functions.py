def pickle_save(name, item):
    PIK = str(name)+ ".pickle"
    with open(PIK,"wb") as f:
        pickle.dump(item, f)


def pickle_load(name):
    PIK = str(name) + ".pickle"
    with open(PIK,"rb") as f:
        temp_item = pickle.load(f)
    return temp_item
