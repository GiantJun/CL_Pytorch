class iData(object):
    train_trsf = []
    strong_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None
    class_to_idx = {} # need to be initialize
    has_valid = False