from collections import defaultdict

dd = defaultdict(bool)

print(dd["missing"])  # False (instead of KeyError)