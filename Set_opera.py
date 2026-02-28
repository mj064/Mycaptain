# Define two sets
E = set(map(int, input("Enter elements of set E (comma separated): ").split(',')))
N = set(map(int, input("Enter elements of set N (comma separated): ").split(',')))

union_set = E | N
intersection_set = E & N
difference_set = E - N
sym_diff_set = E ^ N

print(f"Union of E and N is {union_set}")
print(f"Intersection of E and N is {intersection_set}")
print(f"Difference of E and N is {difference_set}")
print(f"Symmetric difference of E and N is {sym_diff_set}")

