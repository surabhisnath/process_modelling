with open("../files/datafreqlistlog.txt", "r") as f:
    lines = f.readlines()

entries = [line.strip().split(",") for line in lines]
entries = [(k, float(v)) for k, v in entries]

# Sort by value in descending order
entries.sort(key=lambda x: x[1], reverse=True)

# Write to a new file
with open("../files/datafreqlistlog_sorted.txt", "w") as f:
    for k, v in entries:
        f.write(f"{k},{v}\n")