with open('/home/nikos/msc-thesis/tmp/losses', 'r') as file:
    lines = file.readlines()

lines = [(line.strip()) for line in lines[-5:]]
print(lines)