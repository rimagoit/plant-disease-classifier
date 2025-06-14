import os
import re

libs = set()
pattern = re.compile(r'^\s*(import|from)\s+([\w_]+)')

for root, _, files in os.walk("."):
    for file in files:
        if file.endswith(".py"):
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                for line in f:
                    match = pattern.match(line)
                    if match:
                        libs.add(match.group(2))

print("Used libraries:")
print("\n".join(sorted(libs)))
