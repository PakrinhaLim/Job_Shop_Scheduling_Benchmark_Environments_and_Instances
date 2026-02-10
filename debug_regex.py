
import re
import tomli

config_path = "configs/FJSP_DRL.toml"
with open(config_path, "r") as f:
    content = f.read()

updates = [
    ('problem_instance = ".*"', 'problem_instance = "/fjsp/brandimarte/Mk01.fjs"'),
    ('save_results = .*', 'save_results = false'),
    ('show_gantt = .*', 'show_gantt = false'),
    ('save_gantt = .*', 'save_gantt = false'),
    ('show_precedences = .*', 'show_precedences = false'),
    ('trained_policy = ".*"', 'trained_policy = "/saved_models/train_20240314_192906/song_10_5.pt"'),
    ('device = ".*"', 'device = "cpu"'),
]

print("Original content sample:")
print(content[:200])

for pattern, replacement in updates:
    content = re.sub(pattern, replacement, content)

print("\nModified content sample:")
print(content[:200])

try:
    tomli.loads(content)
    print("\nTOML parse SUCCESS")
except Exception as e:
    print(f"\nTOML parse FAILED: {e}")
