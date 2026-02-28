import subprocess

try:
    print("Files in main branch containing 'Tesis' or 'metodologia':")
    output = subprocess.check_output(['git', '--no-pager', 'ls-tree', '-r', 'main', '--name-only'], text=True)
    for line in output.split('\n'):
        if 'Tesis' in line or 'metodologia' in line:
            print(line)
except Exception as e:
    print(f"Error: {e}")
