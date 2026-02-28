import subprocess

try:
    print("TEX files in main branch:")
    output = subprocess.check_output(['git', '--no-pager', 'ls-tree', '-r', 'main', '--name-only'], text=True)
    for line in output.split('\n'):
        if line.endswith('.tex'):
            print(line.strip())
except Exception as e:
    print(f"Error: {e}")
