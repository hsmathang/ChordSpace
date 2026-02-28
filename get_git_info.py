import subprocess
with open('git_info.txt', 'w', encoding='utf-8') as f:
    try:
        f.write("BRANCHES:\n")
        f.write(subprocess.check_output(['git', '--no-pager', 'branch', '-a']).decode('utf-8'))
        f.write("\nSTATUS:\n")
        f.write(subprocess.check_output(['git', '--no-pager', 'status']).decode('utf-8'))
    except Exception as e:
        f.write(str(e))
