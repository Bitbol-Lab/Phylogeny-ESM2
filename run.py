import os
import subprocess

if __name__ == "__main__":
    directory = 'MSA'

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            if f.endswith('.seed'):
                directory, filename = os.path.split(f)
                filename_no_extension = os.path.splitext(filename)[0]
                tree_path = 'Tree/FastTree/' + filename_no_extension + '.newick'
                # subprocess.run(["python", "main.py", "-f", f, "-m", "IQTree"])
                subprocess.run(["python", "main.py", "-f", f, "-t", tree_path])
                print("ciao")
