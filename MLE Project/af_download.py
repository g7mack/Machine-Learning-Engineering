import airfrans as af

ROOT = r"C:\Users\...\AirfRANS\af" #where to save dataset to, change as necessary

af.dataset.download(root = ROOT, file_name = 'Dataset', unzip = True, OpenFOAM = False)