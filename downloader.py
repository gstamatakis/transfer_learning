from google.colab import files

with open('down1.tar.gz', 'w') as f:
    f.write('ColabTar')

files.download('down1.tar.gz')