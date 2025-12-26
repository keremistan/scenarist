import regex as re
from os import listdir, path

files1 = listdir('/')
files2 = listdir('./')
print("files1: {}".format(files1))
print("files2: {}".format(files2))

with open('./sample outputs/openai gpt-5.2 - second time') as fp:
    file_content = fp.read()
    print("file_content: {}".format(file_content))
    
    findings = re.findall('--- Reference Scene ', file_content)
    
    print("findings: {}".format(findings))