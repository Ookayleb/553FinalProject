import pandas as pd
import re

text = "Hello have a good day"

textArr = re.findall(r'\w+', text)
for c in textArr:
	print(c)