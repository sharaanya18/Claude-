import sys
import pandas as pd
from transformers import AutoModel
test = pd.read_csv(sys.argv[1])
model = AutoModel.from_pretrained("bert-base-uncased")
test.to_csv(sys.argv[2], index=False)
