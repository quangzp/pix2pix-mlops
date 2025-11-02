ml-project/
├── data/
│   └── raw/                 # dữ liệu gốc (do DVC quản lý)
├── src/
│   ├── prepare.py           # tiền xử lý dữ liệu
│   ├── train.py             # huấn luyện model
│   └── evaluate.py          # đánh giá model
├── models/                  # lưu model output
├── dvc.yaml                 # pipeline DVC
├── dvc.lock
├── requirements.txt
├── README.md
└── .gitignore
