# initialization
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# run
```
python train.py
```

# TODO:
- [x] add model save and reload
- [ ] add episode-utility graph
- [ ] how to handle different observation space? (If its too hard, consider fixing task but changing simulator)


# Research question
- does vanila transfer learning help
- does matching action space help
- how to do transfer learning between observation space?

- reload
- reward shaping
- representation A. Zhang, H. Satija, and J. Pineau, “Decoupling dynamics and reward for transfer learning,” arXiv preprint
arXiv:1804.10689, 2018