#!/usr/bin/env bash
# Full pipeline: download -> prepare data -> train -> evaluate on test -> example prediction.
# Run from the repository root:  bash run_all.sh
set -e
python src/download_data.py
python src/prepare_data.py
python src/train.py
python src/evaluate.py
python src/predict.py "Тауар керемет, рахмет!" "Телефон бір аптада бұзылды, қотақбас сатушы"
