#!/bin/sh

> ~/nkodedemo01/nkode/remote_train_currentrun.py

cat ~/nkodedemo01/nkode/train_model.py >> ~/nkodedemo01/nkode/remote_train_currentrun.py

cat ~/nkodedemo01/nkode/remote_train.py >> ~/nkodedemo01/nkode/remote_train_currentrun.py


python ~/nkodedemo01/nkode/remote_train_currentrun.py

#give path to file in nkcli when you write final script