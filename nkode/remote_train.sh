#!/bin/sh

> remote_train_currentrun.py

cat train_model.py >> remote_train_currentrun.py

cat ~/nkodedemo01/nkode/remote_train.py >> remote_train_currentrun.py

#cat ~/nkodedemo01/nkode/remote_train.py >> ~/nkodedemo01/nkode/remote_train_currentrun.py


python remote_train_currentrun.py

#give path to file in nkcli when you write final script