#!/bin/sh

> remote_deploy_currentrun.py

cat train_model.py >> remote_deploy_currentrun.py

cat ~/nkodedemo01/nkode/remote_deploy.py >> remote_deploy_currentrun.py


python remote_deploy_currentrun.py

#give path to file in nkcli when you write final script